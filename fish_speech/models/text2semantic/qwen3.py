"""
Serving utilities for FishQwen3OmniForCausalLM with CUDA graph support.

This module provides functions for efficient text generation with KV cache
and optional CUDA graph capture for decode steps. Supports both single-token
generation and multi-codebook (Dual-AR) generation for audio synthesis.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

if TYPE_CHECKING:
    from fish_speech.models.text2semantic.lora import LoraConfig

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, PreTrainedTokenizerFast

from fish_speech.content_sequence import ContentSequence, TextPart, VQPart
from fish_speech.models.text2semantic import FishQwen3OmniForCausalLM
from fish_speech.models.text2semantic.modeling import FishQwen3OmniOutput


def multinomial_with_seed(
    inputs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Samples n elements from an input tensor `inputs` of shape (n, m) using
    a unique random seed for each row. This is a deterministic batched alternative to
    `torch.multinomial`.

    Args:
        inputs: A float tensor of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
        seed:   An integer tensor of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.

    Returns:
        A tensor of shape (n, 1) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.
    """
    n, m = inputs.shape
    col_indices = torch.arange(m, device=inputs.device).unsqueeze(0)
    step_seed = (seed * 19349663) ^ (positions * 73856093)
    seed_expanded = step_seed.unsqueeze(-1)
    hashed = (seed_expanded * 8589934591) ^ (col_indices * 479001599)
    uniform_samples = (hashed % (2**24)).float() / (2**24)
    epsilon = 1e-10
    uniform_samples = uniform_samples.clamp(epsilon, 1.0 - epsilon)
    gumbel_noise = -torch.log(-torch.log(uniform_samples))
    log_probs = torch.log(inputs + epsilon)
    perturbed_log_probs = log_probs + gumbel_noise
    return torch.argmax(perturbed_log_probs, dim=1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    """Convert logits to probabilities with top-p and top-k sampling.

    Uses argsort+gather instead of scatter for deterministic mode compatibility
    during CUDA graph capture.

    Args:
        logits: Logits tensor (batch_size, vocab_size)
        temperature: Sampling temperature (batch_size, 1)
        top_p: Top-p sampling parameter (batch_size, 1)
        top_k: Top-k sampling parameter (batch_size, 1), as tensor for CUDA graph compatibility
    """
    # Sort logits first (before temperature scaling)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Compute cumulative probabilities (without temperature)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Determine which indices to remove
    sorted_indices_to_remove = cum_probs > top_p

    # Top-K sampling: use tensor comparison instead of slicing for CUDA graph compatibility
    # indices shape: (1, vocab_size), top_k shape: (batch_size, 1)
    indices = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    sorted_indices_to_remove = sorted_indices_to_remove | (indices >= top_k)

    # Always keep at least one token
    sorted_indices_to_remove[..., 0] = False

    # Apply temperature scaling to sorted logits
    sorted_logits = sorted_logits / torch.clip(temperature, min=1e-5)

    # Mask out indices and compute final probs
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, -float("Inf"))
    probs_sort = F.softmax(sorted_logits, dim=-1)

    # Map back to original indices using argsort+gather instead of scatter
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    probs = torch.gather(probs_sort, dim=-1, index=inverse_indices)
    return probs


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    positions: torch.Tensor,
    top_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample next token from logits using deterministic seed-based sampling.

    Args:
        logits: Logits tensor (batch_size, seq_len, vocab_size)
        temperature: Sampling temperature (batch_size, 1)
        top_p: Top-p sampling parameter (batch_size, 1)
        seed: Seed tensor (batch_size,) for deterministic sampling
        positions: Positions tensor (batch_size,) for deterministic sampling
        top_k: Top-k sampling parameter (batch_size, 1)

    Returns:
        Tuple of (next_token_ids, probabilities)
    """
    probs = logits_to_probs(
        logits=logits[:, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    idx_next = multinomial_with_seed(probs, seed, positions)
    return idx_next, probs


def _decode_codebooks(
    model: FishQwen3OmniForCausalLM,
    main_token: torch.Tensor,
    hidden_states: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    positions: torch.Tensor,
    top_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate codebook tokens for a main token using the audio decoder.

    This is an internal helper that handles the Dual-AR codebook generation.
    It resets the audio decoder cache, projects the hidden state, and generates
    all codebook values autoregressively.

    Args:
        model: The FishQwen3OmniForCausalLM model (must have audio_decoder)
        main_token: The sampled main token (batch_size, 1)
        hidden_states: Hidden states from text model (batch_size, seq_len, dim)
        temperature: Sampling temperature (batch_size, 1)
        top_p: Top-p sampling parameter (batch_size, 1)
        seed: Seed tensor (batch_size,) for deterministic sampling
        positions: Positions tensor (batch_size,) for deterministic sampling
        top_k: Top-k sampling parameter (batch_size, 1)

    Returns:
        Tuple of:
            - Stacked token with codebooks (batch_size, num_codebooks+1, 1)
            - Stacked logits for codebooks (batch_size, num_codebooks, 1, codebook_vocab_size)
    """
    num_codebooks = model.config.audio_decoder_config.num_codebooks
    codebook_size = model.config.audio_decoder_config.vocab_size

    codebooks = [main_token]
    codebook_logits = []

    # Reset audio decoder caches for this token
    model.audio_decoder.reset_caches()

    # Project hidden state to audio decoder dimension and start generation
    hidden_state = model.audio_decoder.project_in(hidden_states[:, -1:, :])

    # First position: use the hidden state (codebook_idx=0)
    first_decoder_logits = model.audio_decoder.forward_kvcached(hidden_state, 0)
    # Store logits for codebook 0 (batch_size, 1, codebook_size)
    codebook_logits.append(first_decoder_logits[:, :, :codebook_size])

    # Get the first codebook token (main token minus semantic offset)
    # Always decode even for non-semantic tokens (CUDA graph compatibility)
    # Clamp to valid codebook range [0, codebook_size-1]
    first_codebook = main_token - model.config.semantic_start_token_id
    first_codebook = first_codebook.clamp(0, codebook_size - 1)
    codebooks.append(first_codebook)

    # Embed and generate remaining codebooks
    hidden = model.audio_decoder.embeddings(first_codebook.squeeze(-1))

    for codebook_idx in range(1, num_codebooks):
        decoder_logits = model.audio_decoder.forward_kvcached(
            hidden.unsqueeze(1), codebook_idx
        )

        # Use codebook_idx as sub-position for seed-based sampling
        # This ensures different codebooks get different random values
        codebook_positions = positions + codebook_idx

        # Store full logits (batch_size, 1, codebook_size)
        codebook_logits.append(decoder_logits[:, :, :codebook_size])

        # Sample from codebook logits
        codebook_token, _ = sample(
            decoder_logits[:, :, :codebook_size],
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            positions=codebook_positions,
            top_k=top_k,
        )

        codebooks.append(codebook_token)
        hidden = model.audio_decoder.embeddings(codebook_token.squeeze(-1))

    # Stack all codebooks: (batch_size, num_codebooks+1, 1)
    stacked_tokens = torch.stack(codebooks, dim=1)
    # Stack all codebook logits: (batch_size, num_codebooks, 1, codebook_vocab_size)
    stacked_logits = torch.stack(
        codebook_logits, dim=1
    )  # (batch_size, num_codebooks, 1, codebook_vocab_size)

    return stacked_tokens, stacked_logits


class DecodeOneTokenOutput(NamedTuple):
    """Output from decode_one_token. Uses NamedTuple for CUDA graph compatibility."""

    tokens: torch.Tensor  # (batch_size, codebook_dim, 1)
    token_logits: torch.Tensor  # (batch_size, 1, vocab_size)
    vq_logits: (
        torch.Tensor
    )  # (batch_size, num_codebooks, 1, codebook_vocab_size) or empty tensor
    vq_mask_tokens: torch.Tensor  # (batch_size, 1) boolean mask for semantic tokens
    expert_indices: torch.Tensor  # (num_layers, batch_size * 1, top_k) or empty tensor


@dataclass
class GeneratedSample:
    """
    Output for a single generated sample.

    Attributes:
        token_ids: Main token IDs for this sample (seq_len,)
        vq_parts: Codebook values for semantic tokens (num_semantic_tokens, num_codebooks) or None
        vq_mask_tokens: Boolean mask indicating semantic tokens (seq_len,)
        token_logits: Full logits for main tokens (seq_len, vocab_size)
        vq_logits: Full logits for semantic tokens (num_semantic_tokens, num_codebooks, codebook_vocab_size) or None
        expert_indices: Expert indices (num_layers, seq_len, top_k) or None
        content_sequence: Decoded ContentSequence representation
        prompt_idx: Index of the prompt this sample was generated from (for multi-prompt generation)
    """

    token_ids: torch.Tensor
    vq_parts: Optional[torch.Tensor]
    vq_mask_tokens: torch.Tensor
    token_logits: torch.Tensor
    vq_logits: Optional[torch.Tensor]
    expert_indices: Optional[torch.Tensor]
    content_sequence: "ContentSequence"
    prompt_idx: int = 0


@dataclass
class GenerateOutput:
    """
    Output from the generate function containing all generation results.

    Attributes:
        samples: List of GeneratedSample, one per batch item
    """

    samples: List[GeneratedSample]


def create_semantic_logit_bias(
    vocab_size: int,
    semantic_token_id_start: int,
    semantic_token_id_end: int,
    im_end_id: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Create a logit bias for constrained decoding to semantic tokens only.

    The bias contains 0.0 for allowed tokens (semantic tokens + im_end_id) and
    -inf for disallowed tokens. Add this bias to logits before sampling.

    Args:
        vocab_size: Total vocabulary size
        semantic_token_id_start: Start of semantic token ID range
        semantic_token_id_end: End of semantic token ID range (inclusive)
        im_end_id: The im_end token ID to also allow
        device: Device for the bias tensor
        dtype: Data type for the bias tensor

    Returns:
        Bias tensor of shape (vocab_size,) with 0.0 for allowed, -inf for disallowed
    """
    bias = torch.full((vocab_size,), float("-inf"), device=device, dtype=dtype)
    bias[semantic_token_id_start : semantic_token_id_end + 1] = 0.0
    bias[im_end_id] = 0.0
    return bias


def _decode_dummy_codebooks(
    model: FishQwen3OmniForCausalLM,
    main_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy codebook tokens for models without audio decoder.

    For models without audio decoder, we still need consistent output format.
    Returns (batch_size, 2, 1) tokens with main token and semantic offset,
    and empty vq_logits tensor.

    Args:
        model: The model (used for config)
        main_token: The sampled main token (batch_size, 1)

    Returns:
        Tuple of (tokens, vq_logits) where tokens is (batch_size, 2, 1)
    """
    # Convert semantic token to offset, clamp for non-semantic tokens
    semantic_vocab_size = (
        model.config.semantic_end_token_id - model.config.semantic_start_token_id + 1
    )
    semantic_offset = (main_token - model.config.semantic_start_token_id).clamp(
        min=0, max=semantic_vocab_size - 1
    )

    # Stack main token and semantic offset: (batch_size, 2, 1)
    tokens = torch.stack([main_token, semantic_offset], dim=1)

    # Empty vq_logits tensor for text-only models
    vq_logits = torch.empty(0, device=main_token.device, dtype=torch.bfloat16)

    return tokens, vq_logits


RAS_WIN_SIZE = 10  # Window size for Repetition Aware Sampling
RAS_HIGH_TEMP = 1.0  # High temperature for RAS fallback
RAS_HIGH_TOP_P = 0.9  # High top_p for RAS fallback


def decode_one_token(
    model: FishQwen3OmniForCausalLM,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    semantic_logit_bias: torch.Tensor,
    top_k: torch.Tensor,
    previous_tokens: torch.Tensor,
) -> DecodeOneTokenOutput:
    """
    Decode a single token using the model with KV cache.

    If the model has an audio decoder, this function also generates codebook values
    for all tokens (always decoding codebooks for CUDA graph compatibility).

    The model config must have `semantic_start_token_id` and `semantic_end_token_id`.

    Implements Repetition Aware Sampling (RAS) for semantic tokens: if a sampled
    semantic token appears in the previous_tokens window, re-sample with higher
    temperature to encourage diversity.

    Args:
        model: The FishQwen3OmniForCausalLM model
        x: Input token IDs (batch_size, codebook_dim, 1) - consistent 3D format
        input_pos: Current position in the sequence (1,)
        temperature: Sampling temperature (batch_size, 1)
        top_p: Top-p sampling parameter (batch_size, 1)
        seed: Seed tensor (batch_size,) for deterministic sampling
        semantic_logit_bias: Pre-computed bias (vocab_size,) with 0.0 for allowed
            tokens and -inf for disallowed tokens. Use zeros for unconstrained
            decoding, or create_semantic_logit_bias() for constrained decoding.
        top_k: Top-k sampling parameter (batch_size, 1)
        previous_tokens: Previous token window for RAS (batch_size, RAS_WIN_SIZE),
            default value -100 indicates empty slot.

    Returns:
        DecodeOneTokenOutput with tokens, logits, and expert_indices

    Note:
        RAS uses model.ras_temperature and model.ras_top_p buffers registered by load_model.
    """
    has_audio_decoder = model.audio_decoder is not None

    # Get main token and VQ parts
    if x.ndim == 3:
        main_token_ids = x[:, 0, :]  # (batch_size, 1)
        vq_parts = x[:, 1:, :].squeeze(-1) if has_audio_decoder else None
    else:
        main_token_ids = x  # (batch_size, 1)
        vq_parts = None

    batch_size = main_token_ids.shape[0]

    # Compute embeddings - uses embed_one_token for models with audio decoder
    if has_audio_decoder:
        input_embeds = model.embed_one_token(main_token_ids, vq_parts)
        output = model.forward_kvcached(
            main_token_ids, input_pos, input_embeds=input_embeds
        )
    else:
        output = model.forward_kvcached(main_token_ids, input_pos)

    # Expand input_pos to batch_size for seed-based sampling
    positions = input_pos.expand(batch_size)

    # Get token logits and hidden states
    token_logits = output.token_logits
    hidden_for_codebooks = output.token_hidden_states

    # Apply constrained decoding bias (adds 0 for allowed, -inf for disallowed)
    logits_for_sample = token_logits + semantic_logit_bias

    # Sample main token (normal sampling)
    main_token_normal, _ = sample(
        logits_for_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        positions=positions,
        top_k=top_k,
    )

    # RAS: Always sample with high temp for CUDA graph compatibility
    # Use different seed offset for high-temp sampling to get different results
    # ras_temperature and ras_top_p are registered as buffers on the model by load_model
    # Slice to actual batch_size for non-CUDA-graph mode compatibility
    main_token_high_temp, _ = sample(
        logits_for_sample,
        temperature=model.ras_temperature[:batch_size],
        top_p=model.ras_top_p[:batch_size],
        seed=seed + 1,  # Different seed for diversity
        positions=positions,
        top_k=top_k,
    )

    # Determine which tokens to use based on RAS
    # Check if normal token is in the previous window (batch_size, 1) vs (batch_size, win_size)
    # main_token_normal is (batch_size, 1)
    in_window = (previous_tokens == main_token_normal).any(dim=1, keepdim=True)

    # Check if the token is semantic
    is_semantic_normal = (main_token_normal >= model.config.semantic_start_token_id) & (
        main_token_normal <= model.config.semantic_end_token_id
    )

    # Use high-temp sample if: token is semantic AND token is in window
    use_high_temp = in_window & is_semantic_normal

    # Select between normal and high-temp samples
    main_token = torch.where(use_high_temp, main_token_high_temp, main_token_normal)

    # Generate codebooks or dummy tokens
    if has_audio_decoder:
        tokens, vq_logits = _decode_codebooks(
            model,
            main_token,
            hidden_for_codebooks,
            temperature,
            top_p,
            seed=seed,
            positions=positions,
            top_k=top_k,
        )
    else:
        tokens, vq_logits = _decode_dummy_codebooks(model, main_token)

    # Check if the generated token is semantic
    is_semantic = (main_token >= model.config.semantic_start_token_id) & (
        main_token <= model.config.semantic_end_token_id
    )

    # Convert expert_indices tuple to stacked tensor, or empty tensor if None
    if output.expert_indices is not None:
        expert_indices = torch.stack(output.expert_indices, dim=0)
    else:
        expert_indices = torch.empty(0, device=x.device, dtype=torch.long)

    return DecodeOneTokenOutput(
        tokens=tokens,
        token_logits=token_logits,
        vq_logits=vq_logits,
        vq_mask_tokens=is_semantic,
        expert_indices=expert_indices,
    )


def _tokens_to_content_sequence(
    token_ids: torch.Tensor,
    vq_mask: torch.Tensor,
    vq_parts: Optional[torch.Tensor],
) -> ContentSequence:
    """
    Convert token_ids and vq_parts to a ContentSequence.

    Internal helper function to convert generated tokens into a ContentSequence
    representation for a single sample.

    Args:
        token_ids: Token IDs (seq_len,)
        vq_mask: Boolean mask indicating semantic tokens (seq_len,)
        vq_parts: Codebook values for semantic tokens (num_semantic_tokens, num_codebooks) or None

    Returns:
        ContentSequence containing TextPart and VQPart elements
    """
    parts = []
    current_text_tokens = []
    current_vq_indices = []
    semantic_idx = 0

    def flush_text():
        nonlocal current_text_tokens
        if current_text_tokens:
            parts.append(TextPart(tokens=current_text_tokens, cal_loss=True))
            current_text_tokens = []

    def flush_vq():
        nonlocal current_vq_indices
        if current_vq_indices and vq_parts is not None:
            indices = torch.tensor(
                current_vq_indices, dtype=torch.long, device=vq_parts.device
            )
            codes = vq_parts[indices].T  # (num_codebooks, num_consecutive)
            parts.append(VQPart(codes=codes.clone().cpu(), cal_loss=True))
            current_vq_indices = []

    for pos in range(len(token_ids)):
        is_semantic = vq_mask[pos].item()

        if is_semantic:
            flush_text()
            current_vq_indices.append(semantic_idx)
            semantic_idx += 1
        else:
            flush_vq()
            token_id = token_ids[pos].item()
            current_text_tokens.append(token_id)

    flush_text()
    flush_vq()

    return ContentSequence(parts=parts)


class _PrefillOutput(NamedTuple):
    """Output from _prefill_and_sample_first."""

    first_token: torch.Tensor  # (batch_size, codebook_dim, 1)
    first_is_semantic: torch.Tensor  # (batch_size, 1)
    first_vq_logits: Optional[
        torch.Tensor
    ]  # (batch_size, num_codebooks, 1, vocab) or None
    prefill_token_logits: torch.Tensor  # (batch_size, 1, vocab_size)
    prefill_expert_indices: Optional[
        torch.Tensor
    ]  # (num_layers, batch_size, T, top_k) or None
    prefill_time: float
    first_pos: torch.Tensor  # (batch_size,) - position for first decode step


def _prefill_and_sample_first(
    model: FishQwen3OmniForCausalLM,
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    num_prompts: int,
    num_samples: int,
    temperature_t: torch.Tensor,
    top_p_t: torch.Tensor,
    top_k_t: torch.Tensor,
    seed_t: torch.Tensor,
    semantic_logit_bias: torch.Tensor,
    input_lens: Optional[torch.Tensor] = None,
) -> _PrefillOutput:
    """
    Perform prefill and sample the first token for multi-prompt generation.

    This function handles both single-prompt and multi-prompt cases:
    - Single prompt (num_prompts=1): Prefill once, expand KV cache for num_samples
    - Multi prompt (num_prompts>1): Prefill all prompts, expand KV cache with interleave pattern

    Args:
        model: The FishQwen3OmniForCausalLM model
        input_ids: Input token IDs (num_prompts, seq_len) - padded
        input_embeds: Precomputed embeddings (num_prompts, seq_len, hidden_dim)
        num_prompts: Number of prompts
        num_samples: Number of samples per prompt
        temperature_t: Sampling temperature (batch_size, 1)
        top_p_t: Top-p parameter (batch_size, 1)
        top_k_t: Top-k parameter (batch_size, 1)
        seed_t: Seed tensor (batch_size,)
        semantic_logit_bias: Logit bias for constrained decoding (vocab_size,)
        input_lens: Lengths of each prompt (num_prompts,), required if num_prompts > 1

    Returns:
        _PrefillOutput containing first token, logits, and timing info
    """
    has_audio_decoder = model.audio_decoder is not None
    device = input_ids.device
    batch_size = num_prompts * num_samples

    # Determine codebook dimension
    if has_audio_decoder:
        num_codebooks = model.config.audio_decoder_config.num_codebooks
        codebook_dim = num_codebooks + 1
    else:
        codebook_dim = 2

    prefill_start_time = time.time()

    if num_prompts == 1:
        # Single prompt case: prefill once, then expand
        T = input_ids.shape[1]
        input_pos = torch.arange(0, T, device=device)
        output = model.forward_kvcached(input_ids, input_pos, input_embeds=input_embeds)
        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start_time

        # Expand KV cache if num_samples > 1
        if num_samples > 1:
            model.expand_kv_cache(num_samples, T)

            # Repeat logits and hidden_states for sampling
            output = FishQwen3OmniOutput(
                token_logits=output.token_logits.repeat(num_samples, 1, 1),
                token_hidden_states=(
                    output.token_hidden_states.repeat(num_samples, 1, 1)
                    if output.token_hidden_states is not None
                    else None
                ),
                token_weights=output.token_weights,
                codebook_logits=output.codebook_logits,
                codebook_hidden_states=output.codebook_hidden_states,
                codebook_weights=output.codebook_weights,
                router_logits=output.router_logits,
                expert_indices=(
                    tuple(ei.repeat(num_samples, 1) for ei in output.expert_indices)
                    if output.expert_indices is not None
                    else None
                ),
            )

        first_pos = torch.full((batch_size,), T, device=device, dtype=torch.long)
        prefill_seq_len = T
    else:
        # Multi-prompt case: prefill all prompts together
        assert input_lens is not None, "input_lens required for multi-prompt generation"
        T = input_ids.shape[1]  # Padded sequence length

        # Forward pass for all prompts
        input_pos = torch.arange(0, T, device=device)
        output = model.forward_kvcached(input_ids, input_pos, input_embeds=input_embeds)
        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start_time

        # Expand KV cache with interleave pattern: [0,1,2,3] -> [0,1,2,3,0,1,2,3] for num_samples=2
        if num_samples > 1:
            _expand_kv_cache_interleaved(model, num_prompts, num_samples, T)

        # Repeat output for each sample
        output = FishQwen3OmniOutput(
            token_logits=output.token_logits.repeat(num_samples, 1, 1),
            token_hidden_states=(
                output.token_hidden_states.repeat(num_samples, 1, 1)
                if output.token_hidden_states is not None
                else None
            ),
            token_weights=output.token_weights,
            codebook_logits=output.codebook_logits,
            codebook_hidden_states=output.codebook_hidden_states,
            codebook_weights=output.codebook_weights,
            router_logits=output.router_logits,
            expert_indices=(
                tuple(ei.repeat(num_samples, 1) for ei in output.expert_indices)
                if output.expert_indices is not None
                else None
            ),
        )

        # Each sample starts at its prompt's length
        # input_lens is (num_prompts,), we need (batch_size,) = (num_prompts * num_samples,)
        first_pos = input_lens.repeat(num_samples)
        prefill_seq_len = T

    # Sample first token from last position (apply bias for constrained decoding)
    # For multi-prompt, we need to gather logits at each prompt's actual last position
    if num_prompts == 1:
        logits_for_sample = output.token_logits[:, -1:] + semantic_logit_bias
        hidden_for_codebooks = output.token_hidden_states
    else:
        # Gather logits at each prompt's last token position
        # input_lens is (num_prompts,), repeated for num_samples
        gather_positions = (input_lens - 1).repeat(num_samples)  # (batch_size,)
        logits_for_sample = (
            output.token_logits[
                torch.arange(batch_size, device=device), gather_positions
            ].unsqueeze(1)
            + semantic_logit_bias
        )  # (batch_size, 1, vocab_size)

        if output.token_hidden_states is not None:
            hidden_for_codebooks = output.token_hidden_states[
                torch.arange(batch_size, device=device), gather_positions
            ].unsqueeze(
                1
            )  # (batch_size, 1, hidden_dim)
        else:
            hidden_for_codebooks = None

    first_main_token, _ = sample(
        logits_for_sample,
        temperature=temperature_t,
        top_p=top_p_t,
        seed=seed_t,
        positions=first_pos,
        top_k=top_k_t,
    )

    # Generate codebooks or dummy tokens for first token
    if not has_audio_decoder:
        first_is_semantic = (
            first_main_token >= model.config.semantic_start_token_id
        ) & (first_main_token <= model.config.semantic_end_token_id)
        semantic_vocab_size = (
            model.config.semantic_end_token_id
            - model.config.semantic_start_token_id
            + 1
        )
        first_semantic_offset = (
            first_main_token - model.config.semantic_start_token_id
        ).clamp(0, semantic_vocab_size - 1)
        first_token = torch.stack([first_main_token, first_semantic_offset], dim=1)
        first_vq_logits = None
    else:
        first_token, first_vq_logits = _decode_codebooks(
            model,
            first_main_token,
            hidden_for_codebooks,
            temperature_t,
            top_p_t,
            seed=seed_t,
            positions=first_pos,
            top_k=top_k_t,
        )
        first_is_semantic = (
            first_main_token >= model.config.semantic_start_token_id
        ) & (first_main_token <= model.config.semantic_end_token_id)

    # Prepare prefill expert indices
    if output.expert_indices is not None:
        prefill_expert_indices = torch.stack(
            [ei.view(batch_size, prefill_seq_len, -1) for ei in output.expert_indices],
            dim=0,
        )
    else:
        prefill_expert_indices = None

    # Get the logits at the last position for storage
    if num_prompts == 1:
        prefill_token_logits = output.token_logits[:, -1:, :]
    else:
        prefill_token_logits = logits_for_sample - semantic_logit_bias  # Remove bias

    # first_pos is the position for the decode loop start.
    # In the decode loop, we process cur_token (the previously generated token) at input_pos,
    # writing its KV to the cache and generating the next token.
    # The first generated token was sampled at position T (the prompt length), so the
    # decode loop should start at position T to process that token and generate the next.
    decode_start_pos = first_pos

    return _PrefillOutput(
        first_token=first_token,
        first_is_semantic=first_is_semantic,
        first_vq_logits=first_vq_logits,
        prefill_token_logits=prefill_token_logits,
        prefill_expert_indices=prefill_expert_indices,
        prefill_time=prefill_time,
        first_pos=decode_start_pos,
    )


def _expand_kv_cache_interleaved(
    model: FishQwen3OmniForCausalLM,
    num_prompts: int,
    num_samples: int,
    seq_len: int,
):
    """
    Expand KV cache with interleaved pattern for multi-prompt generation.

    For num_prompts=4, num_samples=2:
    Original slots: [0, 1, 2, 3]
    Target slots:   [0, 1, 2, 3, 0, 1, 2, 3]
    Meaning: batch indices [0,1,2,3,4,5,6,7] correspond to prompts [0,1,2,3,0,1,2,3]

    Args:
        model: The model with KV caches
        num_prompts: Number of original prompts
        num_samples: Number of samples per prompt
        seq_len: Sequence length of the prefill
    """
    batch_size = num_prompts * num_samples

    for layer in model.text_model.model.layers:
        if layer.attention.kv_cache is not None:
            kv_cache = layer.attention.kv_cache
            # Original KV cache content is in slots 0..num_prompts-1
            # We need to copy to slots num_prompts..batch_size-1 with interleave pattern

            # First, save the original content
            k_original = kv_cache.k_cache[:num_prompts, :seq_len].clone()
            v_original = kv_cache.v_cache[:num_prompts, :seq_len].clone()

            # Now fill all slots with interleaved pattern
            for sample_idx in range(num_samples):
                start_idx = sample_idx * num_prompts
                end_idx = start_idx + num_prompts
                kv_cache.k_cache[start_idx:end_idx, :seq_len].copy_(k_original)
                kv_cache.v_cache[start_idx:end_idx, :seq_len].copy_(v_original)


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: FishQwen3OmniForCausalLM,
    input_ids: torch.Tensor,
    audio_features: Optional[torch.Tensor] = None,
    audio_feature_lens: Optional[torch.Tensor] = None,
    audio_masks: Optional[torch.Tensor] = None,
    audio_feature_masks: Optional[torch.Tensor] = None,
    vq_parts: Optional[torch.Tensor] = None,
    vq_mask_tokens: Optional[torch.Tensor] = None,
    max_new_tokens: int,
    im_end_id: Optional[int] = None,
    decode_one_token_fn=None,
    num_samples: int = 1,
    early_stop_threshold: float = 1.0,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 30,
    seed: Optional[Union[int, List[int]]] = None,
    constrain_to_semantic: bool = False,
    enable_logging: bool = False,
    input_lens: Optional[torch.Tensor] = None,
) -> GenerateOutput:
    """
    Generate tokens given input_ids with optional audio features.

    Automatically handles both text-only and multi-codebook generation based on
    whether the model has an audio decoder.

    The model config must have `semantic_start_token_id` and `semantic_end_token_id`.

    Supports multi-prompt generation: if input_ids has shape (num_prompts, seq_len)
    with num_prompts > 1, the function will generate num_samples for each prompt.
    The batch size will be num_prompts * num_samples.

    Args:
        model: The FishQwen3OmniForCausalLM model
        input_ids: Input token IDs (seq_len,) or (num_prompts, seq_len)
        audio_features: Audio mel-spectrogram features (num_mel_bins, total_length)
        audio_feature_lens: Lengths of audio features (batch_size,)
        audio_masks: Boolean mask for audio positions in input_ids (batch_size, seq_len)
        audio_feature_masks: Boolean mask for audio feature positions (total_audio_tokens,)
        vq_parts: VQ codebook IDs (num_vq_tokens, num_codebooks)
        vq_mask_tokens: Boolean mask for VQ token positions (batch_size, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        im_end_id: End of message token ID (optional, uses model.im_end_id if not provided)
        decode_one_token_fn: Optional custom decode function (for CUDA graphs)
        num_samples: Number of samples to generate per prompt
        early_stop_threshold: Early stopping threshold
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter (default 30)
        seed: Random seed for deterministic sampling. Can be:
              - None: generates a random seed
              - int: used directly if batch_size=1, otherwise used to generate per-sample seeds
              - List[int]: used directly for each sample (must have len == batch_size)
        constrain_to_semantic: If True, constrain decoding to only semantic tokens
            and im_end_id. All other tokens will have -inf logits.
        enable_logging: If True, print detailed timing logs via loguru.debug including
            prefill time, decode time, and tokens per second.
        input_lens: Lengths of each prompt (num_prompts,). Required if input_ids has
            num_prompts > 1. Used to handle variable-length prompts in padded batches.

    Returns:
        GenerateOutput containing samples: list[GeneratedSample], one per batch item.
        Total samples = num_prompts * num_samples.
        Each GeneratedSample contains:
            - token_ids: Main token IDs (seq_len,)
            - vq_parts: Codebook values for semantic tokens (num_semantic_tokens, num_codebooks) or None
            - vq_mask_tokens: Boolean mask for semantic tokens (seq_len,)
            - token_logits: Full logits for main tokens (seq_len, vocab_size)
            - vq_logits: Full logits for semantic tokens (num_semantic_tokens, num_codebooks, codebook_vocab_size) or None
            - expert_indices: Expert indices (num_layers, seq_len, top_k) or None
            - content_sequence: Decoded ContentSequence representation
            - prompt_idx: Index of the prompt this sample was generated from
    """

    if decode_one_token_fn is None:
        decode_one_token_fn = decode_one_token

    # Use model.config.im_end_id if im_end_id is not provided
    if im_end_id is None:
        im_end_id = model.config.eos_token_id

    has_audio_decoder = model.audio_decoder is not None

    # Handle 1D input
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    # Also expand masks if needed
    if audio_masks is not None and audio_masks.ndim == 1:
        audio_masks = audio_masks.unsqueeze(0)
    if vq_mask_tokens is not None and vq_mask_tokens.ndim == 1:
        vq_mask_tokens = vq_mask_tokens.unsqueeze(0)

    # Determine number of prompts and validate input_lens
    num_prompts, T = input_ids.shape
    device = input_ids.device

    # Validate input_lens for multi-prompt generation
    if num_prompts > 1:
        assert (
            input_lens is not None
        ), f"input_lens is required when num_prompts > 1 (got {num_prompts} prompts)"
        assert input_lens.shape[0] == num_prompts, (
            f"input_lens.shape[0] ({input_lens.shape[0]}) must match "
            f"num_prompts ({num_prompts})"
        )

    # Final batch size = num_prompts * num_samples
    batch_size = num_prompts * num_samples
    max_seq_len = model.config.text_config.max_seq_len

    if T >= max_seq_len:
        raise ValueError(f"Input sequence length {T} exceeds max_seq_len {max_seq_len}")

    if max_new_tokens:
        if T + max_new_tokens > max_seq_len:
            max_new_tokens = max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

    # Reset caches (but don't recreate them - they should already be set up by load_model)
    # This preserves the tensor references that CUDA graph captured
    model.reset_caches()

    # Handle seed - can be None, int, or List[int]
    # batch_size = num_prompts * num_samples
    if seed is None:
        # Generate a random seed
        seed = torch.randint(0, 2**31 - 1, (1,), device=device).item()

    if isinstance(seed, list):
        # List of seeds provided - use directly
        # Must match total batch_size = num_prompts * num_samples
        assert len(seed) == batch_size, (
            f"seed list length ({len(seed)}) must match "
            f"batch_size ({batch_size} = {num_prompts} prompts x {num_samples} samples)"
        )
        seed_t = torch.tensor(seed, device=device, dtype=torch.long)
    elif batch_size == 1:
        # Single sample with int seed - use directly without sampling
        seed_t = torch.tensor([seed], device=device, dtype=torch.long)
    else:
        # Multiple samples with int seed - generate per-sample seeds deterministically
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        seed_t = torch.randint(
            0,
            2**31 - 1,
            (batch_size,),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

    # Create sampling tensors
    temperature_t = torch.tensor(
        [[temperature]], device=device, dtype=torch.bfloat16
    ).repeat(batch_size, 1)
    top_p_t = torch.tensor([[top_p]], device=device, dtype=torch.bfloat16).repeat(
        batch_size, 1
    )
    top_k_t = torch.tensor([[top_k]], device=device, dtype=torch.long).repeat(
        batch_size, 1
    )

    # Create semantic logit bias for constrained decoding
    vocab_size = model.config.text_config.vocab_size
    if constrain_to_semantic:
        semantic_logit_bias = create_semantic_logit_bias(
            vocab_size=vocab_size,
            semantic_token_id_start=model.config.semantic_start_token_id,
            semantic_token_id_end=model.config.semantic_end_token_id,
            im_end_id=im_end_id,
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        semantic_logit_bias = torch.zeros(
            vocab_size, device=device, dtype=torch.bfloat16
        )

    # Compute embeddings with audio features if provided
    input_embeds = model.embed(
        input_ids=input_ids,
        audio_features=audio_features,
        audio_feature_lens=audio_feature_lens,
        audio_masks=audio_masks,
        audio_feature_masks=audio_feature_masks,
        vq_parts=vq_parts,
        vq_mask_tokens=vq_mask_tokens,
    )

    # Prefill and sample first token
    prefill_output = _prefill_and_sample_first(
        model=model,
        input_ids=input_ids,
        input_embeds=input_embeds,
        num_prompts=num_prompts,
        num_samples=num_samples,
        temperature_t=temperature_t,
        top_p_t=top_p_t,
        top_k_t=top_k_t,
        seed_t=seed_t,
        semantic_logit_bias=semantic_logit_bias,
        input_lens=input_lens,
    )

    first_token = prefill_output.first_token
    first_is_semantic = prefill_output.first_is_semantic
    first_vq_logits = prefill_output.first_vq_logits
    prefill_token_logits = prefill_output.prefill_token_logits
    prefill_time = prefill_output.prefill_time

    if enable_logging:
        total_prefill_tokens = T * num_prompts
        logger.debug(
            f"Prefill: {total_prefill_tokens} tokens ({num_prompts} prompts x {T}) "
            f"in {prefill_time * 1000:.2f}ms "
            f"({total_prefill_tokens / prefill_time:.2f} tokens/s)"
        )

    # Lists to collect results
    all_token_ids = []
    all_vq_parts = []  # Always track vq_parts (semantic offsets for no-audio-decoder)
    all_vq_mask_tokens = []
    all_token_logits = []
    all_vq_logits = [] if has_audio_decoder else None
    # expert_indices: list of tensors, shape (num_layers, batch_size, 1, top_k)
    all_expert_indices = (
        [] if prefill_output.prefill_expert_indices is not None else None
    )

    # Determine codebook dimension
    if has_audio_decoder:
        num_codebooks = model.config.audio_decoder_config.num_codebooks
        codebook_dim = num_codebooks + 1
    else:
        # For no-audio-decoder: [0] = main token, [1] = semantic offset
        codebook_dim = 2

    # Store first token results
    all_token_ids.append(first_token[:, 0, :])  # (batch_size, 1)
    all_token_logits.append(prefill_token_logits)  # (batch_size, 1, vocab_size)
    all_vq_mask_tokens.append(first_is_semantic)  # (batch_size, 1)

    # Store vq_parts: codebooks for audio decoder, semantic offset for no-audio-decoder
    all_vq_parts.append(first_token[:, 1:, :])  # (batch_size, num_codebooks or 1, 1)
    if has_audio_decoder:
        all_vq_logits.append(
            first_vq_logits
        )  # (batch_size, num_codebooks, 1, codebook_vocab_size)

    # Store prefill expert indices (for the entire prompt)
    if (
        all_expert_indices is not None
        and prefill_output.prefill_expert_indices is not None
    ):
        all_expert_indices.append(prefill_output.prefill_expert_indices)

    # Track if finished and sequence lengths
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    finished = finished | (first_token[:, 0, -1] == im_end_id)
    # Track sequence length for each sample (where EOS was hit, or max length)
    seq_lens = torch.full(
        (batch_size,), max_new_tokens, device=device, dtype=torch.long
    )
    # If first token is EOS, set seq_len to 1
    seq_lens = torch.where(finished, torch.ones_like(seq_lens), seq_lens)

    # Initialize previous_tokens buffer for RAS (Repetition Aware Sampling)
    # Shape: (batch_size, RAS_WIN_SIZE), default -100 indicates empty slot
    previous_tokens = torch.full(
        (batch_size, RAS_WIN_SIZE), -100, device=device, dtype=torch.long
    )
    # Add first token to the buffer (at position 0)
    previous_tokens[:, 0] = first_token[:, 0, 0]

    # Decode remaining tokens
    cur_token = first_token
    # input_pos must be (seq_len,) for forward_kvcached, not (batch_size,)
    # For uniform prompt lengths (all prompts have same length), we can use a scalar
    # For variable lengths, we'd need more complex handling (not supported yet)
    # Take first element since all should be the same for uniform-length prompts
    input_pos = prefill_output.first_pos[:1]  # (1,)

    start_time = time.time()

    for i in range(max_new_tokens - 1):
        if finished.all() or (
            0 < early_stop_threshold < 1
            and finished.sum() >= round(batch_size * early_stop_threshold)
        ):
            break

        decode_output = decode_one_token_fn(
            model=model,
            x=cur_token,
            input_pos=input_pos,
            temperature=temperature_t,
            top_p=top_p_t,
            seed=seed_t,
            semantic_logit_bias=semantic_logit_bias,
            top_k=top_k_t,
            previous_tokens=previous_tokens,
        )

        next_token = decode_output.tokens.clone()
        input_pos = input_pos + 1

        # Always use consistent 3D format
        cur_token = next_token.view(batch_size, codebook_dim, 1)

        # Update previous_tokens buffer for RAS: roll left and add new token at the end
        previous_tokens = previous_tokens.roll(-1, dims=1)
        previous_tokens[:, -1] = cur_token[:, 0, 0]

        # Store results - clone tensors to avoid CUDA graph static tensor overwrites
        all_token_ids.append(cur_token[:, 0, :].clone())  # (batch_size, 1)
        all_token_logits.append(
            decode_output.token_logits.clone()
        )  # (batch_size, 1, vocab_size)
        all_vq_mask_tokens.append(
            decode_output.vq_mask_tokens.clone()
        )  # (batch_size, 1)

        # Store vq_parts: codebooks for audio decoder, semantic offset for no-audio-decoder
        all_vq_parts.append(
            cur_token[:, 1:, :].clone()
        )  # (batch_size, num_codebooks or 1, 1)
        if has_audio_decoder:
            all_vq_logits.append(
                decode_output.vq_logits.clone()
            )  # (batch_size, num_codebooks, 1, codebook_vocab_size)

        # Store decode expert indices - clone to avoid CUDA graph static tensor overwrites
        # decode_output.expert_indices is (num_layers, batch_size * 1, top_k) or empty
        if all_expert_indices is not None and decode_output.expert_indices.numel() > 0:
            # Reshape from (num_layers, batch_size * 1, top_k) to (num_layers, batch_size, 1, top_k)
            num_layers = decode_output.expert_indices.shape[0]
            top_k = decode_output.expert_indices.shape[2]
            decode_ei = decode_output.expert_indices.view(
                num_layers, batch_size, 1, top_k
            ).clone()
            all_expert_indices.append(decode_ei)

        # Track newly finished samples and their sequence lengths
        newly_finished = ~finished & (cur_token[:, 0, -1] == im_end_id)
        # i + 2 because: i is 0-indexed for remaining tokens, +1 for first token, +1 for current token
        seq_lens = torch.where(
            newly_finished, torch.full_like(seq_lens, i + 2), seq_lens
        )
        finished = finished | newly_finished

    torch.cuda.synchronize()
    decode_time = time.time() - start_time
    generated_tokens = len(all_token_ids)
    decode_tokens_per_second = (
        (generated_tokens / decode_time) * batch_size if decode_time > 0 else 0
    )

    if enable_logging:
        codebook_str = " with codebooks" if has_audio_decoder else ""
        total_time = prefill_time + decode_time
        total_tokens = T + generated_tokens
        total_tokens_per_second = (
            (total_tokens / total_time) * batch_size if total_time > 0 else 0
        )
        logger.debug(
            f"Decode: {generated_tokens} x {batch_size} tokens{codebook_str} in {decode_time * 1000:.2f}ms "
            f"({decode_tokens_per_second:.2f} tokens/s)"
        )

    # Concatenate all results
    token_ids = torch.cat(all_token_ids, dim=1)  # (batch_size, seq_len)
    token_logits = torch.cat(
        all_token_logits, dim=1
    )  # (batch_size, seq_len, vocab_size)
    vq_mask_tokens_out = torch.cat(all_vq_mask_tokens, dim=1)  # (batch_size, seq_len)

    # Always concatenate vq_parts (codebooks for audio decoder, semantic offsets for no-audio-decoder)
    vq_parts_full = torch.cat(
        all_vq_parts, dim=2
    )  # (batch_size, num_codebooks or 1, seq_len)

    if has_audio_decoder:
        vq_logits_full = torch.cat(
            all_vq_logits, dim=2
        )  # (batch_size, num_codebooks, seq_len, codebook_vocab_size)
    else:
        vq_logits_full = None

    # Combine expert_indices: (num_layers, batch_size, total_seq_len, top_k)
    expert_indices_combined = None
    if all_expert_indices is not None and len(all_expert_indices) > 0:
        expert_indices_combined = torch.cat(
            all_expert_indices, dim=2
        )  # (num_layers, batch_size, total_seq_len, top_k)

    # Build GeneratedSample for each batch item
    samples = []
    for b in range(batch_size):
        # Truncate to actual sequence length (where EOS was hit)
        sample_seq_len = seq_lens[b].item()
        sample_token_ids = token_ids[b, :sample_seq_len]  # (seq_len,)
        sample_vq_mask = vq_mask_tokens_out[b, :sample_seq_len]  # (seq_len,)
        sample_token_logits = token_logits[b, :sample_seq_len]  # (seq_len, vocab_size)

        # Extract VQ parts for this sample (works for both audio decoder and no-audio-decoder)
        # vq_parts_full is (batch_size, num_codebooks or 1, seq_len)
        # We need to extract only semantic tokens for this sample
        sample_vq_parts_full = vq_parts_full[b, :, :sample_seq_len].permute(
            1, 0
        )  # (seq_len, num_codebooks or 1)
        sample_vq_parts = sample_vq_parts_full[
            sample_vq_mask
        ]  # (num_semantic, num_codebooks or 1)

        # Extract VQ logits (only for audio decoder)
        if has_audio_decoder and vq_logits_full is not None:
            sample_vq_logits_full = vq_logits_full[b, :, :sample_seq_len].permute(
                1, 0, 2
            )  # (seq_len, num_codebooks, vocab)
            sample_vq_logits = sample_vq_logits_full[
                sample_vq_mask
            ]  # (num_semantic, num_codebooks, vocab)
        else:
            sample_vq_logits = None

        # Extract expert indices for this sample
        if expert_indices_combined is not None:
            sample_expert_indices = expert_indices_combined[
                :, b, : sample_seq_len + T, :
            ]  # (num_layers, seq_len, top_k)
        else:
            sample_expert_indices = None

        # Build content sequence
        content_seq = _tokens_to_content_sequence(
            sample_token_ids, sample_vq_mask, sample_vq_parts
        )

        # Compute prompt_idx: batch layout is interleaved
        # For num_prompts=4, num_samples=2: indices [0,1,2,3,4,5,6,7] -> prompts [0,1,2,3,0,1,2,3]
        prompt_idx = b % num_prompts

        samples.append(
            GeneratedSample(
                token_ids=sample_token_ids.clone().cpu(),
                vq_parts=(
                    sample_vq_parts.clone().cpu()
                    if sample_vq_parts is not None
                    else None
                ),
                vq_mask_tokens=sample_vq_mask.clone().cpu(),
                token_logits=sample_token_logits.clone().cpu(),
                vq_logits=(
                    sample_vq_logits.clone().cpu()
                    if sample_vq_logits is not None
                    else None
                ),
                expert_indices=(
                    sample_expert_indices.clone().cpu()
                    if sample_expert_indices is not None
                    else None
                ),
                content_sequence=content_seq,
                prompt_idx=prompt_idx,
            )
        )

    return GenerateOutput(samples=samples)


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    use_cuda_graph: bool = True,
    disable_audio_decoder: bool = False,
    use_torch_compile: bool = False,
    lora_config: Optional["LoraConfig"] = None,
) -> Tuple[FishQwen3OmniForCausalLM, PreTrainedTokenizerFast, callable]:
    """
    Load a FishQwen3OmniForCausalLM model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        dtype: Data type for the model
        max_seq_len: Maximum sequence length for KV cache
        max_batch_size: Maximum batch size for KV cache
        use_cuda_graph: Whether to capture decode step into CUDA graph
        disable_audio_decoder: If True, set audio_decoder to None after loading
            (useful for text-only inference with models that have audio decoders)
        use_torch_compile: If True, compile the model with torch.compile()
        lora_config: Optional LoRA configuration. If provided, applies LoRA to the model
            before CUDA graph capture. This is required for training with LoRA.
        top_k: Top-k sampling parameter for CUDA graph capture (default 30)

    Returns:
        Tuple of (model, tokenizer, decode_one_token_fn)
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Use AutoModel to support different model types
    model = AutoModel.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Optionally disable audio decoder for text-only inference
    if disable_audio_decoder and model.audio_decoder is not None:
        logger.info("Disabling audio decoder for text-only inference")
        model.audio_decoder = None

    # Setup KV caches
    model.setup_caches(max_batch_size, max_seq_len, dtype)

    # Register RAS (Repetition Aware Sampling) buffers on the model
    # These are constant tensors used for high-temp fallback sampling
    model.register_buffer(
        "ras_temperature",
        torch.full((max_batch_size, 1), RAS_HIGH_TEMP, device=device, dtype=dtype),
    )
    model.register_buffer(
        "ras_top_p",
        torch.full((max_batch_size, 1), RAS_HIGH_TOP_P, device=device, dtype=dtype),
    )

    # Apply LoRA if config provided (MUST be before CUDA graph capture)
    if lora_config is not None:
        from flash_fish.models.lora import setup_lora

        setup_lora(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        logger.info(
            f"LoRA setup complete: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
            f"trainable={trainable_params / 1e6:.2f}M, frozen={frozen_params / 1e9:.2f}B"
        )

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_path)

    # Log semantic token ID range from config
    config = model.config
    semantic_start = config.semantic_start_token_id
    semantic_end = config.semantic_end_token_id
    logger.info(f"Semantic token ID range: {semantic_start} - {semantic_end}")

    logger.info(f"Model loaded successfully on {device}")

    # Create CUDA graph decode function if requested
    if use_torch_compile and torch.cuda.is_available():
        logger.info("Compiling model with torch.compile()")
        decode_one_token_fn = torch.compile(
            decode_one_token, mode="max-autotune", fullgraph=True
        )
    elif use_cuda_graph and torch.cuda.is_available():
        decode_one_token_fn = create_cuda_graph_decode_fn(
            model=model,
            batch_size=max_batch_size,
            device=device,
            dtype=dtype,
        )
    else:
        decode_one_token_fn = decode_one_token

    return model, tokenizer, decode_one_token_fn


@dataclass
class CUDAGraphRunner:
    """
    Helper class to capture and replay CUDA graphs for decode steps.

    This captures the decode_one_token function into a CUDA graph for
    efficient replay during generation. Handles both text-only and
    audio decoder (multi-codebook) cases.

    The model config must have `semantic_start_token_id` and `semantic_end_token_id`.
    """

    model: FishQwen3OmniForCausalLM
    graph: torch.cuda.CUDAGraph
    has_audio_decoder: bool
    static_input_ids: torch.Tensor
    static_input_pos: torch.Tensor
    static_temperature: torch.Tensor
    static_top_p: torch.Tensor
    static_top_k: torch.Tensor
    static_seed: torch.Tensor
    static_semantic_logit_bias: torch.Tensor
    static_previous_tokens: torch.Tensor  # For RAS (Repetition Aware Sampling)
    static_output: DecodeOneTokenOutput  # The output NamedTuple from the captured graph

    @classmethod
    def capture(
        cls,
        model: FishQwen3OmniForCausalLM,
        batch_size: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_warmup: int = 3,
    ) -> "CUDAGraphRunner":
        """
        Capture decode_one_token into a CUDA graph.

        Args:
            model: The model to capture (config must have semantic_start_token_id/end)
            batch_size: Batch size for the graph
            device: Device for the graph
            dtype: Data type for tensors
            num_warmup: Number of warmup iterations before capture
            top_k: Top-k sampling parameter to use in captured graph (default 30)

        Returns:
            CUDAGraphRunner instance with captured graph
        """
        has_audio_decoder = model.audio_decoder is not None

        # Determine codebook dimension
        if has_audio_decoder:
            num_codebooks = model.config.audio_decoder_config.num_codebooks
            codebook_dim = num_codebooks + 1
        else:
            # Maincodebook + dummy semantic offset for text-only / no-audio-decoder
            codebook_dim = 2

        # Create static buffers with consistent 3D format: (batch_size, codebook_dim, seq_len)
        static_input_ids = torch.zeros(
            (batch_size, codebook_dim, 1), device=device, dtype=torch.long
        )

        static_input_pos = torch.zeros(1, device=device, dtype=torch.long)
        static_temperature = (
            torch.ones((batch_size, 1), device=device, dtype=dtype) * 0.7
        )
        static_top_p = torch.ones((batch_size, 1), device=device, dtype=dtype) * 0.9
        static_top_k = torch.full((batch_size, 1), 30, device=device, dtype=torch.long)
        static_seed = torch.zeros(batch_size, device=device, dtype=torch.long)

        # Create static buffer for semantic logit bias (zeros = no constraint)
        vocab_size = model.config.text_config.vocab_size
        static_semantic_logit_bias = torch.zeros(vocab_size, device=device, dtype=dtype)

        # Create static buffer for previous_tokens (RAS)
        # Shape: (batch_size, RAS_WIN_SIZE), default -100 indicates empty slot
        static_previous_tokens = torch.full(
            (batch_size, RAS_WIN_SIZE), -100, device=device, dtype=torch.long
        )

        # Warmup runs - call full decode_one_token to warm up all code paths
        # semantic_start_token_id/end are in config, audio_decoder.input_pos is registered on the model
        logger.info(f"Running {num_warmup} warmup iterations for CUDA graph capture")
        for _ in range(num_warmup):
            _ = decode_one_token(
                model=model,
                x=static_input_ids,
                input_pos=static_input_pos,
                temperature=static_temperature,
                top_p=static_top_p,
                seed=static_seed,
                semantic_logit_bias=static_semantic_logit_bias,
                top_k=static_top_k,
                previous_tokens=static_previous_tokens,
            )
        torch.cuda.synchronize()

        # Capture the graph
        logger.info(f"Capturing CUDA graph (has_audio_decoder={has_audio_decoder})")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = decode_one_token(
                model=model,
                x=static_input_ids,
                input_pos=static_input_pos,
                temperature=static_temperature,
                top_p=static_top_p,
                seed=static_seed,
                semantic_logit_bias=static_semantic_logit_bias,
                top_k=static_top_k,
                previous_tokens=static_previous_tokens,
            )
        torch.cuda.synchronize()

        logger.info("CUDA graph captured successfully")

        return cls(
            model=model,
            graph=graph,
            has_audio_decoder=has_audio_decoder,
            static_input_ids=static_input_ids,
            static_input_pos=static_input_pos,
            static_temperature=static_temperature,
            static_top_p=static_top_p,
            static_top_k=static_top_k,
            static_seed=static_seed,
            static_semantic_logit_bias=static_semantic_logit_bias,
            static_previous_tokens=static_previous_tokens,
            static_output=static_output,
        )

    def decode_one_token(
        self,
        model: FishQwen3OmniForCausalLM,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        seed: torch.Tensor,
        semantic_logit_bias: torch.Tensor,
        top_k: torch.Tensor,
        previous_tokens: torch.Tensor,
    ) -> DecodeOneTokenOutput:
        """
        Decode one token using the captured CUDA graph.

        Args:
            model: The model (ignored, uses captured model)
            x: Input token IDs (batch_size, codebook_dim, 1) - consistent 3D format
            input_pos: Current position (1,)
            temperature: Sampling temperature (batch_size, 1)
            top_p: Top-p parameter (batch_size, 1)
            seed: Seed tensor (batch_size,) for deterministic sampling
            semantic_logit_bias: Logit bias (vocab_size,) for constrained decoding
            top_k: Top-k parameter (batch_size, 1)
            previous_tokens: Previous token window for RAS (batch_size, RAS_WIN_SIZE)

        Returns:
            DecodeOneTokenOutput with tokens, logits, and expert_indices
        """
        # Copy inputs to static buffers
        self.static_input_ids.copy_(x)
        self.static_input_pos.copy_(input_pos)
        self.static_temperature.copy_(temperature)
        self.static_top_p.copy_(top_p)
        self.static_top_k.copy_(top_k)
        self.static_seed.copy_(seed)
        self.static_semantic_logit_bias.copy_(semantic_logit_bias)
        self.static_previous_tokens.copy_(previous_tokens)

        # Replay the graph
        self.graph.replay()

        # Return the static output (tensors inside the NamedTuple were updated by graph replay)
        return self.static_output


def create_cuda_graph_decode_fn(
    model: FishQwen3OmniForCausalLM,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 3,
):
    """
    Create a CUDA graph-accelerated decode function.

    This function performs the following:
    1. Runs warmup iterations with the full decode_one_token
    2. Captures the decode step into a CUDA graph
    3. Returns a function that can be passed to decode_n_tokens/generate

    Handles both text-only and audio decoder (multi-codebook) cases automatically.

    Args:
        model: The model to capture
        batch_size: Batch size
        device: Device for the graph
        dtype: Data type
        num_warmup: Number of warmup iterations

    Returns:
        A decode function compatible with decode_n_tokens
    """
    runner = CUDAGraphRunner.capture(
        model=model,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        num_warmup=num_warmup,
    )
    return runner.decode_one_token
