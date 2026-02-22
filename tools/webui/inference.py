"""
Test script for multi-turn TTS generation using flash_fish with Dual-AR model.

This script splits text at speaker tags, groups them into batches (every 3 speakers
or 300 UTF-8 bytes), generates audio batch by batch, accumulating context for each
subsequent batch.
"""

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path

import soundfile as sf
import torch

from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.models.text2semantic.qwen3 import generate, load_model
from fish_speech.models.dac.vqgan import batch_encode as vqgan_encode
from fish_speech.models.dac.vqgan import decode as vqgan_decode
from fish_speech.models.dac.vqgan import load_model as load_vqgan_model
from fish_speech.tokenizer import IM_END_TOKEN


def split_text_by_speaker(text: str) -> list[str]:
    """
    Split text into turns based on <|speaker:X|> tags.

    Args:
        text: The full text with speaker tags

    Returns:
        List of speaker turns, each starting with <|speaker:X|>
    """
    # Split on speaker tags, keeping the delimiter
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)
    # print(f"Text devided into SPEAKER style is {parts}")

    # Combine speaker tags with their following text
    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            # This is a speaker tag, combine with next part
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            # Skip empty or non-speaker parts at the beginning
            i += 1

    return turns


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    """
    Group turns into batches based on speaker count or byte limit.

    Args:
        turns: List of speaker turns
        max_speakers: Maximum number of speakers per batch (default 3)
        max_bytes: Maximum UTF-8 bytes per batch (default 300)

    Returns:
        List of batched text strings
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))

        # Check if adding this turn would exceed limits
        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            # Flush current batch
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    # Don't forget the last batch
    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-turn TTS generation with Dual-AR model"
    )

    parser.add_argument(
        "--model-path",
        default="checkpoints/s2_demo",
        help="Path to the TTS model checkpoint",
    )
    parser.add_argument(
        "--vqgan-config",
        default="modded_dac_vq",
        help="VQGAN config name",
    )
    parser.add_argument(
        "--vqgan-checkpoint",
        default="checkpoints/modded-dac-msstftd-step-1380000.pth",
        help="Path to VQGAN checkpoint",
    )

    parser.add_argument(
        "--prompt-text",
        action="append",
        default=[],
        help="Prompt text (use multiple times for multiple prompts)",
    )
    parser.add_argument(
        "--prompt-audio",
        action="append",
        default=[],
        help="Prompt audio file path (must match --prompt-text count)",
    )

    parser.add_argument(
        "--text",
        help="Text to generate audio for (with speaker tags)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output.wav",
        help="Output audio file path",
    )

    parser.add_argument(
        "--id-file",
        type=Path,
        help="File containing audio IDs for batch inference (one per line: lang/audio_id)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="FLEURS test dataset directory",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        help="Directory containing prompt audio files per language",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch inference",
    )
    parser.add_argument(
        "--prompt-strategy",
        choices=["random", "fixed", "gender-match"],
        default="random",
        help="Strategy for selecting prompts: random, fixed, or gender-match",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection",
    )

    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=8192)

    args = parser.parse_args()

    batch_mode = args.id_file is not None
    single_mode = args.text is not None

    if batch_mode and single_mode:
        parser.error("Cannot use both --id-file and --text")

    if not batch_mode and not single_mode:
        parser.error("Must specify either --id-file (batch) or --text (single)")

    if batch_mode:
        if not args.dataset_dir:
            parser.error("--dataset-dir required for batch mode")
        if not args.prompt_dir:
            parser.error("--prompt-dir required for batch mode")
        if not args.output_dir:
            parser.error("--output-dir required for batch mode")
    else:
        # prompt_text can be empty (pure voice cloning) or must match prompt_audio count
        if len(args.prompt_text) > 0 and len(args.prompt_text) != len(
            args.prompt_audio
        ):
            parser.error(
                f"--prompt-text count ({len(args.prompt_text)}) must match "
                f"--prompt-audio count ({len(args.prompt_audio)})"
            )

    return args


def load_sample_metadata(dataset_dir: Path, sample_id: str) -> tuple[dict, Path]:
    """Load sample metadata, supporting both FLEURS and simple formats.

    Returns: (metadata_dict, output_subpath)
    """
    parts = sample_id.split()

    if len(parts) == 3:
        lang, audio_file, json_file = parts
        json_path = dataset_dir / json_file
        output_subpath = Path(lang) / audio_file
    elif len(parts) == 1 and "/" in sample_id:
        lang, audio_id = sample_id.split("/")
        json_path = dataset_dir / lang / "audio" / "test" / f"{audio_id}.json"
        output_subpath = Path(lang) / "audio" / "test" / f"{audio_id}.wav"
    else:
        raise ValueError(f"Unsupported ID format: {sample_id}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta, output_subpath


LANG_MAP = {
    "ar": "ar_eg",
    "en": "en_us",
    "es": "es_419",
    "ja": "ja_jp",
    "pt": "pt_br",
    "zh": "cmn_hans_cn",
    "de": "de_de",
    "ko": "ko_kr",
    "ru": "ru_ru",
}


def load_prompt_pool(prompt_dir: Path, lang: str) -> list[dict]:
    mapped_lang = LANG_MAP.get(lang, lang)
    metadata_path = prompt_dir / mapped_lang / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prompt metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    for p in prompts:
        if "audio_path" not in p or not Path(p["audio_path"]).exists():
            p["audio_path"] = str(prompt_dir / mapped_lang / p["filename"])
    return prompts


def select_prompt(
    prompt_pool: list[dict],
    strategy: str,
    gender: str | None = None,
    rng: random.Random | None = None,
) -> dict:
    if not prompt_pool:
        raise ValueError("Empty prompt pool")

    if strategy == "fixed":
        return prompt_pool[0]

    if strategy == "gender-match" and gender:
        gender_upper = gender.upper()
        matching = [
            p for p in prompt_pool if p.get("gender", "").upper() == gender_upper
        ]
        if matching:
            if rng:
                return rng.choice(matching)
            return random.choice(matching)

    if rng:
        return rng.choice(prompt_pool)
    return random.choice(prompt_pool)


def build_initial_prompt(
    tokenizer,
    vqgan_model,
    input_texts: list[str],
    input_audios: list,
) -> Conversation:
    """
    Build the initial Conversation with reference audio using chat format.

    Chat format structure:
    - System message: contains reference text and audio
    - User message: text to generate (added later)
    - Assistant message: generated audio (added later)

    Args:
        tokenizer: The tokenizer
        vqgan_model: The VQGAN model for encoding audio
        input_texts: List of text prompts (e.g., "<|speaker:0|> Hello...")
        input_audios: List of audio bytes or tensors, one per text

    Returns:
        Conversation object with the initial system prompt
    """
    # Allow empty input_texts for pure voice cloning (audio-only reference)
    if len(input_texts) > 0 and len(input_texts) != len(input_audios):
        raise ValueError(
            f"input_texts ({len(input_texts)}) must match input_audios ({len(input_audios)}) or be empty"
        )

    conversation = Conversation()

    # If no audio prompts provided, return empty conversation (zero-shot generation)
    if len(input_audios) == 0:
        return conversation

    # Encode all audios to VQ codes
    features = vqgan_encode(vqgan_model, input_audios)

    # Concatenate all VQ codes
    # print(f"Feature codes shapes: {[codes.shape for codes in features]}")
    all_codes = torch.cat([codes for codes in features], dim=1)
    # torch.save(all_codes, "debug_reference_codes.pt")

    # Build system message with references
    # If input_texts is empty, use audio-only reference
    if len(input_texts) > 0:
        reference_text = "\n".join(input_texts)
        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
            TextPart(text=reference_text, cal_loss=False),
            TextPart(text="\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=all_codes, cal_loss=False),
        ]
    else:
        # Audio-only reference (no text)
        system_parts = [
            TextPart(
                text="convert the provided text to speech",
                cal_loss=False,
            ),
            TextPart(text="\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=all_codes, cal_loss=False),
        ]

    conversation.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    return conversation


class TTSGenerator:
    """TTS Generator that loads models once and can generate multiple audios."""

    def __init__(
        self,
        model_path: str,
        vqgan_config: str,
        vqgan_checkpoint: str,
        device: str = "cuda",
        max_seq_len: int = 8192,
        use_cuda_graph: bool = False,
        use_torch_compile: bool = True,
    ):
        """
        Initialize the TTS generator by loading all models.

        Args:
            model_path: Path to the TTS model checkpoint
            vqgan_config: VQGAN config name
            vqgan_checkpoint: Path to VQGAN checkpoint
            device: Device to run on ("cuda" or "cpu")
            max_seq_len: Maximum sequence length
            use_cuda_graph: Whether to use CUDA graph optimization
            use_torch_compile: Whether to use torch.compile optimization
        """
        self.device = device
        self.max_seq_len = max_seq_len
        self.use_cuda_graph = use_cuda_graph
        self.use_torch_compile = use_torch_compile
        self.num_samples = 1  # Always 1 for turn-by-turn generation

        print(f"Loading TTS model from {model_path}...")
        self.model, self.tokenizer, self.decode_one_token_fn = load_model(
            checkpoint_path=model_path,
            device=device,
            dtype=torch.bfloat16,
            max_seq_len=max_seq_len,
            max_batch_size=self.num_samples,
            use_cuda_graph=use_cuda_graph,
            use_torch_compile=use_torch_compile,
        )

        print("Loading VQGAN model...")
        self.vqgan_model = load_vqgan_model(
            config_name=vqgan_config,
            checkpoint_path=vqgan_checkpoint,
            device=device,
        )

        print("Models loaded successfully!")

    def generate(
        self,
        text: str,
        prompt_texts: list[str] | None = None,
        prompt_audios: list | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 30,
        max_new_tokens: int = 2048,
        max_speakers: int = 3,
        max_bytes: int = 300,
    ):
        """
        Generate audio from text.

        Args:
            text: Text to generate audio for (must include speaker tags)
            prompt_texts: List of prompt texts (optional)
            prompt_audios: List of prompt audio bytes/tensors (optional)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum new tokens to generate
            max_speakers: Maximum speakers per batch
            max_bytes: Maximum UTF-8 bytes per batch

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if prompt_texts is None:
            prompt_texts = []
        if prompt_audios is None:
            prompt_audios = []

        # print(
        #     f"Infer with metrics -- temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")

        # Auto-add speaker tag if not present
        if not re.search(r"<\|speaker:\d+\|>", text):
            text = f"<|speaker:0|>{text}"
            # print(f"Auto-added speaker tag: {text[:60]}...")

        # Split text into speaker turns, then group into batches
        turns = split_text_by_speaker(text)
        # print(f"Split into {len(turns)} turns")
        if len(turns) == 0:
            raise ValueError(
                "No valid speaker turns found in text. Text must contain <|speaker:X|> tags."
            )

        batches = group_turns_into_batches(turns, max_speakers, max_bytes)
        # print(f"Grouped into {len(batches)} batches:")
        for i, batch in enumerate(batches):
            print(f"  Batch {i} ({len(batch.encode('utf-8'))} bytes): {batch[:60]}...")

        # Build initial conversation with system message (chat format)
        conversation = build_initial_prompt(
            self.tokenizer,
            self.vqgan_model,
            input_texts=prompt_texts,
            input_audios=prompt_audios,
        )

        # Store all generated codes for final merging
        all_generated_codes = []

        # print(f"\nGenerating {len(batches)} batches...")
        # print("=" * 50)

        for batch_idx, batch_text in enumerate(batches):
            print(
                f"\n--- Batch {batch_idx} ({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            print(batch_text)

            # Add the current batch text as a user message
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=batch_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            # Create a copy for generation (with empty assistant message)
            conversation_gen = deepcopy(conversation)
            conversation_gen.append(
                Message(
                    role="assistant",
                    parts=[],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=False,
                )
            )

            # Encode the conversation
            encoded = conversation_gen.encode(
                self.tokenizer,
                max_length=self.max_seq_len,
                add_shift=False,
            )

            # conversation_gen.visualize(
            #     self.tokenizer,
            #     merge_audio_tokens=True,
            #     merge_semantic_tokens=True,
            #     use_color=True,
            # )

            # print(f"Input tokens shape: {encoded.tokens.shape}")

            # Prepare inputs
            input_ids = encoded.tokens.to(self.device)

            # Generate
            result = generate(
                model=self.model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                decode_one_token_fn=self.decode_one_token_fn
                if self.use_cuda_graph
                else None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_samples=self.num_samples,
                constrain_to_semantic=True,
                vq_parts=(
                    torch.cat(encoded.vq_parts, dim=1).to(self.device).mT
                    if encoded.vq_parts
                    else None
                ),
                vq_mask_tokens=(
                    encoded.vq_mask_tokens.to(self.device)
                    if encoded.vq_mask_tokens is not None
                    else None
                ),
                enable_logging=False,
            )

            # Get the single sample (num_samples=1)
            sample = result.samples[0]
            # print(f"Generated tokens shape: {sample}")
            codebook_tokens = sample.vq_parts
            vq_mask = sample.vq_mask_tokens

            # Count tokens
            num_semantic = vq_mask.sum().item()
            num_text = (~vq_mask).sum().item()
            # print(
            #     f"Generated {num_semantic} semantic tokens and {num_text} text tokens"
            # )

            if codebook_tokens is not None and codebook_tokens.numel() > 0:
                print(f"Codebook tokens shape: {codebook_tokens.shape}")
                all_generated_codes.append(codebook_tokens)

                # Add generated audio as assistant message (for context in next batch)
                conversation.append(
                    Message(
                        role="assistant",
                        parts=[VQPart(codes=codebook_tokens.mT, cal_loss=False)],
                        cal_loss=False,
                        modality="voice",
                        add_im_start=True,
                        add_im_end=True,
                    )
                )
            else:
                print("Warning: No codebook tokens generated for this batch")

        # print("\n" + "=" * 50)
        # print("All batches generated!")

        # Merge all generated codes
        if all_generated_codes:
            # (total_tokens, num_codebooks)
            merged_codes = torch.cat(all_generated_codes, dim=0)
            # print(f"Merged codes shape: {merged_codes.shape}")

            # vqgan.decode expects (batch, num_codebooks, seq_len)
            codes_for_decode = [merged_codes.mT]

            # print("Decoding audio...")
            audios = vqgan_decode(self.vqgan_model, codes_for_decode)

            # Return audio array and sample rate
            audio = audios[0].numpy().T
            return audio, self.vqgan_model.sample_rate
        else:
            raise RuntimeError("No codes were generated!")


def run_batch_inference(
    generator: TTSGenerator,
    id_file: Path,
    dataset_dir: Path,
    prompt_dir: Path,
    output_dir: Path,
    prompt_strategy: str,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
):
    rng = random.Random(seed)

    with open(id_file, "r", encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sample_ids)} sample IDs from {id_file}")

    prompt_pools: dict[str, list[dict]] = {}

    success_count = 0
    skip_count = 0
    fail_count = 0
    processed_ids = []

    for idx, sample_id in enumerate(sample_ids):
        try:
            sample_meta, output_subpath = load_sample_metadata(dataset_dir, sample_id)
        except (ValueError, FileNotFoundError) as e:
            print(f"[{idx + 1}/{len(sample_ids)}] {sample_id}: {e}")
            fail_count += 1
            continue

        lang = sample_meta.get("lang", output_subpath.parts[0])

        output_audio_path = output_dir / output_subpath
        output_json_path = output_audio_path.with_suffix(".json")

        if output_audio_path.exists() and output_json_path.exists():
            print(
                f"[{idx + 1}/{len(sample_ids)}] Skipping {sample_id} (already exists)"
            )
            skip_count += 1
            processed_ids.append(sample_id)
            continue

        if lang not in prompt_pools:
            try:
                prompt_pools[lang] = load_prompt_pool(prompt_dir, lang)
                print(f"Loaded {len(prompt_pools[lang])} prompts for {lang}")
            except FileNotFoundError as e:
                print(f"[{idx + 1}/{len(sample_ids)}] {sample_id}: {e}")
                fail_count += 1
                continue

        prompt = select_prompt(
            prompt_pools[lang],
            prompt_strategy,
            gender=sample_meta.get("gender"),
            rng=rng,
        )

        prompt_audio_path = prompt["audio_path"]
        prompt_text = f"<|speaker:0|>{prompt.get('text', '')}"
        text_with_tag = f"<|speaker:0|>{sample_meta['text']}"

        print(f"[{idx + 1}/{len(sample_ids)}] Generating {sample_id}...")
        print(f"  Prompt: {prompt['filename']} ({prompt.get('gender', 'unknown')})")

        try:
            with open(prompt_audio_path, "rb") as f:
                prompt_audio_bytes = f.read()

            audio, sample_rate = generator.generate(
                text=text_with_tag,
                prompt_texts=[prompt_text],
                prompt_audios=[prompt_audio_bytes],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )

            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_audio_path), audio, sample_rate)

            output_meta = sample_meta.copy()
            output_meta["prompt_info"] = {
                "audio": str(prompt_audio_path),
                "text": prompt.get("text", ""),
                "gender": prompt.get("gender", ""),
            }
            output_meta["duration"] = len(audio) / sample_rate

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output_meta, f, ensure_ascii=False, indent=2)

            print(f"  Saved: {output_audio_path}")
            success_count += 1
            processed_ids.append(sample_id)

        except Exception as e:
            print(f"  Error: {e}")
            fail_count += 1
            continue

    output_id_path = output_dir / "id"
    with open(output_id_path, "w", encoding="utf-8") as f:
        for sample_id in processed_ids:
            f.write(f"{sample_id}\n")

    print("\n" + "=" * 50)
    print(f"Batch inference complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Output:  {output_dir}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = TTSGenerator(
        model_path=args.model_path,
        vqgan_config=args.vqgan_config,
        vqgan_checkpoint=args.vqgan_checkpoint,
        device=device,
        max_seq_len=args.max_seq_len,
        use_cuda_graph=True,
    )

    if args.id_file:
        run_batch_inference(
            generator=generator,
            id_file=args.id_file,
            dataset_dir=args.dataset_dir,
            prompt_dir=args.prompt_dir,
            output_dir=args.output_dir,
            prompt_strategy=args.prompt_strategy,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        prompt_audios = [open(p, "rb").read() for p in args.prompt_audio]
        # If no prompt_text provided, use empty strings for pure voice cloning
        prompt_texts = args.prompt_text if args.prompt_text else None

        try:
            audio, sample_rate = generator.generate(
                text=args.text,
                prompt_texts=prompt_texts,
                prompt_audios=prompt_audios,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            )

            sf.write(args.output, audio, sample_rate)
            print(f"\nSaved audio to {args.output} (sample rate: {sample_rate})")
            print("Generation complete!")

        except Exception as e:
            print(f"\nError during generation: {e}")
            raise


if __name__ == "__main__":
    main()
