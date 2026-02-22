from io import BytesIO

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_model(
    *,
    config_name,
    checkpoint_path,
    device="cuda",
    encoder: bool = True,
    decoder: bool = True,
):
    # encoder/decoder args are for API compatibility with vits.py
    # DAC always loads both since they're small
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    # vq.xxxx -> quantizer.xxxx
    state_dict = {k.replace("vq.", "quantizer."): v for k, v in state_dict.items()}

    # backbone.channel_layers. -> backbone.downsample_layers.
    state_dict = {
        k.replace("backbone.channel_layers.", "backbone.downsample_layers."): v
        for k, v in state_dict.items()
    }

    print("Error keys:", model.load_state_dict(state_dict, strict=False))
    # model.remove_parametrizations()
    model.eval()
    model.to(device)

    return model


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def batch_encode(model, audios: list[bytes | torch.Tensor]):
    audios = [
        (
            torch.from_numpy(librosa.load(BytesIO(audio), sr=model.sample_rate)[0])
            if isinstance(audio, bytes)
            else audio
        )
        for audio in audios
    ]
    lengths = torch.tensor([audio.shape[-1] for audio in audios], device=model.device)
    max_length = int(lengths.max().item())
    padded = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[-1]))
            for audio in audios
        ]
    ).to(model.device)

    padded = torch.clamp(padded, -1.0, 1.0)

    features, feature_lengths = model.encode(padded, lengths)
    features, feature_lengths = features.cpu(), feature_lengths.cpu()

    return [feature[..., :length] for feature, length in zip(features, feature_lengths)]


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def decode(model, features):
    """Decode features (indices or mels) to audio.

    For VQ models: features are indices, uses model.from_indices
    For mel models: features are mel spectrograms, uses model.decode
    """
    lengths = torch.tensor(
        [feature.shape[-1] for feature in features], device=model.device
    )
    max_length = int(lengths.max().item())
    padded = torch.stack(
        [
            torch.nn.functional.pad(feature, (0, max_length - feature.shape[-1]))
            for feature in features
        ]
    ).to(model.device)

    # Check if model has from_indices (VQ model) or decode (mel model)
    if hasattr(model, "from_indices"):
        audios = model.from_indices(padded)
        audio_lengths = lengths * model.frame_length
    else:
        audios, audio_lengths = model.decode(padded, lengths)

    audios, audio_lengths = audios.cpu(), audio_lengths.cpu()

    return [audio[..., :length].float() for audio, length in zip(audios, audio_lengths)]


if __name__ == "__main__":
    model = load_model(
        config_name="modded_dac_vq",
        checkpoint_path="checkpoints/modded-dac-msstftd-step-1380000.pth",
        device="cuda",
    )
    audios = [
        open(
            "bug.wav",
            "rb",
        ).read(),
        # open(
        #     "ElevenLabs_2025-11-18T15_11_32_Hope - upbeat and clear_pvc_sp100_s86_sb75_v3.mp3",
        #     "rb",
        # ).read(),
    ]
    features = batch_encode(model, audios)
    print(features)

    # features = [i.half() for i in features]

    # features = np.load("data/fine-grained-control/stage1/pack3.5-untar/000000267-003/pack/albumId-267-trackId-11259/0000-N.npy")
    # features1 = np.load("data/fine-grained-control/stage1/pack3.5-untar/000000267-003/pack/albumId-267-trackId-11259/0000.npy")
    # features = [torch.from_numpy(features).to(model.device), torch.from_numpy(features1).to(model.device)]

    audios = decode(model, features)
    for i, audio in enumerate(audios):
        sf.write(f"test_{i}.wav", audio.numpy().T, model.sample_rate)
