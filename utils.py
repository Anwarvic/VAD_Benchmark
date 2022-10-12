import torch
import torchaudio
import numpy as np

    
def load_audio(audio_filepath):
    audio, sr = torchaudio.load(audio_filepath)
    return (audio, sr)


def save_audio(audio, sr, out_path):
    torchaudio.save(out_path, audio, sr)


def change_num_channels(audio, sr, tgt_ch):
    ch = audio.size(0)
    if tgt_ch != ch:
        effects = [["channels", str(tgt_ch)]]
        audio, _ = (
            torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
        )
    return audio


def change_sample_rate(audio, sr, tgt_sr):
    if tgt_sr != sr:
        effects = [["rate", str(tgt_sr)]]
        audio, _ = (
            torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
        )
    return audio


def get_closest_sample_rate(curr_sr, valid_sr):
    diff = [abs(curr_sr - sr) for sr in valid_sr]
    new_sr = valid_sr[diff.index(min(diff))]
    return new_sr


def convert_tensor_to_bytes(audio, dtype="int16"):
    # torch.Tensor -> np.float32 ->  np.int16(PCM_16) -> bytes
    return convert_tensor_to_pcm(audio, dtype).tobytes()


def convert_tensor_to_pcm(audio, dtype="int16"):
    # torch.Tensor -> np.float32 ->  np.int16(PCM_16)
    # src: https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    audio = np.asarray(audio)
    if audio.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (audio * abs_max + offset).clip(i.min, i.max).astype(dtype)


def convert_byte_to_tensor(audio):
    # bytes -> np.åint16(PCM_16) -> np.float32 -> torch.Tensor
    return torch.Tensor(
        np.expand_dims(convert_byte_to_numpy(audio), 0)
    )


def convert_byte_to_numpy(audio):
    # byte -> int16(PCM_16) -> np.float32
    return convert_pcm_to_numpy(
        np.frombuffer(audio, dtype=np.int16), dtype="float32"
    )


def convert_pcm_to_numpy(audio, dtype="float32"):
    # int16(PCM_16) -> np.float32
    # src: https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    audio = np.asarray(audio)
    if audio.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(audio.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (audio.astype(dtype) - offset) / abs_max


