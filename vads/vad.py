from utils import *


class Vad:
    def __init__(self, threshold, window_size_ms):
        if not (0 < threshold < 1):
            raise ValueError(
                "Aggression threshold is a decimal value between 0 and 1."
            )
        if type(window_size_ms) != int:
            raise ValueError(
                f"Expected integer for Window Size, got {type(window_size_ms)}!"
            )
    

    def _preprocess_audio(self, audio, sr):
        # check if audio is mono
        if audio.size(0) != 1:
            audio = change_num_channels(audio, sr, 1)
        # check if sample rate is valid
        if sr not in self._valid_sr:
            tgt_sr = get_closest_sample_rate(sr, self._valid_sr)
            audio = change_sample_rate(audio, sr, tgt_sr)
            sr = tgt_sr
        return audio, sr
    

    def _postprocess_audio(self, audio, sr):
        return audio, sr
    

    def _split_to_frames(self, audio, sr):
        raise NotImplementedError
    

    def _get_speech_frames(self, frames, audio, sr):
        raise NotImplementedError


    def get_speech_boundaries(self, audio, sr):
        raise NotImplementedError
    

    def _merge_speech_frames(self, speech_frames, audio, sr):
        pass


    def trim_silence(self, audio, sr):
        orig_sr = sr
        # preprocess audio if needed
        audio, sr = self._preprocess_audio(audio, sr)
        # get frames having speech
        frames = list(self._split_to_frames(audio, sr))
        speech_frames = self._get_speech_frames(frames, audio, sr)
        # merge speech frames
        audio, sr = self._merge_speech_frames(speech_frames, audio, sr)
        # post-process audio if needed
        audio, sr = self._postprocess_audio(audio, sr)
        # change sample rate back to the original one
        if sr != orig_sr:
            audio = change_sample_rate(audio, sr, orig_sr)
        return audio, orig_sr
