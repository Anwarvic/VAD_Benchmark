import torch
import warnings
warnings.filterwarnings("ignore")

from vads.vad import Vad


class Silero(Vad):
    def __init__(self,
            threshold: float = 0.9,
            window_size_ms: int = 96,
            min_speech_ms: int = 250,
            min_silence_ms: int = 100,
            speech_pad_ms: int = 30,
        ):
        super().__init__(threshold, window_size_ms)
        self._vad, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=True
        )
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._window_size_ms = window_size_ms
        self._speech_pad_ms = speech_pad_ms
        self._valid_sr = [8000, 16000]


    def _preprocess_audio(self, audio, sr):
        if sr == 8000 and self._window_size_ms > 48:
            warnings.warn(
                "window_size_ms is too big for 8000 sampling_rate! " +
                "Better set window_size_ms to 16, 32 or 48 for 8000" + 
                " sample rate!"
            )
        if self._window_size_ms not in [16, 32, 48, 64, 96]:
            warnings.warn(
                "Unusual window_size_ms! Try:\n" +
                "- [32, 64, 96] for 16000 sampling_rate\n" +
                "- [16, 32, 48] for 8000 sampling_rate"
            )
        # change audio to mono & fix sample rate
        audio, sr = super()._preprocess_audio(audio, sr)
        # convert audio to 1d array
        audio = audio.squeeze()
        return audio, sr


    def _split_to_frames(self, audio, sr):
        self._vad.reset_states()
        end_sample = len(audio)
        window_size_samples = sr * self._window_size_ms // 1000

        for start_sample in range(0, end_sample, window_size_samples):
            chunk = audio[start_sample : start_sample+window_size_samples]
            yield {
                "data": chunk,
                "start": start_sample, # in samples
                "end": len(chunk) # in samples
            }


    def _get_speech_frames(self, frames, audio, sr):
        triggered = False
        speech_frames = []
        current_speech = {}
        neg_threshold = self._threshold - 0.15
        temp_end = 0

        min_speech_samples = sr * self._min_speech_ms // 1000
        min_silence_samples = sr * self._min_silence_ms // 1000
        window_size_samples = sr * self._window_size_ms // 1000
        audio_length_samples = len(audio)
        for i, frame in enumerate(frames):
            # pad frame if shorter than window_size
            if len(frame["data"]) < window_size_samples:
                frame["data"] = torch.nn.functional.pad(
                    frame["data"],
                    (0, int(window_size_samples - len(frame["data"])))
                )
            # check if frame has speech in it
            speech_prob = self._vad(frame["data"], sr).item()
            if (speech_prob >= self._threshold) and temp_end:
                temp_end = 0

            if (speech_prob >= self._threshold) and not triggered:
                triggered = True
                current_speech['start'] = window_size_samples * i
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    dur = current_speech['end'] - current_speech['start']
                    if dur > min_speech_samples:
                        speech_frames.append(current_speech)
                    temp_end = 0
                    current_speech = {}
                    triggered = False
                    continue
        if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
            current_speech['end'] = audio_length_samples
            speech_frames.append(current_speech)
        
        speech_pad_samples = sr * self._speech_pad_ms // 1000
        for i, speech in enumerate(speech_frames):
            if i == 0:
                speech['start'] = int(
                    max(0, speech['start'] - speech_pad_samples)
                )
            if i != len(speech_frames) - 1:
                silence_duration = speech_frames[i+1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:
                    speech['end'] += int(silence_duration // 2)
                    speech_frames[i+1]['start'] = int(
                        max(
                            0,
                            speech_frames[i+1]['start'] - silence_duration//2
                        )
                    )
                else:
                    speech['end'] = int(
                        min(
                            audio_length_samples,
                            speech['end'] + speech_pad_samples
                        )
                    )
                    speech_frames[i+1]['start'] = int(
                        max(
                            0,
                            speech_frames[i+1]['start'] - speech_pad_samples
                        )
                    )
            else:
                speech['end'] = int(
                    min(
                        audio_length_samples,
                        speech['end'] + speech_pad_samples
                    )
                )
        return speech_frames


    def _merge_speech_frames(self, speech_frames, audio, sr):
        audio = torch.cat([
            audio[frame["start"]: frame["end"]]
            for frame in speech_frames
        ])
        return audio, sr


    def _postprocess_audio(self, audio, sr):
        audio, sr = super()._postprocess_audio(audio, sr)
        # convert audio to 2D array
        audio = audio.unsqueeze(0)
        return audio, sr



if __name__ == "__main__":
    from os.path import dirname, abspath, join
    from utils import load_audio, save_audio

    print("Running Silero Vad")
    samples_dir = join(dirname(dirname(abspath(__file__))), "samples")
    audio_filepath = join(samples_dir, "example_16k.wav")
    audio, sr = load_audio(audio_filepath)

    vad = Silero()
    audio, sr = vad.trim_silence(audio, sr)
    save_audio(audio, sr, join(samples_dir, "silero_example_16k.wav"))
    