
import torch
from nemo.collections.asr.models import EncDecClassificationModel

from vads.vad import Vad
from utils import (
    convert_byte_to_tensor,
    convert_tensor_to_bytes,
)

class MarbleNet(Vad):
    def __init__(self,
            threshold=0.8,
            window_size_ms=150,
            step_size_ms=10,
        ):
        super().__init__(threshold, window_size_ms)
        self._vad = EncDecClassificationModel.from_pretrained(
            'vad_marblenet'
        )
        self._vad.preprocessor = self._vad.from_config_dict(
            self._vad._cfg.preprocessor
        )
        self._vad.eval() # set model to inference mode
        if torch.cuda.is_available():
            self._vad = self._vad.cuda()
        self._threshold = threshold
        self._step_size_ms = step_size_ms
        self._window_size_ms = window_size_ms
        self._valid_sr = [self._vad._cfg.sample_rate]
    

    def _preprocess_audio(self, audio, sr):
        # change audio to mono & fix sample rate
        audio, sr = super()._preprocess_audio(audio, sr)
        # convert audio to bytes
        audio = convert_tensor_to_bytes(audio)
        return audio, sr


    def _split_to_frames(self, audio, sr):
        offset, start_frame = 0, 0
        num_frames = int(sr * (self._step_size_ms/ 1000.0))
        n = num_frames * 2
        while offset < len(audio):
            yield {
                "data": audio[offset : offset+n],
                "start": start_frame,
                "end": start_frame+num_frames
            }
            start_frame += num_frames
            offset += n


    def _get_speech_frames(self, frames, audio, sr):
        step_size_samples = int(self._step_size_ms * sr / 1000)
        window_size_samples = int(self._window_size_ms * sr / 1000)
        # create buffer
        buffer = torch.zeros(
            size= (2 * window_size_samples + step_size_samples, ),
        )
        for frame in frames:
            signal = convert_byte_to_tensor(frame["data"]).squeeze()
            # pad shorter signals
            if len(signal) < step_size_samples:
                signal = torch.nn.functional.pad(
                    signal,
                    pad=[0, step_size_samples - len(signal)],
                    mode='constant'
                )
            older_signals = torch.clone(buffer[step_size_samples : ].data)
            buffer[ : -step_size_samples].data.copy_(older_signals)
            buffer[-step_size_samples : ].data.copy_(signal.data)
            # change the shape to [#batches, #frames] where #batches = 1
            audio_signal = buffer.unsqueeze(dim=0).to(self._vad.device)
            audio_signal_length = (
                torch.as_tensor(buffer.size()).to(self._vad.device)
            )
            logits = self._vad.forward(
                input_signal=audio_signal,
                input_signal_length=audio_signal_length
            )[0]
            # get probs, which is [background_prob, speech_prob]
            probs = torch.softmax(logits, dim=-1)
            # get speech probability
            speech_prob = probs[1].item()
            if speech_prob >= self._threshold:
                yield frame


    def _merge_speech_frames(self, speech_frames, audio, sr):
        audio_bytes = b''.join([
            frame["data"]
            for frame in speech_frames
        ])
        # convert bytes to tensor
        return convert_byte_to_tensor(audio_bytes), sr




if __name__ == "__main__":
    from os.path import dirname, abspath, join

    print("Running MarbleNet Vad")
    samples_dir = join(dirname(dirname(abspath(__file__))), "samples")
    audio_filepath = join(samples_dir, "double_48k.wav")
    audio_filepath = "VAD_demo.wav"


    vad = MarbleNet()
    audio, sr = vad.read_audio(audio_filepath)
    audio, sr = vad.trim_silence(audio, sr)
    vad.save_audio(audio, sr,  "marblenet.wav")
    # vad.save_audio(audio, sr, join(samples_dir, "marblenet_example_16k.wav"))
    