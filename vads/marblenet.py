import torch
from nemo.collections.asr.models import EncDecClassificationModel

from vads.vad import Vad
from utils import (
    convert_byte_to_tensor,
    convert_tensor_to_bytes,
    convert_tensor_to_numpy,
)


class MarbleNet(Vad):
    def __init__(self,
            threshold: float = 0.4,
            window_size_ms: int = 150,
            step_size_ms: int = 10,
            model_name: str = "vad_marblenet",
            use_onnx: bool = True,
            use_cuda=False
        ):
        super().__init__(threshold, window_size_ms)
        # load vad model
        self._vad = self._load_pretrained_model(model_name, use_cuda)
        # use ONNX runtime for faster inference
        if use_onnx:
            import onnxruntime
            from tempfile import TemporaryDirectory
            # export the model to a tmp directory to be used by ONNX
            with TemporaryDirectory() as temp_dir:
                tmp_filepath = f"{temp_dir}/{model_name}.onnx"
                self._vad.export(tmp_filepath)
                self._onnx_session = onnxruntime.InferenceSession(tmp_filepath)
        else:
            self._onnx_session = None
        self._threshold = threshold
        self._window_size_ms = window_size_ms
        self._step_size_ms = step_size_ms
        self._valid_sr = [self._vad._cfg.sample_rate]
    

    def _load_pretrained_model(self, model_name, use_cuda=False):
        AVAILABLE_MODEL_NAMES = {
            model_info.pretrained_model_name
            for model_info in EncDecClassificationModel.list_available_models()
            if "vad" in model_info.pretrained_model_name
        }
        if model_name not in AVAILABLE_MODEL_NAMES:
            raise ValueError(
                f"{model_name} is not a valid VAD model name.\n" + \
                f"Available VAD model names: {AVAILABLE_MODEL_NAMES}"
            )
        # Load pre-trained model
        vad = EncDecClassificationModel.from_pretrained(
            model_name = model_name,
            map_location = (
                torch.device("cuda")
                if use_cuda and torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        # set model to inference mode
        vad.eval()
        return vad
    

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
            # change the shape to [#batches=1, #frames]
            audio_signal = buffer.unsqueeze(dim=0).to(self._vad.device)
            audio_signal_length = (
                torch.as_tensor(buffer.size()).to(self._vad.device)
            )
            # ONNX is enabled
            if self._onnx_session is not None:
                # preprocess signal
                processed_signal, _ = self._vad.preprocessor(
                    input_signal=audio_signal, length=audio_signal_length,
                )
                # calculate logits for [background, speech]
                logits = torch.from_numpy(
                    self._onnx_session.run(
                        output_names = None,
                        input_feed = {
                        'audio_signal': convert_tensor_to_numpy(processed_signal)
                        }
                    )[0]
                )[0]
            else:
                # calculate logits for [background, speech]
                logits = self._vad.forward(
                    input_signal=audio_signal,
                    input_signal_length=audio_signal_length
                )[0]
            # get probs
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
    audio_filepath = join(samples_dir, "example_48k.wav")

    vad = MarbleNet()
    audio, sr = vad.read_audio(audio_filepath)
    audio, sr = vad.trim_silence(audio, sr)
    vad.save_audio(audio, sr, join(samples_dir, "marblenet_example_48k.wav"))
    
