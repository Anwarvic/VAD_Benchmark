
import torch
from vads.vad import Vad
from nemo.collections.asr.models import EncDecClassificationModel

from utils import (
    convert_byte_to_numpy,
    convert_byte_to_tensor,
    convert_tensor_to_bytes,
)


class MarbleNet(Vad):
    def __init__(self, threshold, window_size_ms, frame_len=0.1):
        super().__init__(threshold, window_size_ms)
        self._frame_len = frame_len
        self._vad = EncDecClassificationModel.from_pretrained(
            'vad_marblenet'
        )
        self._vad.eval() # set model to inference mode
        if torch.cuda.is_available():
            self._vad = self._vad.cuda()
        self.vocab = list(self._vad._cfg.labels) + '_'
        self._valid_sr = [self._vad._cfg.sample_rate]
    

    def _preprocess_audio(self, audio, sr):
        # change audio to mono & fix sample rate
        audio, sr = super()._preprocess_audio(audio, sr)
        # convert audio to bytes
        audio = convert_tensor_to_bytes(audio)
        return audio, sr


    def _split_to_frames(self, audio, sr):
        offset = 0
        start_frame = 0.0
        num_frames = int(sr * (self._chunk_size_ms / 1000.0))
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
        pass
    

    def _merge_speech_frames(self, speech_frames, audio, sr):
        audio_bytes = b''.join([
            frame["data"]
            for frame in speech_frames
        ])
        # convert bytes to tensor
        return convert_byte_to_tensor(audio_bytes), sr




if __name__ == '__main__':
    import wave

    CHUNK_SIZE = int(0.025*16000)
    # first method
    blocks = []
    with wave.open("VAD_demo.wav", 'rb') as wf:
        data = wf.readframes(CHUNK_SIZE)
        while len(data) > 0:
            blocks.append(data)
            data = wf.readframes(CHUNK_SIZE)
    
    # second method
    from utils import load_audio
    my_blocks = []
    audio, sr = load_audio("VAD_demo.wav")
    audio = convert_tensor_to_bytes(audio)
    offset = 0
    n = CHUNK_SIZE * 2
    while offset < len(audio):
        my_blocks.append(audio[offset : offset+n])
        offset += n

    # verify they are the same
    assert len(blocks) == len(my_blocks)
    from tqdm import tqdm
    for b1, b2 in tqdm(zip(blocks, my_blocks)):
        assert b1 == b2
    


