from vads.vad import Vad
from auditok import (
    AudioReader,
    StreamTokenizer,
    AudioEnergyValidator,
)

from utils import (
    convert_tensor_to_bytes,
    convert_byte_to_tensor,
)


class Auditok(Vad):
    def __init__(self,
            threshold=0.55,
            window_size_ms=50,
            min_speech_ms=200, # minimum valid speech event
            max_speech_ms=4000, # maximum valid speech event
            max_silence_ms=300, # maximum tolerated continuous silence
            eps=1e-10, # very small for stability
        ):
        super().__init__(threshold, window_size_ms)
        self._window_size_ms = window_size_ms
        self._sample_width = 2 # 16 bit
        self._channels = 1 # mono
        # create vad
        vad = AudioEnergyValidator(
            energy_threshold=threshold*100,
            sample_width=self._sample_width, 
            channels=self._channels,
            use_channel=None
        )
        # create tokenizer
        self._tokenizer = StreamTokenizer(
            validator=vad,
            min_length=int(min_speech_ms / window_size_ms + eps), #4
            max_length=int(max_speech_ms / window_size_ms + eps), #80
            max_continuous_silence=int(max_silence_ms / window_size_ms + eps), #5
            mode=0,
        )
        self._valid_sr = [8000, 16000, 32000, 48000]
    

    def _preprocess_audio(self, audio, sr):
        # change audio to mono & fix sample rate
        audio, sr = super()._preprocess_audio(audio, sr)
        # convert audio to bytes
        audio = convert_tensor_to_bytes(audio)
        return audio, sr
    

    def _split_to_frames(self, audio, sr):
        # convert audio to AudioReader
        audio_reader = AudioReader(
            input=bytes(audio),
            block_dur=self._window_size_ms/1000,
            sampling_rate=sr,
            sample_width=self._sample_width,
            channels=self._channels
        )
        audio_reader.open()
        token_gen = self._tokenizer.tokenize(audio_reader, generator=True)
        for token in token_gen:
            data = b"".join(token[0])
            start_frame = token[1]
            frame_duration = self._window_size_ms/1000
            start = start_frame * frame_duration
            duration = len(data)/(sr * self._sample_width * self._channels)
            yield {
                "data": data,
                "start": start,
                "end": start+duration
            }
        # audio_reader.close()


    def _get_speech_frames(self, frames, audio, sr):
        return frames
    

    def _merge_speech_frames(self, speech_frames, audio, sr):
        audio_bytes = b''.join([
            frame["data"]
            for frame in speech_frames
        ])
        # convert bytes to tensor
        return convert_byte_to_tensor(audio_bytes), sr
    
    
    def _postprocess_audio(self, audio, sr):
        audio, sr = super()._postprocess_audio(audio, sr)
        return audio, sr





if __name__ == "__main__":
    import auditok

    filepath = "samples/auditok_example.wav"
    
    # first method
    vad = Auditok()
    audio, sr = vad.read_audio(filepath)
    audio, sr = vad._preprocess_audio(audio, sr)
    speech_frames = vad._split_to_frames(audio, sr)
    for i, frame in enumerate(speech_frames):
        print(frame["start"], frame["end"])
        vad.save_audio(convert_byte_to_tensor(frame["data"]), sr, f"{i}.wav")


    # second method
    region = auditok.load(filepath)
    audio_regions = auditok.split(
        region,
        min_dur=0.2,     
        max_dur=4,       
        max_silence=0.3, 
        energy_threshold=55,
        drop_trailing_silence=True
    )
    for i, r in enumerate(audio_regions):
        # Regions returned by `split` have 'start' and 'end' metadata fields
        print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))
        filename = r.save("region_{meta.start:.3f}-{meta.end:.3f}.wav")
        # print("region saved as: {}".format(filename))
