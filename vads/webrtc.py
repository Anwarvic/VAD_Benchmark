import webrtcvad
from collections import deque

from utils import *
from vads.vad import Vad


class WebRTC(Vad):
    def __init__(self,
            threshold=0.9,
            window_size_ms=96,
            merge_ratio=0.9
        ):
        super().__init__(threshold, window_size_ms)
        agg = (
            0 if threshold == 0
            else
                1 if 0 < threshold <= 1/3
                else
                    2 if 1/3 <= threshold <= 2/3
                    else 3
        )
        self._vad = webrtcvad.Vad(agg)
        self._chunk_size_ms = 10
        self._window_size_ms = window_size_ms
        self._merge_ratio = merge_ratio
        self._valid_sr = [8000, 16000, 32000, 48000]
    

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
        while offset + n < len(audio):
            yield {
                "data": audio[offset : offset+n],
                "start": start_frame,
                "end": start_frame+num_frames
            }
            start_frame += num_frames
            offset += n


    def _get_speech_frames(self, frames, audio, sr):
        num_blocks = int(self._window_size_ms / self._chunk_size_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = deque(maxlen=num_blocks)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = self._vad.is_speech(frame["data"], sr)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > self._merge_ratio * ring_buffer.maxlen:
                    triggered = True
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, _ in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, is_speech in ring_buffer if not is_speech]
                )
                # If more than {self._merge_ratio}% of the frames in the
                # ring buffer are unvoiced, then enter NOTTRIGGERED and
                # yield whatever audio we've collected.
                if num_unvoiced > self._merge_ratio * ring_buffer.maxlen:
                    triggered = False
                    yield {
                        "data": b''.join([f["data"] for f in voiced_frames]),
                        "start": voiced_frames[0]["start"],
                        "end": voiced_frames[-1]["end"]
                    }
                    ring_buffer.clear()
                    voiced_frames = []
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield {
                "data": b''.join([f["data"] for f in voiced_frames]),
                "start": voiced_frames[0]["start"],
                "end": voiced_frames[-1]["end"]
            }
    

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
    from os.path import dirname, abspath, join
    
    print("Running WebRTC Vad")
    samples_dir = join(dirname(dirname(abspath(__file__))), "samples")
    audio_filepath = join(samples_dir, "double_48k.wav")
    audio, sr = load_audio(audio_filepath)

    vad = WebRTC()
    audio, sr = vad.trim_silence(audio, sr)
    save_audio(audio, sr, join(samples_dir, "webrtc_example_48k.wav"))