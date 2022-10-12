import webrtcvad
from collections import deque

from utils import *


class WebRTC():
    def __init__(self, agg=3, chunk_size=30, window_size=300, merge_ratio=0.9):
        self._vad = webrtcvad.Vad(agg)
        self._chunk_size = chunk_size
        self._window_size = window_size
        self._merge_ratio = merge_ratio
        self._valid_sr = [8000, 16000, 32000, 48000]
    

    def _preprocess_audio(self, audio, sr):
        # check if audio is mono
        if audio.size(0) != 1:
            audio = change_num_channels(audio, sr, 1)
        # check if sample rate is valid
        if sr not in self._valid_sr:
            tgt_sr = get_closest_sample_rate(sr, self._valid_sr)
            audio = change_sample_rate(audio, sr, tgt_sr)
            sr = tgt_sr
        # convert audio to bytes
        audio = convert_tensor_to_bytes(audio)
        return audio, sr
    

    def get_frames(self, audio, sr):
        offset = 0
        timestamp = 0.0
        n = int(sr * (self._chunk_size / 1000.0) * 2)
        duration = (float(n) / sr) / 2.0
        while offset + n < len(audio):
            yield {
                "data": audio[offset:offset + n],
                "start": timestamp,
                "duration": duration
            }
            timestamp += duration
            offset += n


    def get_speech_frames(self, frames, sr):
        num_frames = int(self._window_size / self._chunk_size)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = deque(maxlen=num_frames)
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
                    [f for f, speech in ring_buffer if not speech]
                )
                # If more than {self._merge_ratio}% of the frames in the
                # ring buffer are unvoiced, then enter NOTTRIGGERED and
                # yield whatever audio we've collected.
                if num_unvoiced > self._merge_ratio * ring_buffer.maxlen:
                    triggered = False
                    yield [
                        b''.join([f["data"] for f in voiced_frames]),
                        voiced_frames[0]["start"],
                        voiced_frames[-1]["start"]
                    ]
                    ring_buffer.clear()
                    voiced_frames = []
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield [
                b''.join([f["data"] for f in voiced_frames]),
                voiced_frames[0]["start"],
                voiced_frames[-1]["start"]
            ]
    

    def merge_frames(self, speech_frames):
        SCALE = 6e-5
        THRESHOLD = 0.3
        merge_frames = list()
        timestamp_start = 0.0
        timestamp_end = 0.0
        # removing start, end, and long sequences of sils
        for i, segment in enumerate(speech_frames):
            merge_frames.append(segment[0])
            if i and timestamp_start:
                sil_duration = segment[1] - timestamp_end # TODO: can be optimized
                if sil_duration > THRESHOLD:
                    merge_frames.append(
                        int(THRESHOLD / SCALE // 2)*(b'\x00\x00')
                    )
                else:
                    merge_frames.append(
                        int((sil_duration / SCALE // 2))*(b'\x00\x00')
                    )
            timestamp_start = segment[1]
            timestamp_end = segment[2]
        return b''.join(merge_frames)
    
    
    def _postprocess_audio(self, audio, sr):
        # convert bytes to tensor
        audio = convert_byte_to_tensor(audio) 
        return audio, sr


    def trim_silence(self, audio, sr):
        orig_sr = sr
        # preprocess audio if needed
        audio, sr = self._preprocess_audio(audio, sr)
        # get frames having speech
        frames = list(self.get_frames(audio, sr))
        speech_frames = self.get_speech_frames(frames, sr)
        # merge speech frames
        merged_frame = self.merge_frames(speech_frames)
        # post-process audio if needed
        audio, sr = self._postprocess_audio(merged_frame, sr)
        # change sample rate back to the original one
        if sr != orig_sr:
            audio = change_sample_rate(audio, sr, orig_sr)
        return audio, orig_sr



if __name__ == "__main__":
    from os.path import dirname, abspath, join
    
    print("Running WebRTC Vad")
    samples_dir = join(dirname(dirname(abspath(__file__))), "samples")
    audio_filepath = join(samples_dir, "example_48k.mp3")
    audio, sr = load_audio(audio_filepath)
    
    vad = WebRTC()
    audio, sr = vad.trim_silence(audio, sr)
    save_audio(audio, sr, join(samples_dir, "example_vad_48k.wav"))