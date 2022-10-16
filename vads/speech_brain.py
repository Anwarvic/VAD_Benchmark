from speechbrain.pretrained import VAD
from utils import *


class SpeechBrain():
    def __init__(self, hf_src):
        self._vad = VAD.from_hparams(source=hf_src)



    def _split_to_frames(self, audio, sr):
        pass


    def _get_speech_frames(self, frames, sr):
        pass


    def _merge_speech_frames(self, speech_frames):
        pass


    def get_speech_boundaries(self, audio, sr):
        pass
    
    
    def trim_silence(self, audio, sr):
        pass
    





# audio_file = 'pretrained_model_checkpoints/example_vad.wav'
# prob_chunks = VAD.get_speech_prob_file(audio_file)

# # 2- Let's apply a threshold on top of the posteriors
# prob_th = VAD.apply_threshold(prob_chunks).float()

# # 3- Let's now derive the candidate speech segments
# boundaries = VAD.get_boundaries(prob_th)

# # 4- Apply energy VAD within each candidate speech segment (optional)

# boundaries = VAD.energy_VAD(audio_file,boundaries)

# # 5- Merge segments that are too close
# boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)

# # 6- Remove segments that are too short
# boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)


if __name__ == "__main__":
    from os.path import dirname, abspath, join
    
    print("Running SpeechBrain Vad")
    samples_dir = join(dirname(dirname(abspath(__file__))), "samples")
    audio_filepath = join(samples_dir, "example_16k.wav")

    vad = SpeechBrain("speechbrain/vad-crdnn-libriparty")
    boundaries = vad._vad.get_speech_segments(audio_filepath)
    vad._vad.save_boundaries(boundaries)
