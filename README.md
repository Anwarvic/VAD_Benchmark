# VAD Benchmark
Benchmarking different VAD models on AVA-Speech dataset:
    - Website: http://research.google.com/ava/download.html#ava_speech_download
    - Paper: https://arxiv.org/pdf/1808.00606.pdf

## Dataset

You can download & pre-process audio from AVA-Speech dataset using the
following bash script:
```
$ cd dataset
$ bash download_ava_speech.sh
```

## Models
The following are the list of the available VAD models:

- **WebRTC**: https://github.com/wiseman/py-webrtcvad
- **Silero**: https://github.com/snakers4/silero-vad
- **auditok**: https://github.com/amsehili/auditok
- **pyannote:** https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/voice_activity_detection.ipynb
- **Nvidia's Marblenet**: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Voice_Activity_Detection.ipynb
- **SpeechBrain**: https://huggingface.co/speechbrain/vad-crdnn-libriparty#perform-voice-activity-detection
- **Voice Activity Detection Project:** https://github.com/filippogiruzzi/voice_activity_detection
- ~~**PicoVoice Cobra**: https://github.com/Picovoice/cobra. [WEB-BASED]~~


## TODO

- [ ] create test cases.
- [ ] create viz function for VAD.