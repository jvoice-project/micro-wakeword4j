# Training data sources

The wake word models are trained only on audio that may be used commercially and needs at most attribution.
The audio was resampled, cut, mixed and turned into spectrogram features for training.
None of it is included in the models.

## Wake word and sound-alike samples

- **Piper `en_US-libritts-high` voice**, by Michael Hansen (rhasspy), used through
  [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) (MIT) to synthesize the samples.
  License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  [Model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/libritts/high/MODEL_CARD).
  The voice was trained from scratch, so no other voice's recordings went into it.
- **LibriTTS**, by H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen and Y. Wu: train-clean-360,
  which the voice was trained on.
  License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [openslr.org/60](https://www.openslr.org/60/).

## Negative audio

- **LibriSpeech**, by V. Panayotov, G. Chen, D. Povey and S. Khudanpur: train-clean-100 and part of train-other-500,
  as general speech. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  [openslr.org/12](https://www.openslr.org/12/), through
  [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr).
- **AMI Meeting Corpus**, by the AMI Consortium: whole meetings from a single distant microphone, as background
  conversation. The training meetings are used for training, the others to measure false accepts.
  License: [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml), through
  [diarizers-community/ami](https://huggingface.co/datasets/diarizers-community/ami).
- **DiPCo, Dinner Party Corpus**, by Amazon: one far-field microphone of each session, only to measure false
  accepts after training. License: [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/), through
  [huckiyang/DiPCo](https://huggingface.co/datasets/huckiyang/DiPCo).
- **MUSAN**, by D. Snyder, G. Chen and D. Povey: the music and noise parts, as background sounds.
  License: [CC BY 4.0](https://www.openslr.org/17/). Of these, only the files whose own license is CC BY, CC0 or public
  domain are used. Their credits are in `attribution/musan_music.txt` and `attribution/musan_noise.txt`.
- **FSD50K**, by E. Fonseca, X. Favory, J. Pons, F. Font and X. Serra: as background sounds.
  License: [CC BY 4.0](https://zenodo.org/records/4060432). Of these, only the clips whose own license is CC0 or
  CC BY 3.0 are used. The credits of the CC BY 3.0 clips are in `attribution/fsd50k.csv`. Through the 16 kHz repack
  [quinnlue/FSD50K-16k](https://huggingface.co/datasets/quinnlue/FSD50K-16k).

## Room echo

- **Room Impulse Response and Noise Database**, by T. Ko, V. Peddinti, D. Povey, M. L. Seltzer and S. Khudanpur:
  3,000 of the simulated room impulse responses. License: Apache 2.0.
  [openslr.org/28](https://www.openslr.org/28/), through
  [EaseZh/rirs_noises](https://huggingface.co/datasets/EaseZh/rirs_noises).

## Training code

- **microWakeWord**, by Kevin Ahrendt. License: Apache 2.0.
  [github.com/kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord).
