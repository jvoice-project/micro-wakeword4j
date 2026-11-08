# Wake word training

This pipeline trained the bundled Hey Jeeves and Hey openHAB models with [microWakeWord](https://github.com/kahrendt/microWakeWord).
It follows microWakeWord's training notebook, with these changes:

- It uses only audio that may be used commercially and needs at most attribution, see [DATA_SOURCES.md](DATA_SOURCES.md).
  The credits those licenses ask for are in [attribution](attribution).
- The samples come from Piper's `en_US-libritts-high` voice, which was trained from scratch on LibriTTS.
  The notebook's `en_US-libritts_r-medium` voice is fine-tuned from a voice whose recordings are licensed for research only.
- Each phrase is spoken by its own shuffled order of the voice's speakers and speech settings.
  Without that, the samples of every phrase start with speaker 0, 1, 2 and so on, all at the same speed.
- The silence the voice leaves after the speech is cut, so each sample ends where its last word does.
  Without that, about 0.1 seconds of silence remains, and Hey openHAB triggered on conversation five times as often.
- The model learns to stay quiet anywhere in a sound-alike phrase and in up to 1.2 seconds of silence after it, not only just after it ends.
  Without that, phrases such as "I use open hab at home" and "hey open" still triggered Hey openHAB.

## Setup

It needs Python 3.12, [uv](https://docs.astral.sh/uv/), and about 40 GB of disk.
Everything runs on the CPU, except that generating samples uses an NVIDIA GPU when there is one.
These are the versions the bundled models were trained with:

```sh
git clone https://github.com/kahrendt/microWakeWord
git -C microWakeWord checkout 4665173cd35f1cff9a61e06fc427f124766c488e
git clone https://github.com/rhasspy/piper-sample-generator
git -C piper-sample-generator checkout 2971426a55072f7d22fec416ca7800df8bd23207
git -C piper-sample-generator apply ../piper-sample-generator-speaker-seed.patch

# microWakeWord, with the audio-metadata fork its training notebook installs
uv venv .venv-train --python 3.12
uv pip install --python .venv-train \
    'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'
uv pip install --python .venv-train -e ./microWakeWord 'datasets<4' soundfile librosa pyarrow tensorboard

# Piper, run from the patched clone
uv venv .venv-piper --python 3.12
uv pip install --python .venv-piper piper-sample-generator
mkdir -p models
voice=https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33a10e145e109b886d661984201ac3c4/en/en_US/libritts/high
curl -fL -o models/en_US-libritts-high.onnx "$voice/en_US-libritts-high.onnx"
curl -fL -o models/en_US-libritts-high.onnx.json "$voice/en_US-libritts-high.onnx.json"
```

## Training

[wake_words.json](wake_words.json) lists each wake word with its spellings, its sound-alike phrases and its cutoff.

```sh
.venv-train/bin/python prepare.py   # samples, downloads and features, about two hours the first time
./train.sh                          # about 5 minutes per wake word on an M4 Pro
.venv-train/bin/python package.py   # the model, manifest, credits and test clips in free/output/<id>/
```

Everything they write goes to `free/`, named for the freely licensed data it holds.
`prepare.py` skips the steps it finished before, so it can be run again after an interruption.
Its `samples` step mostly uses the CPU and its `data` step mostly waits on the network, so `prepare.py samples` and `prepare.py data` can run at the same time before `prepare.py features`.

The test at the end of `free/<id>/training.log` lists, for each cutoff, the share of wake words missed (`frr`) and the false accepts per hour (`faph`).
The false accepts come from 14.4 hours of meetings and dinner-party conversation that the model did not train on.
The bundled models use the lowest cutoff with at most 0.14 false accepts per hour, about one every 7 hours.
Set it in `wake_words.json` and run `package.py` again.

The results vary a lot between training runs on the same features.
Each bundled model is the best of three runs.

## Results

| Wake word   | Cutoff | Missed | False accepts per hour | Sound-alikes that trigger it                                                                                            |
|-------------|--------|--------|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Hey Jeeves  | 0.77   | 17.8%  | 0.14                   | "hey geez" 91%, "hey jeans" 79%, "hey keys" 43%, "hey cheese" 40%, "hey, chief" 13%, "hey jesus" 11%, others 7% or less |
| Hey openHAB | 0.99   | 21.5%  | 0.14                   | "hey, open up" 4%, others 1% or less                                                                                    |

The missed wake words are synthesized samples with noise and room echo mixed in.
The sound-alike rates come from clean samples that the model did not train on.
Real voices will give somewhat different numbers.
