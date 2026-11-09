"""Generates the samples, downloads the augmentation and negative audio, and writes the training features and
configuration of every wake word in wake_words.json.

Only audio that may be used commercially and needs at most attribution is used, see DATA_SOURCES.md. Everything is
written under free/, and the credits the licenses ask for go to free/attribution/. Each step is skipped when its
output already exists, so this can be run again after an interruption or after adding a wake word. Downloads are
deleted once their features are written.

Usage: prepare.py [samples|data|features]..., all three by default. Generating samples mostly uses the CPU and
downloading the data mostly waits on the network, so the two can run at the same time.

Adapted from microWakeWord's basic_training_notebook.ipynb.
"""
import csv
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile
import yaml
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
WAKE_WORDS = json.loads(Path("wake_words.json").read_text())

RUN = Path("free")
DOWNLOADS = RUN / "downloads"
RIRS = RUN / "rirs"
BACKGROUNDS = RUN / "backgrounds"
NEGATIVES = RUN / "negatives"
ATTRIBUTION = RUN / "attribution"
HF = "https://huggingface.co/datasets"
SAMPLE_RATE = 16000

# Trained from scratch on LibriTTS. Most other Piper voices, libritts_r among them, are fine-tuned from voices whose
# recordings are licensed for research or non-commercial use only.
VOICE = "models/en_US-libritts-high.onnx"

# The frontend's features are whole numbers times this. Storing the whole numbers halves the size of the features,
# and microWakeWord scales them back when it reads them.
FEATURE_SCALE = 0.0390625

# one far-field microphone of each DiPCo session
DIPCO_FILES = [f"audio/{split}/{session}.CH1.wav" for split, session in [
    ("eval", "S01_U01"), ("dev", "S02_U01"), ("eval", "S03_U04"), ("dev", "S04_U01"), ("dev", "S05_U04"),
    ("eval", "S06_U04"), ("eval", "S07_U01"), ("eval", "S08_U01"), ("dev", "S09_U04"), ("dev", "S10_U04"),
]]

MUSAN_ID = re.compile(r"\b(?:music|noise)-[a-z-]+-\d{4}\b")


def run(*command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def done(folder):
    return (Path(folder) / ".done").exists()


def mark_done(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / ".done").touch()


def hf_files(repo, folder):
    """Lists the files in a folder of a Hugging Face dataset."""
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/main/{folder}") as response:
        return sorted(entry["path"] for entry in json.load(response) if entry["type"] == "file")


def download(repo, file):
    """Downloads a file of a Hugging Face dataset, resuming a partial download."""
    path = DOWNLOADS / repo.split("/")[1] / file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        run("curl", "-fsSL", "--retry", "5", "-C", "-", "-o", str(partial), f"{HF}/{repo}/resolve/main/{file}")
        partial.rename(path)
    return path


def parquet_rows(path, columns):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=columns):
        yield from batch.to_pylist()


def read_audio(source):
    audio, rate = soundfile.read(source, dtype="int16")
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate} Hz")
    return audio if audio.ndim == 1 else audio[:, 0]


def features(audio):
    return np.round(generate_features_for_clip(audio, step_ms=10) / FEATURE_SCALE).astype(np.uint16)


def five_second_windows(audio):
    """Yields non-overlapping 5 second spectrograms, the way microWakeWord's own negative sets are made."""
    if len(audio) == 0:
        return
    minimum = 6 * SAMPLE_RATE
    if len(audio) < minimum:
        audio = np.tile(audio, -(-minimum // len(audio)))
    spectrogram = features(audio)
    # skips the first 200 ms, where the frontend is still adapting to the noise level
    for start in range(20, len(spectrogram) - 500 + 1, 500):
        yield spectrogram[start:start + 500]


def write_mmap(out_dir, spectrograms):
    """Writes spectrograms as a feature set, which microWakeWord finds by the _mmap suffix of its folder."""
    partial = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(out_dir=str(partial), sample_generator=spectrograms, batch_size=1000, verbose=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    partial.rename(out_dir)
    print("wrote", out_dir, flush=True)


def trim_silence(path):
    """Cuts the silence the voice leaves before and after the speech.

    The features place each sample so that it ends a set time before the end of the window. The speech only ends
    there if the sample ends with the speech.
    """
    audio, rate = soundfile.read(path, dtype="int16")
    # 40 dB below the loudest part keeps quiet endings, such as the "s" of "jeeves"
    _, (start, end) = librosa.effects.trim(audio.astype(np.float32), top_db=40, frame_length=512, hop_length=128)
    soundfile.write(path, audio[start:end], rate, subtype="PCM_16")


def generate(texts, count, out_dir):
    """Generates count samples of each text, named <text index>_<n>.wav, skipping texts generated before.

    Each text gets its own shuffled order of the voice's speakers and speech settings. Otherwise the generator starts
    every text with speaker 0, 1, 2 and so on, all at the same speed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        speaker_seed = zlib.crc32(text.encode())
        marker = out_dir / f".done-{i}"
        if marker.exists() and marker.read_text() == f"{text}\n{count}\n{speaker_seed}\n{VOICE}":
            continue
        for old in out_dir.glob(f"{i}_*.wav"):
            old.unlink()
        # the generator names files 0.wav, 1.wav, ... so each text gets its own folder first
        tmp = out_dir.with_name(out_dir.name + "_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        run(str(ROOT / ".venv-piper/bin/python"), "-m", "piper_sample_generator", text,
            "--model", VOICE, "--max-samples", str(count), "--max-speakers", "700", "--speaker-seed", str(speaker_seed),
            "--output-dir", str(tmp), env={**os.environ, "PYTHONPATH": str(ROOT / "piper-sample-generator")})
        for name in os.listdir(tmp):
            trim_silence(tmp / name)
            shutil.move(tmp / name, out_dir / f"{i}_{name}")
        shutil.rmtree(tmp)
        marker.write_text(f"{text}\n{count}\n{speaker_seed}\n{VOICE}")
        print(out_dir, text, count, "samples", flush=True)


def prepare_rirs(count=3000):
    if done(RIRS):
        return
    RIRS.mkdir(parents=True, exist_ok=True)
    path = download("EaseZh/rirs_noises", "rirs_noises.zip")
    with zipfile.ZipFile(path) as archive:
        # only the simulated rooms, as the recorded ones come from other databases with their own terms
        names = sorted(name for name in archive.namelist()
                       if name.startswith("RIRS_NOISES/simulated_rirs/") and name.endswith(".wav"))
        for name in random.Random(1).sample(names, count):
            room_size = name.split("/")[2]
            (RIRS / f"{room_size}_{Path(name).name}").write_bytes(archive.read(name))
    path.unlink()
    mark_done(RIRS)


def mentions_license(text):
    return any(term in text.lower() for term in ["licen", "cc ", "attribution", "public domain"])


def license_allowed(text):
    text = text.lower()
    if any(term in text for term in ["sharealike", "share alike", "by-sa", "noderiv", "by-nd", "noncommercial",
                                     "by-nc", "sampling"]):
        return False
    return any(term in text for term in ["public domain", "cc0", "attribution", "cc by"])


def musan_credits(folder):
    """Maps the names of the files in a MUSAN folder that may be used to the LICENSE text that credits them.

    Each block of a LICENSE file lists some files and their license. A block that lists no files, if there is one,
    gives the license of the files whose own block names none, and of the files no block lists.
    """
    text = (folder / "LICENSE").read_text(errors="replace")
    blocks = [block.strip() for block in re.split(r"^=+\s*$", text, flags=re.M) if block.strip()]
    folder_wide = "\n".join(block for block in blocks if not MUSAN_ID.search(block))
    credits = {}
    for block in blocks:
        credit = block if mentions_license(block) else f"{block}\n{folder_wide}"
        for file_id in MUSAN_ID.findall(block):
            credits[file_id] = credit
    for wav in folder.glob("*.wav"):
        credits.setdefault(wav.stem, folder_wide)
    return {name: credit for name, credit in credits.items()
            if license_allowed(credit) and (folder / f"{name}.wav").exists()}


def prepare_musan():
    music, noise = BACKGROUNDS / "musan_music", BACKGROUNDS / "musan_noise"
    if done(music) and done(noise):
        return
    extracted = DOWNLOADS / "musan"
    if not done(extracted):
        # streamed, so the speech part, which is not used, is never written to disk
        url = f"{HF}/huseinzol05/musan-mirror/resolve/main/musan.tar.gz"
        with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.startswith(("musan/music/", "musan/noise/")):
                    archive.extract(member, extracted, filter="data")
        mark_done(extracted)
    ATTRIBUTION.mkdir(parents=True, exist_ok=True)
    for kind, out_dir in [("music", music), ("noise", noise)]:
        if done(out_dir):
            continue
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        files, credits = [], set()
        for folder in sorted(p for p in (extracted / "musan" / kind).iterdir() if p.is_dir()):
            for name, credit in sorted(musan_credits(folder).items()):
                files.append(folder / f"{name}.wav")
                credits.add(credit)
        for file in files:
            audio = read_audio(file)
            # cut into 10 second pieces, so mixing in a background does not read a whole song
            for i, start in enumerate(range(0, len(audio) - SAMPLE_RATE + 1, 10 * SAMPLE_RATE)):
                soundfile.write(out_dir / f"{file.stem}_{i:03d}.flac", audio[start:start + 10 * SAMPLE_RATE],
                                SAMPLE_RATE)
        write_mmap(NEGATIVES / "sounds" / "training" / f"musan_{kind}_mmap",
                   (window for file in files for window in five_second_windows(read_audio(file))))
        (ATTRIBUTION / f"musan_{kind}.txt").write_text("\n\n".join(sorted(credits)) + "\n")
        print(kind, len(files), "MUSAN files", flush=True)
        mark_done(out_dir)
    shutil.rmtree(extracted)


def prepare_fsd50k():
    out_dir = BACKGROUNDS / "fsd50k"
    credits_dir = ATTRIBUTION / "fsd50k"
    out_dir.mkdir(parents=True, exist_ok=True)
    credits_dir.mkdir(parents=True, exist_ok=True)
    for file in hf_files("quinnlue/FSD50K-16k", "data"):
        shard = Path(file).stem
        mmap = NEGATIVES / "sounds" / "training" / f"fsd50k_{shard}_mmap"
        if mmap.exists():
            continue
        path = download("quinnlue/FSD50K-16k", file)
        kept = []
        with open(credits_dir / f"{shard}.csv", "w", newline="") as credits:
            writer = csv.writer(credits)
            for row in parquet_rows(path, ["audio", "freesound_id", "freesound_url", "title", "uploader",
                                           "license"]):
                license_url = row["license"] or ""
                if "/publicdomain/zero/" in license_url or "/licenses/by/" in license_url:
                    flac = out_dir / f"{row['freesound_id']}.flac"
                    flac.write_bytes(row["audio"]["bytes"])
                    kept.append(flac)
                    writer.writerow([row["freesound_url"], row["title"], row["uploader"], license_url])
        write_mmap(mmap, (window for flac in kept for window in five_second_windows(read_audio(flac))))
        path.unlink()


def prepare_speech():
    # all of train-clean-100 and the first 14 shards of train-other-500, about 100 hours each
    for split, shards in [("train.clean.100", None), ("train.other.500", 14)]:
        for file in hf_files("openslr/librispeech_asr", f"all/{split}")[:shards]:
            mmap = NEGATIVES / "speech" / "training" / f"librispeech_{split.replace('.', '_')}_{Path(file).stem}_mmap"
            if mmap.exists():
                continue
            path = download("openslr/librispeech_asr", file)
            write_mmap(mmap, (window for row in parquet_rows(path, ["audio"])
                              for window in five_second_windows(read_audio(io.BytesIO(row["audio"]["bytes"])))))
            path.unlink()


def prepare_ami():
    # whole meetings recorded by a single distant microphone
    for file in hf_files("diarizers-community/ami", "sdm"):
        name = Path(file).stem
        split = name.split("-")[0]
        if split == "train":
            mmap = NEGATIVES / "conversation" / "training" / f"ami_{name}_mmap"
        else:
            ambient = "validation_ambient" if split == "validation" else "testing_ambient"
            mmap = NEGATIVES / "ambient" / ambient / f"ami_{name}_mmap"
        if mmap.exists():
            continue
        path = download("diarizers-community/ami", file)
        meetings = (read_audio(io.BytesIO(row["audio"]["bytes"])) for row in parquet_rows(path, ["audio"]))
        if split == "train":
            write_mmap(mmap, (window for meeting in meetings for window in five_second_windows(meeting)))
        else:
            # the ambient sets measure false accepts per hour, so they keep each meeting whole
            write_mmap(mmap, (features(meeting) for meeting in meetings))
        path.unlink()


def prepare_dipco():
    mmap = NEGATIVES / "ambient" / "testing_ambient" / "dipco_mmap"
    if mmap.exists():
        return
    paths = [download("huckiyang/DiPCo", file) for file in DIPCO_FILES]
    write_mmap(mmap, (features(read_audio(path)) for path in paths))
    for path in paths:
        path.unlink()


def to_uint16(spectrograms):
    for spectrogram in spectrograms:
        yield np.round(spectrogram / FEATURE_SCALE).astype(np.uint16)


def write_features(clips, augmenter, output_dir, training_slides=10):
    for split, split_name, repetition, slide_frames in [
        ("training", "train", 2, training_slides),
        ("validation", "validation", 1, 10),
        # the testing set uses the streaming model, so it needs no artificial repetition
        ("testing", "test", 1, 1),
    ]:
        mmap = output_dir / split / "wakeword_mmap"
        if mmap.exists():
            continue
        spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide_frames,
                                             step_ms=10)
        write_mmap(mmap, to_uint16(spectrograms.spectrogram_generator(split=split_name, repeat=repetition)))


def clips_in(folder):
    # Also drops the frames voice activity detection finds silent, as microWakeWord's notebook does. On its own that
    # leaves about 0.1 s of the silence after the speech, which is why generate() trims the samples too.
    return Clips(input_directory=str(folder), file_pattern="*.wav", max_clip_duration_s=None, remove_silence=True,
                 random_split_seed=10, split_count=0.1)


def feature_set(features_dir, truth, sampling_weight, truncation_strategy):
    return {"features_dir": str(features_dir), "sampling_weight": sampling_weight, "penalty_weight": 1.0,
            "truth": truth, "truncation_strategy": truncation_strategy, "type": "mmap"}


def write_config(model_dir):
    config = {
        "window_step_ms": 10,
        "train_dir": str(model_dir / "trained"),
        "features": [
            feature_set(model_dir / "positive_features", True, 2.0, "truncate_start"),
            # Random windows, so the model also learns to stay quiet in the middle of a sound-alike phrase and in the
            # silence after it, not only just after it ends.
            feature_set(model_dir / "adversarial_features", False, 2.0, "random"),
            feature_set(NEGATIVES / "speech", False, 10.0, "random"),
            feature_set(NEGATIVES / "conversation", False, 10.0, "random"),
            feature_set(NEGATIVES / "sounds", False, 5.0, "random"),
            # only used for validation and testing
            feature_set(NEGATIVES / "ambient", False, 0.0, "split"),
        ],
        "training_steps": [10000],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    with open(model_dir / "training_parameters.yaml", "w") as file:
        yaml.dump(config, file)


VALID_STEPS = {"samples", "data", "features"}
steps = sys.argv[1:] or ["samples", "data", "features"]
if unknown := set(steps) - VALID_STEPS:
    sys.exit(f"Usage: {sys.argv[0]} [samples|data|features]... (unknown: {' '.join(sorted(unknown))})")

if "samples" in steps:
    for wake_word in WAKE_WORDS:
        model_dir = RUN / wake_word["id"]
        generate(wake_word["spellings"], wake_word["samples_per_spelling"], model_dir / "positive_samples")
        generate(wake_word["adversarial"], wake_word["samples_per_adversarial"], model_dir / "adversarial_samples")

if "data" in steps:
    prepare_rirs()
    prepare_musan()
    prepare_fsd50k()
    prepare_speech()
    prepare_ami()
    prepare_dipco()
    shutil.rmtree(DOWNLOADS, ignore_errors=True)

def augmenter(min_jitter_s, max_jitter_s):
    """Mixes in noise and room echo, and places the clip so it ends between min and max jitter before the end."""
    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=[str(RIRS)],
        background_paths=[str(BACKGROUNDS / name) for name in ["musan_music", "musan_noise", "fsd50k"]],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=min_jitter_s,
        max_jitter_s=max_jitter_s,
    )


if "features" in steps:
    for wake_word in WAKE_WORDS:
        model_dir = RUN / wake_word["id"]
        write_features(clips_in(model_dir / "positive_samples"), augmenter(0.195, 0.205),
                       model_dir / "positive_features")
        # Up to 1.2 s of silence after a sound-alike phrase. Training picks random windows of these, so it needs one
        # whole spectrogram per clip rather than copies shifted by a few frames.
        write_features(clips_in(model_dir / "adversarial_samples"), augmenter(0.2, 1.2),
                       model_dir / "adversarial_features", training_slides=None)
        write_config(model_dir)

print("Done:", " ".join(steps), flush=True)
