"""Copies each trained model with a manifest, its training log, the credits of its training data and a few clean
test clips to free/output/<id>."""
import csv
import json
import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
RUN = Path("free")
ATTRIBUTION = RUN / "attribution"

for wake_word in json.loads(Path("wake_words.json").read_text()):
    model_id = wake_word["id"]
    model_dir = RUN / model_id
    model = model_dir / "trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    if not model.exists():
        print("not trained yet:", model_id)
        continue
    package = RUN / "output" / model_id
    shutil.rmtree(package, ignore_errors=True)
    (package / "test_clips").mkdir(parents=True)
    shutil.copy(model, package / f"{model_id}.tflite")
    manifest = {
        "type": "micro",
        "wake_word": wake_word["wake_word"],
        "author": "jVoice project",
        "website": "https://github.com/jvoice-project/micro-wakeword4j",
        "model": f"{model_id}.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            # tuned with the test results at the end of training.log
            "probability_cutoff": wake_word.get("probability_cutoff", 0.97),
            "feature_step_size": 10,
            "sliding_window_size": 5,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }
    (package / f"{model_id}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy(model_dir / "training.log", package)
    shutil.copy(model_dir / "training_parameters.yaml", package)

    shutil.copy("DATA_SOURCES.md", package)
    credits = package / "attribution"
    credits.mkdir()
    for musan in ATTRIBUTION.glob("musan_*.txt"):
        shutil.copy(musan, credits)
    with open(credits / "fsd50k.csv", "w", newline="") as combined:
        writer = csv.writer(combined)
        writer.writerow(["url", "title", "uploader", "license"])
        for shard in sorted((ATTRIBUTION / "fsd50k").glob("*.csv")):
            with open(shard, newline="") as rows:
                # CC0 clips need no credit
                writer.writerows(row for row in csv.reader(rows) if "/publicdomain/zero/" not in row[3])

    # clean 16 kHz clips, padded with a second of silence, for checking the model outside of training
    for kind in ["positive", "adversarial"]:
        folder = model_dir / f"{kind}_samples"
        names = sorted(p.name for p in folder.glob("*.wav"))
        for name in names[::max(1, len(names) // 12)][:12]:
            audio, _ = librosa.load(folder / name, sr=16000)
            padded = np.concatenate([np.zeros(16000), audio, np.zeros(16000)])
            soundfile.write(package / "test_clips" / f"{kind}_{name}", padded, 16000, subtype="PCM_16")
    print("packaged", package)
