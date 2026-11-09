#!/usr/bin/env python3
"""Generate the reference output that the tests compare against.

Features come from the microfrontend C library through pymicro-features. Model outputs come
from the LiteRT interpreter with its reference kernels, which use the same integer math as
TensorFlow Lite Micro.

LiteRT 1.0.1 rescales fully connected outputs with the same integer math as TensorFlow Lite
Micro. Newer releases, such as 2.3.0, rescale them in floating point, which rounds differently in
rare cases. LiteRT 1.0.1 needs Python 3.11 or older.

    pip install ai-edge-litert==1.0.1 pymicro-features 'numpy<2'
    python scripts/generate_reference.py
"""
import json
import wave
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from pymicro_features import MicroFrontend

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [
    ROOT / "src/main/resources/io/github/jvoiceproject/microwakeword/models",
    ROOT / "src/test/resources/models",
]
AUDIO_DIR = ROOT / "src/test/resources/audio"
REFERENCE_DIR = ROOT / "src/test/resources/reference"
FEATURE_WAVS = ["hey_openhab_1.wav", "hey_peppa_pig.wav"]

FRAME_SIZE = 160  # 10 ms
FEATURE_SCALE = 0.0390625  # pymicro-features scales the frontend output by this


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


def features(audio):
    frontend = MicroFrontend()
    frames = []
    for i in range(0, len(audio) - 2 * FRAME_SIZE + 1, 2 * FRAME_SIZE):
        result = frontend.process_samples(audio[i : i + 2 * FRAME_SIZE])
        assert result.samples_read == FRAME_SIZE
        if result.features:
            frames.append([round(f / FEATURE_SCALE) for f in result.features])
    return frames


def outputs(model_path, frames):
    interpreter = Interpreter(
        model_path=str(model_path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero_point = inp["quantization"]
    stride = inp["shape"][1]
    values = []
    for i in range(0, len(frames) - stride + 1, stride):
        block = np.array(frames[i : i + stride], np.float64) * FEATURE_SCALE
        quantized = np.clip(np.round(block / scale + zero_point), -128, 127)
        interpreter.set_tensor(inp["index"], quantized.astype(np.int8).reshape(inp["shape"]))
        interpreter.invoke()
        values.append(int(interpreter.get_tensor(out["index"]).item()))
    return values


def main():
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    frames = {wav.name: features(load_wav(wav)) for wav in wavs}

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in FEATURE_WAVS:
        out = REFERENCE_DIR / f"features-{Path(name).stem}.txt"
        out.write_text("".join(" ".join(map(str, f)) + "\n" for f in frames[name]))
        print(f"{out.relative_to(ROOT)}: {len(frames[name])} frames")

    for manifest in sorted(p for d in MODEL_DIRS for p in d.glob("*.json")):
        model = manifest.parent / json.loads(manifest.read_text())["model"]
        lines = [f"{wav.name} " + " ".join(map(str, outputs(model, frames[wav.name]))) for wav in wavs]
        out = REFERENCE_DIR / f"{model.stem}.txt"
        out.write_text("\n".join(lines) + "\n")
        print(f"{out.relative_to(ROOT)}: {len(wavs)} files")


if __name__ == "__main__":
    main()
