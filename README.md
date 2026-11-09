# microWakeWord for Java

[![Maven Central](https://img.shields.io/maven-central/v/io.github.jvoice-project/micro-wakeword4j.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/io.github.jvoice-project/micro-wakeword4j)
[![CI](https://github.com/jvoice-project/micro-wakeword4j/actions/workflows/build.yml/badge.svg)](https://github.com/jvoice-project/micro-wakeword4j/actions/workflows/build.yml)
[![License](https://img.shields.io/github/license/jvoice-project/micro-wakeword4j.svg)](LICENSE)

A pure Java inference library for [microWakeWord](https://github.com/OHF-Voice/micro-wake-word), the wake word detector of ESPHome voice devices.

microWakeWord models are small streaming neural networks that detect a single wake word phrase in a live audio stream.
This library runs them in plain Java.
It has no dependencies and no native code, and runs on any platform with Java >= 21.

The audio features and the model output are bit-exact with the TensorFlow Lite Micro audio frontend and the TensorFlow Lite reference kernels.
One detector uses about 0.2% of a core on a current desktop CPU.

Only 16 kHz mono audio is supported.

## Usage

The package is distributed through [Maven Central](https://central.sonatype.com/artifact/io.github.jvoice-project/micro-wakeword4j).

### Maven

```xml
<dependency>
    <groupId>io.github.jvoice-project</groupId>
    <artifactId>micro-wakeword4j</artifactId>
    <!-- replace $version with a specific version -->
    <version>$version</version>
</dependency>
```

### Gradle

```gradle
repositories {
    mavenCentral()
}

dependencies {
    implementation 'io.github.jvoice-project:micro-wakeword4j:+' // gets the latest version
}
```

### Example

```java
WakeWordModel model = WakeWordModel.load(BuiltinModel.HEY_JARVIS);
MicroWakeWord detector = new MicroWakeWord(model);

short[] samples = ...; // 16 kHz, 16-bit mono PCM samples
int step = MicroWakeWord.FRAME_SIZE; // 160 samples (10 ms), the only supported frame size

for (int i = 0; i + step <= samples.length; i += step) {
    short[] frame = Arrays.copyOfRange(samples, i, i + step);
    if (detector.process(frame)) {
        System.out.println(model.getWakeWord() + " detected at sample index: " + i);
        detector.reset();
    }
}
```

The detector keeps state between frames, so use one instance per audio stream and feed it consecutive frames.
`process` keeps returning `true` while the average probability stays above the cutoff, so call `reset()` after handling a detection.
Also call `reset()` before reusing an instance for a new stream.
A `WakeWordModel` is immutable and can be shared by any number of detectors.

The model's manifest sets the probability cutoff above which the wake word counts as detected.
To make detection easier or stricter, pass another cutoff between 0 and 1: `new MicroWakeWord(model, 0.9)`.
`getProbability()` returns the current average probability, which helps when tuning the cutoff.
Detection follows ESPHome, so a cutoff behaves the same as on ESPHome devices: a model output of 255 counts as a probability of 1, and the cutoff is rounded down to a multiple of 1/255.

A detection comes 100 to 250 ms after the wake word ends, because the detector averages several model outputs.
To pass the speech that follows the wake word on to speech recognition, start from `getFramesAfterWakeWord()` frames before the detection.
That estimate usually lands within 100 ms of the end of the word.

## Models

These models are bundled:

| `BuiltinModel` | Wake word   | Source                                                                               |
|----------------|-------------|--------------------------------------------------------------------------------------|
| `ALEXA`        | Alexa       | [ESPHome model collection](https://github.com/esphome/micro-wake-word-models)        |
| `HEY_JARVIS`   | Hey Jarvis  | [ESPHome model collection](https://github.com/esphome/micro-wake-word-models)        |
| `HEY_JEEVES`   | Hey Jeeves  | Trained by this project, see [training](training/README.md)                          |
| `HEY_MYCROFT`  | Hey Mycroft | [ESPHome model collection](https://github.com/esphome/micro-wake-word-models)        |
| `HEY_OPENHAB`  | Hey openHAB | Trained by this project, see [training](training/README.md)                          |

Other microWakeWord models, such as those trained with the [microWakeWord training notebook](https://github.com/OHF-Voice/micro-wake-word), are loaded from their JSON manifest:

```java
WakeWordModel model = WakeWordModel.load(Path.of("/path/to/hey_computer.json"));
```

The manifest names the `.tflite` model file, which must be in the same directory.
Models that don't come from files, such as downloads, are loaded from the contents of both files with `WakeWordModel.load(manifestJson, modelBytes)`.
Only version 2 models are supported, the format that microWakeWord has produced since 2024.
The library supports the operators of microWakeWord's streaming models and rejects models with other operators when they are loaded.

## Accuracy

The tests compare every feature and every quantized model output with reference data generated from:

* the microfrontend C library through [pymicro-features](https://github.com/OHF-Voice/pymicro-features), and
* the [LiteRT](https://github.com/google-ai-edge/LiteRT) interpreter with its reference kernels (`BUILTIN_REF`), which use the same integer math as TensorFlow Lite Micro on microcontrollers.

The microfrontend computes its filterbank with floating point math once, and this library matches C builds that round every operation.
Builds that fuse multiply-adds, as Clang does on arm64 by default, compute a slightly different filterbank, which changes the top features a little on some inputs.

The optimized desktop kernels of TensorFlow Lite, which [pymicro-wakeword](https://github.com/OHF-Voice/pymicro-wakeword) uses, round some intermediate values differently.
Around a detection their output differs from the reference kernels by up to about 7%, and on the test audio both detect the same wake words.

To regenerate the reference data, for example after adding a model or test audio, run this with Python 3.11 or older:

```shell
pip install ai-edge-litert==1.0.1 pymicro-features 'numpy<2'
python scripts/generate_reference.py
```

Newer LiteRT releases rescale fully connected outputs in floating point instead of the integer math of TensorFlow Lite Micro, which rounds differently in rare cases.

## Development

You need Java >= 21.

Build the library, run the tests, and package the JAR with Maven:

```shell
./mvnw package
```

Run code formatting checking (`./mvnw spotless:check`) before submitting changes.
