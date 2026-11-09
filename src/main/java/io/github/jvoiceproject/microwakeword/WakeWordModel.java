/*
 * #%L
 * micro-wakeword4j
 * %%
 * Copyright (C) 2026 Contributors to micro-wakeword4j
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.github.jvoiceproject.microwakeword;

import io.github.jvoiceproject.microwakeword.internal.Json;
import io.github.jvoiceproject.microwakeword.internal.tflite.core.Program;
import io.github.jvoiceproject.microwakeword.internal.tflite.core.TfLiteModel;
import io.github.jvoiceproject.microwakeword.internal.tflite.frontend.MicroFrontend;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A microWakeWord model: the quantized TensorFlow Lite network and the detection settings of its
 * JSON manifest.
 *
 * <p>Only version 2 models are supported, the format that microWakeWord and ESPHome have produced
 * since 2024. Models are immutable and can be shared by any number of {@link MicroWakeWord}
 * instances.
 *
 * @author Dan Cunningham - Initial contribution
 */
public final class WakeWordModel {
    private static final double FEATURE_STEP_MS = 10;

    private final String wakeWord;
    private final List<String> trainedLanguages;
    private final double probabilityCutoff;
    private final int slidingWindowSize;
    final Program program;

    /** Number of 10 ms feature frames that the model takes per inference. */
    final int stride;

    private WakeWordModel(Map<String, Object> manifest, byte[] model) throws IOException {
        Object micro = manifest.get("micro");
        if (!"micro".equals(manifest.get("type")) || !(micro instanceof Map<?, ?> settings)) {
            throw new IOException("Not a microWakeWord manifest");
        }
        if (!Double.valueOf(2).equals(manifest.get("version"))) {
            throw new IOException("Only version 2 microWakeWord models are supported");
        }
        Object step = settings.get("feature_step_size");
        if (step != null && !Double.valueOf(FEATURE_STEP_MS).equals(step)) {
            throw new IOException("Only models with 10 ms feature steps are supported");
        }
        wakeWord = string(manifest, "wake_word");
        trainedLanguages = languages(manifest.get("trained_languages"));
        probabilityCutoff = number(settings, "probability_cutoff");
        double window = number(settings, "sliding_window_size");
        if (probabilityCutoff < 0 || probabilityCutoff > 1) {
            throw new IOException("Probability cutoff must be between 0 and 1");
        }
        if (window < 1 || window > 1000 || window != Math.rint(window)) {
            throw new IOException("Invalid sliding window size");
        }
        slidingWindowSize = (int) window;

        try {
            program = Program.compile(TfLiteModel.parse(model));
        } catch (RuntimeException e) {
            throw new IOException("Malformed TensorFlow Lite model", e);
        }
        int[] shape = program.inputShape;
        if (shape.length != 3
                || shape[0] != 1
                || shape[1] < 1
                || shape[2] != MicroFrontend.FEATURE_COUNT) {
            throw new IOException("Model input must have the shape [1, frames, 40]");
        }
        stride = shape[1];
    }

    /**
     * Loads a model from its manifest file. The model file that the manifest names must be in the
     * same directory.
     *
     * @throws IOException when the files can't be read or the model is not supported
     */
    public static WakeWordModel load(Path manifestFile) throws IOException {
        Map<String, Object> manifest = manifest(Files.readString(manifestFile));
        Path directory = manifestFile.toAbsolutePath().normalize().getParent();
        Path modelFile;
        try {
            modelFile = directory.resolve(string(manifest, "model")).normalize();
        } catch (InvalidPathException e) {
            throw new IOException("Invalid model file name", e);
        }
        if (!directory.equals(modelFile.getParent())) {
            throw new IOException("The model must be next to its manifest");
        }
        return new WakeWordModel(manifest, Files.readAllBytes(modelFile));
    }

    /**
     * Loads a model from the contents of its manifest and of the model file that the manifest
     * names, for models that don't come from files, such as downloads.
     *
     * @param manifestJson the JSON manifest
     * @param model the contents of the .tflite model file
     * @throws IOException when the model is not supported
     */
    public static WakeWordModel load(String manifestJson, byte[] model) throws IOException {
        return new WakeWordModel(manifest(manifestJson), model);
    }

    /**
     * Loads one of the bundled models.
     *
     * @throws IOException when the bundled files can't be read
     */
    public static WakeWordModel load(BuiltinModel builtin) throws IOException {
        Map<String, Object> manifest =
                manifest(new String(resource(builtin.getId() + ".json"), StandardCharsets.UTF_8));
        return new WakeWordModel(manifest, resource(string(manifest, "model")));
    }

    /** The phrase that the model detects, for example "Hey Jarvis". */
    public String getWakeWord() {
        return wakeWord;
    }

    /** Languages the model was trained for, as ISO 639-1 codes. Empty when not specified. */
    public List<String> getTrainedLanguages() {
        return trainedLanguages;
    }

    /** The average probability above which the wake word counts as detected. */
    public double getProbabilityCutoff() {
        return probabilityCutoff;
    }

    /** Number of consecutive model outputs that are averaged for detection. */
    public int getSlidingWindowSize() {
        return slidingWindowSize;
    }

    @Override
    public String toString() {
        return "WakeWordModel[" + wakeWord + "]";
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> manifest(String json) throws IOException {
        if (Json.parse(json) instanceof Map<?, ?> manifest) {
            return (Map<String, Object>) manifest;
        }
        throw new IOException("Not a microWakeWord manifest");
    }

    private static byte[] resource(String name) throws IOException {
        try (InputStream in = WakeWordModel.class.getResourceAsStream("models/" + name)) {
            if (in == null) {
                throw new IOException("Missing " + name);
            }
            return in.readAllBytes();
        }
    }

    private static String string(Map<?, ?> map, String key) throws IOException {
        if (map.get(key) instanceof String value && !value.isEmpty()) {
            return value;
        }
        throw new IOException("Manifest is missing \"" + key + "\"");
    }

    private static double number(Map<?, ?> map, String key) throws IOException {
        if (map.get(key) instanceof Double value) {
            return value;
        }
        throw new IOException("Manifest is missing \"" + key + "\"");
    }

    private static List<String> languages(Object value) throws IOException {
        if (value == null) {
            return List.of();
        }
        List<String> languages = new ArrayList<>();
        if (value instanceof List<?> list) {
            for (Object language : list) {
                if (language instanceof String code) {
                    languages.add(code);
                }
            }
            if (languages.size() == list.size()) {
                return List.copyOf(languages);
            }
        }
        throw new IOException("Invalid \"trained_languages\"");
    }
}
