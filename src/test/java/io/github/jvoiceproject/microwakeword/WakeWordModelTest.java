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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

public class WakeWordModelTest {
    private static final String MANIFEST =
            """
            {
              "type": "micro",
              "wake_word": "computer",
              "model": "%s",
              "trained_languages": ["en"],
              "version": %d,
              "micro": {
                "probability_cutoff": 0.97,
                "sliding_window_size": 5,
                "feature_step_size": 10
              }
            }
            """;

    @TempDir Path directory;

    @ParameterizedTest
    @EnumSource(BuiltinModel.class)
    public void loadsBuiltinModels(BuiltinModel builtin) throws IOException {
        WakeWordModel model = WakeWordModel.load(builtin);

        assertEquals(builtin.getId(), model.getWakeWord().toLowerCase().replace(' ', '_'));
        assertEquals(List.of("en"), model.getTrainedLanguages());
        assertEquals(5, model.getSlidingWindowSize());
        assertEquals(3, model.stride);
    }

    @Test
    public void readsTheManifest() throws Exception {
        WakeWordModel model = WakeWordModel.load(TestResources.path("/models/computer.json"));

        assertEquals("computer", model.getWakeWord());
        assertEquals(0.97, model.getProbabilityCutoff());
        assertEquals(5, model.getSlidingWindowSize());
        assertEquals("WakeWordModel[computer]", model.toString());
    }

    @Test
    public void rejectsUnsupportedManifestSettings() throws Exception {
        byte[] computer = computerModel();
        String valid = MANIFEST.formatted("model.tflite", 2);

        for (String invalid :
                List.of(
                        "[]",
                        valid.replace("\"type\": \"micro\"", "\"type\": \"other\""),
                        valid.replace("\"micro\": {", "\"other\": {"),
                        valid.replace("\"wake_word\": \"computer\"", "\"wake_word\": \"\""),
                        valid.replace("\"feature_step_size\": 10", "\"feature_step_size\": 20"),
                        valid.replace(
                                "\"probability_cutoff\": 0.97", "\"probability_cutoff\": -0.1"),
                        valid.replace(
                                "\"probability_cutoff\": 0.97", "\"probability_cutoff\": 1.1"),
                        valid.replace("\"probability_cutoff\": 0.97,\n", ""),
                        valid.replace("\"sliding_window_size\": 5", "\"sliding_window_size\": 0"),
                        valid.replace(
                                "\"sliding_window_size\": 5", "\"sliding_window_size\": 1001"),
                        valid.replace("\"sliding_window_size\": 5", "\"sliding_window_size\": 2.5"),
                        valid.replace("[\"en\"]", "[1]"))) {
            assertThrows(IOException.class, () -> WakeWordModel.load(invalid, computer), invalid);
        }
    }

    @Test
    public void loadsAModelNextToItsManifest() throws Exception {
        Path manifest = writeModel("model.tflite", 2, computerModel());

        assertEquals("computer", WakeWordModel.load(manifest).getWakeWord());
    }

    @Test
    public void loadsAModelFromItsContents() throws Exception {
        WakeWordModel model =
                WakeWordModel.load(MANIFEST.formatted("unused.tflite", 2), computerModel());

        assertEquals("computer", model.getWakeWord());
        assertEquals(0.97, model.getProbabilityCutoff());
        assertEquals(2, model.stride);
    }

    @Test
    public void rejectsContentsThatAreNotModels() {
        assertThrows(
                IOException.class,
                () -> WakeWordModel.load(MANIFEST.formatted("model.tflite", 2), new byte[16]));
        assertThrows(
                IOException.class, () -> WakeWordModel.load("{\"type\": \"micro\", ", new byte[0]));
    }

    @Test
    public void rejectsVersion1Manifests() throws Exception {
        Path manifest = writeModel("model.tflite", 1, computerModel());

        assertThrows(IOException.class, () -> WakeWordModel.load(manifest));
    }

    @Test
    public void rejectsModelsOutsideTheManifestDirectory() throws Exception {
        Files.write(directory.resolve("outside.tflite"), computerModel());
        Path manifest = writeModel("../outside.tflite", 2, null);

        assertThrows(IOException.class, () -> WakeWordModel.load(manifest));
    }

    @Test
    public void rejectsFilesThatAreNotModels() throws Exception {
        Path manifest = writeModel("model.tflite", 2, "not a model".getBytes());

        assertThrows(IOException.class, () -> WakeWordModel.load(manifest));
    }

    @Test
    public void rejectsTruncatedModels() throws Exception {
        byte[] model = computerModel();
        Path manifest = writeModel("model.tflite", 2, Arrays.copyOf(model, model.length / 2));

        assertThrows(IOException.class, () -> WakeWordModel.load(manifest));
    }

    @Test
    public void rejectsInvalidManifests() throws Exception {
        Path manifest = directory.resolve("broken.json");
        Files.writeString(manifest, "{\"type\": \"micro\", ");

        assertThrows(IOException.class, () -> WakeWordModel.load(manifest));
    }

    private Path writeModel(String modelPath, int version, byte[] model) throws IOException {
        Path folder = Files.createDirectories(directory.resolve("model"));
        if (model != null) {
            Files.write(folder.resolve(modelPath), model);
        }
        Path manifest = folder.resolve("model.json");
        Files.writeString(manifest, MANIFEST.formatted(modelPath, version));
        return manifest;
    }

    private static byte[] computerModel() throws Exception {
        return Files.readAllBytes(TestResources.path("/models/computer.tflite"));
    }
}
