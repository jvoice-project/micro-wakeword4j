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
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Audio and reference data of the tests.
 *
 * <p>The WAV files come from the pymicro-wakeword tests, except the Hey Jeeves and Hey openHAB
 * ones, which are synthesized clips that their models did not train on. The reference files hold
 * the output of the microfrontend C library and of the LiteRT reference kernels for them, see
 * {@code scripts/generate_reference.py}.
 */
public final class TestResources {
    public static final List<String> WAVS =
            List.of(
                    "alexa_1.wav",
                    "alexa_2.wav",
                    "computer.wav",
                    "hey_jarvis_1.wav",
                    "hey_jarvis_2.wav",
                    "hey_jeeves_1.wav",
                    "hey_jeeves_2.wav",
                    "hey_mycroft_1.wav",
                    "hey_mycroft_2.wav",
                    "hey_openhab_1.wav",
                    "hey_openhab_2.wav",
                    "hey_peppa_pig.wav");

    private TestResources() {}

    /** Samples of a 16 kHz 16-bit mono WAV file. */
    public static short[] wav(String name) throws IOException {
        byte[] wav = bytes("/audio/" + name);
        ByteBuffer buffer = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        int offset = 12;
        while (!"data".equals(new String(wav, offset, 4, StandardCharsets.US_ASCII))) {
            if ("fmt ".equals(new String(wav, offset, 4, StandardCharsets.US_ASCII))) {
                assertEquals(1, buffer.getShort(offset + 10), "channels of " + name);
                assertEquals(16000, buffer.getInt(offset + 12), "sample rate of " + name);
                assertEquals(16, buffer.getShort(offset + 22), "bits per sample of " + name);
            }
            int size = buffer.getInt(offset + 4);
            // chunks are padded to an even size
            offset += 8 + size + (size & 1);
        }
        short[] samples = new short[buffer.getInt(offset + 4) / 2];
        buffer.position(offset + 8);
        buffer.asShortBuffer().get(samples);
        return samples;
    }

    /** The complete frames of a stream. */
    public static short[][] frames(short[] samples) {
        short[][] frames = new short[samples.length / MicroWakeWord.FRAME_SIZE][];
        for (int i = 0; i < frames.length; i++) {
            int start = i * MicroWakeWord.FRAME_SIZE;
            frames[i] = Arrays.copyOfRange(samples, start, start + MicroWakeWord.FRAME_SIZE);
        }
        return frames;
    }

    /** Features of the C frontend, one line of 40 values per frame. */
    public static int[][] referenceFeatures(String wav) throws IOException {
        String name = "/reference/features-" + wav.replace(".wav", ".txt");
        return lines(name).stream().map(TestResources::ints).toArray(int[][]::new);
    }

    /** Quantized model outputs of the LiteRT reference kernels, per WAV file. */
    static Map<String, int[]> referenceOutputs(String model) throws IOException {
        Map<String, int[]> outputs = new LinkedHashMap<>();
        for (String line : lines("/reference/" + model + ".txt")) {
            int space = line.indexOf(' ');
            outputs.put(line.substring(0, space), ints(line.substring(space + 1)));
        }
        return outputs;
    }

    static Path path(String resource) throws URISyntaxException {
        return Path.of(TestResources.class.getResource(resource).toURI());
    }

    private static List<String> lines(String name) throws IOException {
        return new String(bytes(name), StandardCharsets.UTF_8).lines().toList();
    }

    private static int[] ints(String line) {
        return Arrays.stream(line.trim().split(" ")).mapToInt(Integer::parseInt).toArray();
    }

    private static byte[] bytes(String name) throws IOException {
        try (InputStream in = TestResources.class.getResourceAsStream(name)) {
            assertNotNull(in, name);
            return in.readAllBytes();
        }
    }
}
