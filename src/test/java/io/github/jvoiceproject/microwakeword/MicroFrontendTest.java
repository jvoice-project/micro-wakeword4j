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

import io.github.jvoiceproject.microwakeword.internal.tflite.frontend.MicroFrontend;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

public class MicroFrontendTest {

    @ParameterizedTest
    @ValueSource(strings = {"hey_openhab_1.wav", "hey_peppa_pig.wav"})
    public void matchesCFrontend(String wav) throws IOException {
        int[][] expected = TestResources.referenceFeatures(wav);
        List<int[]> actual = features(new MicroFrontend(), TestResources.wav(wav));

        assertEquals(expected.length, actual.size());
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual.get(i), "frame " + i);
        }
    }

    @Test
    public void firstFeaturesNeedAFullWindow() {
        MicroFrontend frontend = new MicroFrontend();
        int[] features = new int[MicroFrontend.FEATURE_COUNT];
        short[] frame = new short[MicroFrontend.STEP_SIZE];

        assertFalse(frontend.process(frame, features));
        assertFalse(frontend.process(frame, features));
        assertTrue(frontend.process(frame, features));
    }

    @Test
    public void resetStartsANewStream() throws IOException {
        short[] samples = TestResources.wav("hey_openhab_1.wav");
        MicroFrontend frontend = new MicroFrontend();
        Random random = new Random(1);
        short[] noise = new short[MicroFrontend.SAMPLE_RATE];
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (short) (random.nextGaussian() * 3000);
        }

        features(frontend, noise);
        frontend.reset();
        List<int[]> actual = features(frontend, samples);

        int[][] expected = TestResources.referenceFeatures("hey_openhab_1.wav");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual.get(i), "frame " + i);
        }
    }

    private static List<int[]> features(MicroFrontend frontend, short[] samples) {
        List<int[]> frames = new ArrayList<>();
        for (short[] frame : TestResources.frames(samples)) {
            int[] features = new int[MicroFrontend.FEATURE_COUNT];
            if (frontend.process(frame, features)) {
                frames.add(features);
            }
        }
        return frames;
    }
}
