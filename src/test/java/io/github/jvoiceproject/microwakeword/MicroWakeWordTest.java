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
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

public class MicroWakeWordTest {

    @ParameterizedTest
    @ValueSource(
            strings = {
                "alexa",
                "hey_jarvis",
                "hey_jeeves",
                "hey_mycroft",
                "hey_openhab",
                "computer"
            })
    public void matchesReferenceKernels(String name) throws Exception {
        WakeWordModel model = model(name);
        Map<String, int[]> reference = TestResources.referenceOutputs(name);

        assertEquals(TestResources.WAVS, List.copyOf(reference.keySet()));
        for (String wav : TestResources.WAVS) {
            int[] actual = outputs(new MicroWakeWord(model), TestResources.wav(wav));
            assertArrayEquals(reference.get(wav), actual, name + " on " + wav);
        }
    }

    @ParameterizedTest
    @EnumSource(BuiltinModel.class)
    public void detectsOnlyItsWakeWord(BuiltinModel builtin) throws IOException {
        MicroWakeWord detector = new MicroWakeWord(WakeWordModel.load(builtin));

        for (String wav : TestResources.WAVS) {
            detector.reset();
            boolean expected = wav.startsWith(builtin.getId() + "_");
            assertEquals(expected, detects(detector, TestResources.wav(wav)), wav);
        }
    }

    @Test
    public void modelWithTwoFrameInputs() throws Exception {
        WakeWordModel model = model("computer");
        MicroWakeWord detector = new MicroWakeWord(model);

        assertEquals(2, model.stride);
        assertTrue(detects(detector, TestResources.wav("computer.wav")));
        detector.reset();
        assertFalse(detects(detector, TestResources.wav("hey_peppa_pig.wav")));
    }

    @Test
    public void probabilityCrossesTheCutoffOnDetection() throws IOException {
        WakeWordModel model = WakeWordModel.load(BuiltinModel.HEY_JARVIS);
        MicroWakeWord detector = new MicroWakeWord(model);
        float cutoff = (float) ((int) (model.getProbabilityCutoff() * 255) / 255.0);

        assertEquals(0f, detector.getProbability());
        for (short[] frame : TestResources.frames(TestResources.wav("hey_jarvis_1.wav"))) {
            if (detector.process(frame)) {
                assertTrue(detector.getProbability() > cutoff);
                return;
            }
            assertTrue(detector.getProbability() <= cutoff);
        }
        throw new AssertionError("Hey Jarvis was not detected");
    }

    @ParameterizedTest
    @EnumSource(BuiltinModel.class)
    public void detectsLikeESPHome(BuiltinModel builtin) throws IOException {
        WakeWordModel model = WakeWordModel.load(builtin);
        int window = model.getSlidingWindowSize();
        // as in ESPHome's micro_wake_word component
        int cutoff = (int) (model.getProbabilityCutoff() * 255);

        for (String wav : TestResources.WAVS) {
            MicroWakeWord detector = new MicroWakeWord(model);
            MicroWakeWord outputs = new MicroWakeWord(model);
            List<Integer> recent = new ArrayList<>();
            for (short[] frame : TestResources.frames(TestResources.wav(wav))) {
                boolean detected = detector.process(frame);
                int output = outputs.infer(frame);
                if (output != MicroWakeWord.NO_OUTPUT) {
                    recent.add(output);
                    int sum =
                            recent.stream()
                                    .skip(Math.max(recent.size() - window, 0))
                                    .mapToInt(i -> i)
                                    .sum();
                    boolean expected = recent.size() >= window && sum > cutoff * window;
                    assertEquals(expected, detected, wav + " at output " + recent.size());
                }
            }
        }
    }

    @Test
    public void estimatesWhereTheWakeWordEnded() throws IOException {
        int totalError = 0;
        int clips = 0;
        for (BuiltinModel builtin : BuiltinModel.values()) {
            for (String wav : List.of(builtin.getId() + "_1.wav", builtin.getId() + "_2.wav")) {
                MicroWakeWord detector = new MicroWakeWord(WakeWordModel.load(builtin));
                short[][] frames = TestResources.frames(TestResources.wav(wav));
                int detection = 0;
                while (!detector.process(frames[detection])) {
                    detection++;
                }

                int estimatedEnd = detection - detector.getFramesAfterWakeWord();
                int error = Math.abs(estimatedEnd - speechEnd(frames));
                assertTrue(estimatedEnd <= detection, wav);
                assertTrue(error <= 25, wav + " is off by " + error + " frames");
                totalError += error;
                clips++;
            }
        }
        // the detections themselves are 160 ms late on average
        assertTrue(totalError <= 12 * clips, "off by " + totalError / clips + " frames on average");
    }

    @Test
    public void estimatesTheEndOfTheLatestDetectionWithALowCutoff() throws IOException {
        // at this cutoff Hey Mycroft also detects alexa_1, where no output rises above 0.5
        MicroWakeWord detector =
                new MicroWakeWord(WakeWordModel.load(BuiltinModel.HEY_MYCROFT), 0.25);
        assertTrue(detects(detector, TestResources.wav("hey_mycroft_1.wav")));
        for (int i = 0; i < 300; i++) {
            detector.process(new short[MicroWakeWord.FRAME_SIZE]);
        }

        short[][] alexa = TestResources.frames(TestResources.wav("alexa_1.wav"));
        int detection = 0;
        while (!detector.process(alexa[detection])) {
            detection++;
        }
        int estimatedEnd = detection - detector.getFramesAfterWakeWord();
        assertTrue(
                estimatedEnd >= 0, "the estimate is " + -estimatedEnd + " frames before alexa_1");
    }

    @Test
    public void framesAfterWakeWordStartOverOnReset() throws IOException {
        MicroWakeWord detector = new MicroWakeWord(WakeWordModel.load(BuiltinModel.HEY_JARVIS));

        assertEquals(0, detector.getFramesAfterWakeWord());
        assertTrue(detects(detector, TestResources.wav("hey_jarvis_1.wav")));
        assertTrue(detector.getFramesAfterWakeWord() > 0);
        detector.reset();
        assertEquals(0, detector.getFramesAfterWakeWord());
    }

    @Test
    public void cutoffCanBeChanged() throws IOException {
        WakeWordModel model = WakeWordModel.load(BuiltinModel.HEY_MYCROFT);
        short[] alexa = TestResources.wav("alexa_1.wav");
        short[] heyMycroft = TestResources.wav("hey_mycroft_2.wav");

        // the average probability of Hey Mycroft peaks at 0.28 on this clip
        assertFalse(detects(new MicroWakeWord(model), alexa));
        assertTrue(detects(new MicroWakeWord(model, 0.25), alexa));
        assertFalse(detects(new MicroWakeWord(model, 1), heyMycroft));
        assertEquals(0.25, new MicroWakeWord(model, 0.25).getProbabilityCutoff());
        assertEquals(model.getProbabilityCutoff(), new MicroWakeWord(model).getProbabilityCutoff());
    }

    @Test
    public void rejectsInvalidCutoffs() throws IOException {
        WakeWordModel model = WakeWordModel.load(BuiltinModel.HEY_JARVIS);

        assertThrows(IllegalArgumentException.class, () -> new MicroWakeWord(model, -0.1));
        assertThrows(IllegalArgumentException.class, () -> new MicroWakeWord(model, 1.5));
        assertThrows(IllegalArgumentException.class, () -> new MicroWakeWord(model, Double.NaN));
    }

    @Test
    public void resetStartsANewStream() throws Exception {
        MicroWakeWord detector = new MicroWakeWord(model("hey_jarvis"));

        outputs(detector, TestResources.wav("hey_jarvis_2.wav"));
        detector.reset();

        assertEquals(0f, detector.getProbability());
        assertArrayEquals(
                TestResources.referenceOutputs("hey_jarvis").get("hey_jarvis_1.wav"),
                outputs(detector, TestResources.wav("hey_jarvis_1.wav")));
    }

    @Test
    public void instancesSharingAModelAreIndependent() throws Exception {
        WakeWordModel model = model("hey_jarvis");
        MicroWakeWord first = new MicroWakeWord(model);
        MicroWakeWord second = new MicroWakeWord(model);
        short[][] jarvis = TestResources.frames(TestResources.wav("hey_jarvis_1.wav"));
        short[][] alexa = TestResources.frames(TestResources.wav("alexa_1.wav"));

        List<Integer> firstOutputs = new ArrayList<>();
        for (int i = 0; i < jarvis.length; i++) {
            int output = first.infer(jarvis[i]);
            if (output != MicroWakeWord.NO_OUTPUT) {
                firstOutputs.add(output);
            }
            if (i < alexa.length) {
                second.infer(alexa[i]);
            }
        }

        int[] expected = TestResources.referenceOutputs("hey_jarvis").get("hey_jarvis_1.wav");
        assertArrayEquals(expected, firstOutputs.stream().mapToInt(Integer::intValue).toArray());
    }

    @ParameterizedTest
    @EnumSource(BuiltinModel.class)
    public void noiseIsNotAWakeWord(BuiltinModel builtin) throws IOException {
        Random random = new Random(1);
        short[] noise = new short[5 * MicroWakeWord.SAMPLE_RATE];
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (short) (random.nextGaussian() * 1500);
        }

        assertFalse(detects(new MicroWakeWord(WakeWordModel.load(builtin)), noise));
    }

    @Test
    public void rejectsWrongFrameSize() throws IOException {
        MicroWakeWord detector = new MicroWakeWord(WakeWordModel.load(BuiltinModel.ALEXA));

        assertThrows(IllegalArgumentException.class, () -> detector.process(new short[320]));
    }

    private static WakeWordModel model(String name) throws IOException, URISyntaxException {
        for (BuiltinModel builtin : BuiltinModel.values()) {
            if (builtin.getId().equals(name)) {
                return WakeWordModel.load(builtin);
            }
        }
        return WakeWordModel.load(TestResources.path("/models/" + name + ".json"));
    }

    private static int[] outputs(MicroWakeWord detector, short[] samples) {
        List<Integer> outputs = new ArrayList<>();
        for (short[] frame : TestResources.frames(samples)) {
            int output = detector.infer(frame);
            if (output != MicroWakeWord.NO_OUTPUT) {
                outputs.add(output);
            }
        }
        return outputs.stream().mapToInt(Integer::intValue).toArray();
    }

    /** The last frame that is louder than a tenth of the loudest frame. */
    private static int speechEnd(short[][] frames) {
        double[] levels = new double[frames.length];
        double loudest = 0;
        for (int i = 0; i < frames.length; i++) {
            double sum = 0;
            for (short sample : frames[i]) {
                sum += sample * sample;
            }
            levels[i] = Math.sqrt(sum / frames[i].length);
            loudest = Math.max(loudest, levels[i]);
        }
        int end = 0;
        for (int i = 0; i < frames.length; i++) {
            if (levels[i] > loudest / 10) {
                end = i;
            }
        }
        return end;
    }

    private static boolean detects(MicroWakeWord detector, short[] samples) {
        for (short[] frame : TestResources.frames(samples)) {
            if (detector.process(frame)) {
                return true;
            }
        }
        return false;
    }
}
