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

import io.github.jvoiceproject.microwakeword.internal.tflite.core.Interpreter;
import io.github.jvoiceproject.microwakeword.internal.tflite.core.Program;
import io.github.jvoiceproject.microwakeword.internal.tflite.frontend.MicroFrontend;

import java.util.Arrays;

/**
 * microWakeWord
 *
 * <p>Plain Java inference of <a
 * href="https://github.com/OHF-Voice/micro-wake-word">microWakeWord</a> models for 16 kHz mono
 * audio. Audio is turned into features by a port of the TensorFlow Lite Micro audio frontend, and
 * the quantized streaming model runs on integer kernels that match TensorFlow Lite's reference
 * kernels bit for bit.
 *
 * <p>Each {@link #FRAME_SIZE} sample frame updates the features. Every few frames, as many as the
 * model takes per inference, the model returns the probability that the wake word was just spoken.
 * The wake word counts as detected when the average of the last {@link
 * WakeWordModel#getSlidingWindowSize()} probabilities is above the cutoff, which is the model's
 * {@link WakeWordModel#getProbabilityCutoff() cutoff} unless another one is given.
 *
 * <p>Detection follows ESPHome, so that a cutoff behaves the same as on ESPHome devices: a model
 * output of 255 counts as a probability of 1, and the cutoff is rounded down to a multiple of
 * 1/255.
 *
 * <p>The detector keeps state between frames, so an instance should be dedicated to a single
 * continuous audio stream. Call {@link #reset()} after a detection, and before reusing it for a new
 * stream. All methods are safe to be called concurrently.
 *
 * @author Dan Cunningham - Initial contribution
 */
public class MicroWakeWord {
    /** Supported sample rate in Hz. */
    public static final int SAMPLE_RATE = MicroFrontend.SAMPLE_RATE;

    /** Number of samples per frame, 10 ms at {@link #SAMPLE_RATE}. */
    public static final int FRAME_SIZE = MicroFrontend.STEP_SIZE;

    /** Scale from frontend output to the feature values that models are trained on. */
    private static final double FEATURE_SCALE = 0.0390625;

    /** Returned by {@link #infer(short[])} for frames that don't complete a model input. */
    public static final int NO_OUTPUT = Integer.MIN_VALUE;

    /** Model output that counts as a probability of 1, as in ESPHome. */
    private static final int MAX_OUTPUT = 255;

    /** Probability above which a single model output marks the end of the wake word. */
    private static final double END_PROBABILITY = 0.5;

    private final WakeWordModel model;
    private final double probabilityCutoff;

    /** The cutoff as a model output, rounded down like ESPHome does. */
    private final int outputCutoff;

    /**
     * The output of {@link #END_PROBABILITY}, or the cutoff when that is lower, so that every
     * window above the cutoff holds an output above it.
     */
    private final int endOutput;

    private final MicroFrontend frontend = new MicroFrontend();
    private final Interpreter interpreter;
    private final int[] features = new int[MicroFrontend.FEATURE_COUNT];
    private final int[] outputs;
    private int framesInInput;
    private int nextOutput;
    private int outputCount;
    private float probability;
    private long frameCount;

    /** Frame where a single output last rose above endOutput, or -1. */
    private long endFrame = -1;

    private boolean aboveEndOutput;

    /** Creates a detector for one audio stream, using the model's probability cutoff. */
    public MicroWakeWord(WakeWordModel model) {
        this(model, model.getProbabilityCutoff());
    }

    /**
     * Creates a detector for one audio stream with its own probability cutoff. A lower cutoff
     * detects the wake word more easily, at the cost of more false detections.
     *
     * @param probabilityCutoff average probability above which the wake word counts as detected,
     *     rounded down to a multiple of 1/255
     * @throws IllegalArgumentException when the cutoff is not between 0 and 1
     */
    public MicroWakeWord(WakeWordModel model, double probabilityCutoff) {
        if (!(probabilityCutoff >= 0 && probabilityCutoff <= 1)) {
            throw new IllegalArgumentException(
                    "Probability cutoff must be between 0 and 1, got " + probabilityCutoff);
        }
        this.model = model;
        this.probabilityCutoff = probabilityCutoff;
        this.outputCutoff = (int) (probabilityCutoff * MAX_OUTPUT);
        this.endOutput = Math.min((int) (END_PROBABILITY * MAX_OUTPUT), outputCutoff);
        this.interpreter = new Interpreter(model.program);
        this.outputs = new int[model.getSlidingWindowSize()];
    }

    /**
     * Processes the next frame of a stream.
     *
     * @param frame {@link #FRAME_SIZE} signed 16-bit samples
     * @return true when the wake word is detected. It is only true on frames that complete a model
     *     input, and stays true on the following ones while the average probability stays above the
     *     cutoff, so call {@link #reset()} after a detection.
     * @throws IllegalArgumentException when the frame doesn't have {@link #FRAME_SIZE} samples
     */
    public synchronized boolean process(short[] frame) {
        int output = infer(frame);
        long currentFrame = frameCount++;
        if (output == NO_OUTPUT) {
            return false;
        }
        int value = output - model.program.outputZeroPoint;
        if (value > endOutput && !aboveEndOutput) {
            endFrame = currentFrame;
        }
        aboveEndOutput = value > endOutput;
        outputs[nextOutput] = value;
        nextOutput = (nextOutput + 1) % outputs.length;
        outputCount = Math.min(outputCount + 1, outputs.length);
        if (outputCount < outputs.length) {
            return false;
        }
        int sum = 0;
        for (int recent : outputs) {
            sum += recent;
        }
        probability = (float) (sum / ((double) MAX_OUTPUT * outputs.length));
        return sum > outputCutoff * outputs.length;
    }

    /**
     * Adds a frame to the model input and runs the model once the input is complete, without
     * detection. It is meant for comparing model outputs with reference data. A stream should use
     * either this method or {@link #process(short[])}, not both.
     *
     * @param frame {@link #FRAME_SIZE} signed 16-bit samples
     * @return the quantized model output, or {@link #NO_OUTPUT} when the frame doesn't complete a
     *     model input
     * @throws IllegalArgumentException when the frame doesn't have {@link #FRAME_SIZE} samples
     */
    public synchronized int infer(short[] frame) {
        if (frame.length != FRAME_SIZE) {
            throw new IllegalArgumentException(
                    "Expected " + FRAME_SIZE + " samples, got " + frame.length);
        }
        if (!frontend.process(frame, features)) {
            return NO_OUTPUT;
        }
        Program program = model.program;
        byte[] input = interpreter.input();
        int offset = framesInInput * features.length;
        for (int i = 0; i < features.length; i++) {
            double value =
                    features[i] * FEATURE_SCALE / program.inputScale + program.inputZeroPoint;
            input[offset + i] = (byte) Math.clamp((long) Math.rint(value), -128, 127);
        }
        if (++framesInInput < model.stride) {
            return NO_OUTPUT;
        }
        framesInInput = 0;
        return interpreter.invoke();
    }

    /**
     * The average probability of the last model outputs, between 0 and 1. It is 0 until the model
     * has returned as many outputs as its sliding window holds.
     */
    public synchronized float getProbability() {
        return probability;
    }

    /**
     * Estimates how many frames were processed after the wake word ended, meant to be read right
     * after {@link #process(short[])} returned true.
     *
     * <p>Averaging the probabilities delays a detection, typically by 100 to 250 ms after the end
     * of the word. A single model output already rises above 0.5 close to the end of the word, so
     * this counts the frames since the last output that did. That point usually lies within 100 ms
     * of the end of the word. With a cutoff below 0.5 it counts from the last output that rose
     * above the cutoff instead. It is 0 when no output did since the stream started.
     */
    public synchronized int getFramesAfterWakeWord() {
        return endFrame < 0 ? 0 : (int) Math.min(frameCount - 1 - endFrame, Integer.MAX_VALUE);
    }

    /** Clears the stream state, so the instance can be used for a new stream. */
    public synchronized void reset() {
        frontend.reset();
        interpreter.reset();
        Arrays.fill(outputs, 0);
        framesInInput = 0;
        nextOutput = 0;
        outputCount = 0;
        probability = 0;
        frameCount = 0;
        endFrame = -1;
        aboveEndOutput = false;
    }

    /**
     * The average probability above which this detector reports the wake word, as given. Detection
     * rounds it down to a multiple of 1/255.
     */
    public double getProbabilityCutoff() {
        return probabilityCutoff;
    }

    /** The model this detector runs. */
    public WakeWordModel getModel() {
        return model;
    }
}
