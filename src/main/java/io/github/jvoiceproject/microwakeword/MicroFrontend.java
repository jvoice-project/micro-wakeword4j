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

import java.util.Arrays;

/**
 * The TensorFlow Lite Micro audio frontend, which turns 16 kHz audio into the 40 log mel energies
 * per 10 ms that microWakeWord models are trained on.
 *
 * <p>A port of {@code tensorflow/lite/experimental/microfrontend} with the settings of
 * microWakeWord: a 30 ms Hann window every 10 ms, a 40 channel filterbank from 125 to 7500 Hz,
 * noise reduction, PCAN gain control and log scaling. The processing uses integer math only and
 * matches the C library bit for bit. Only the lookup tables are computed with floating point math,
 * once. They match C builds that round every floating point operation. Builds that fuse
 * multiply-adds, as clang does on arm64 by default, compute a slightly different filterbank, which
 * changes the top features a little on some inputs.
 *
 * @author Dan Cunningham - Initial contribution
 */
final class MicroFrontend {
    static final int SAMPLE_RATE = 16000;
    static final int STEP_SIZE = 160;
    static final int FEATURE_COUNT = 40;

    private static final int WINDOW_SIZE = 480;
    private static final int WINDOW_BITS = 12;
    private static final int FFT_SIZE = 512;
    private static final int SPECTRUM_SIZE = FFT_SIZE / 2 + 1;
    private static final int FILTERBANK_BITS = 12;
    private static final float LOWER_BAND_LIMIT = 125f;
    private static final float UPPER_BAND_LIMIT = 7500f;
    private static final int NOISE_REDUCTION_BITS = 14;
    private static final int SMOOTHING_BITS = 10;
    private static final long EVEN_SMOOTHING = (long) (0.025f * (1 << NOISE_REDUCTION_BITS));
    private static final long ODD_SMOOTHING = (long) (0.06f * (1 << NOISE_REDUCTION_BITS));
    private static final long MIN_SIGNAL_REMAINING = (long) (0.05f * (1 << NOISE_REDUCTION_BITS));
    private static final float PCAN_STRENGTH = 0.95f;
    private static final float PCAN_OFFSET = 80f;
    private static final int PCAN_GAIN_BITS = 21;
    private static final int PCAN_SNR_BITS = 12;
    private static final int PCAN_OUTPUT_BITS = 6;
    private static final int WIDE_DYNAMIC_FUNCTION_BITS = 32;
    private static final int LOG_SCALE_SHIFT = 6;
    private static final int LOG_SCALE_BITS = 16;
    private static final int LOG_SEGMENTS_BITS = 7;
    private static final long LOG_COEFFICIENT = 45426;
    private static final long UINT32_MASK = 0xFFFFFFFFL;

    /** Bits gained by the FFT and filterbank, removed again before the log. */
    private static final int CORRECTION_BITS = msb32(FFT_SIZE) - 1 - FILTERBANK_BITS / 2;

    private static final int PCAN_SNR_SHIFT = PCAN_GAIN_BITS - CORRECTION_BITS - PCAN_SNR_BITS;

    private static final int[] LOG_LUT = {
        0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163, 2329, 2490, 2646, 2797,
        2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848, 3960, 4068, 4172, 4272, 4368, 4460, 4549,
        4633, 4714, 4791, 4864, 4934, 5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444,
        5477, 5507, 5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633, 5626,
        5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370, 5332, 5291, 5249, 5203,
        5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762, 4697, 4630, 4561, 4490, 4416, 4341, 4264,
        4184, 4103, 4020, 3935, 3848, 3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875,
        2768, 2659, 2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224, 1094,
        963, 830, 695, 559, 421, 282, 142, 0, 0
    };

    private static final short[] WINDOW = window();
    private static final Fft FFT = new Fft(FFT_SIZE);
    private static final Filterbank FILTERBANK = new Filterbank();
    private static final short[] GAIN_LUT = gainLut();

    private final short[] input = new short[WINDOW_SIZE];
    private int inputUsed;
    private final long[] noiseEstimate = new long[FEATURE_COUNT];

    private final Fft.Workspace fftWorkspace = FFT.workspace();
    private final short[] fftInput = new short[FFT_SIZE];
    private final short[] spectrumRe = new short[SPECTRUM_SIZE];
    private final short[] spectrumIm = new short[SPECTRUM_SIZE];
    private final int[] energy = new int[FILTERBANK.energySize];
    private final long[] accumulated = new long[FEATURE_COUNT + 1];
    private final long[] signal = new long[FEATURE_COUNT];

    /**
     * Adds the next 10 ms of audio.
     *
     * @param samples {@link #STEP_SIZE} signed 16-bit samples
     * @param features receives {@link #FEATURE_COUNT} features between 0 and 65535
     * @return false while the first window is still filling, so {@code features} is unchanged
     */
    boolean process(short[] samples, int[] features) {
        System.arraycopy(samples, 0, input, inputUsed, STEP_SIZE);
        inputUsed += STEP_SIZE;
        if (inputUsed < WINDOW_SIZE) {
            return false;
        }
        int inputShift = applyWindow();
        System.arraycopy(input, STEP_SIZE, input, 0, WINDOW_SIZE - STEP_SIZE);
        inputUsed -= STEP_SIZE;

        FFT.transform(fftWorkspace, fftInput, spectrumRe, spectrumIm);
        for (int i = FILTERBANK.startIndex; i < FILTERBANK.endIndex; i++) {
            energy[i] = spectrumRe[i] * spectrumRe[i] + spectrumIm[i] * spectrumIm[i];
        }
        FILTERBANK.accumulate(energy, accumulated);
        for (int i = 0; i < FEATURE_COUNT; i++) {
            signal[i] = sqrt64(accumulated[i + 1]) >>> inputShift;
        }
        reduceNoise();
        controlGain();
        for (int i = 0; i < FEATURE_COUNT; i++) {
            long value = (signal[i] << CORRECTION_BITS) & UINT32_MASK;
            features[i] = (int) Math.min(value > 1 ? log(value) : 0, 0xFFFF);
        }
        return true;
    }

    /** Starts a new stream. */
    void reset() {
        Arrays.fill(input, (short) 0);
        inputUsed = 0;
        Arrays.fill(noiseEstimate, 0);
    }

    /**
     * Writes the windowed and scaled input of the FFT.
     *
     * @return the shift that scaled the samples up to use the 16-bit range
     */
    private int applyWindow() {
        short maxAbs = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            short value = (short) ((input[i] * WINDOW[i]) >> WINDOW_BITS);
            fftInput[i] = value;
            short abs = value < 0 ? (short) -value : value;
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        int shift = 15 - msb32(maxAbs);
        for (int i = 0; i < WINDOW_SIZE; i++) {
            fftInput[i] = (short) ((fftInput[i] & 0xFFFF) << shift);
        }
        return shift;
    }

    private void reduceNoise() {
        for (int i = 0; i < FEATURE_COUNT; i++) {
            long smoothing = (i & 1) == 0 ? EVEN_SMOOTHING : ODD_SMOOTHING;
            long scaledUp = (signal[i] << SMOOTHING_BITS) & UINT32_MASK;
            long estimate =
                    ((scaledUp * smoothing
                                            + noiseEstimate[i]
                                                    * ((1 << NOISE_REDUCTION_BITS) - smoothing))
                                    >>> NOISE_REDUCTION_BITS)
                            & UINT32_MASK;
            noiseEstimate[i] = estimate;
            long floor = (signal[i] * MIN_SIGNAL_REMAINING) >>> NOISE_REDUCTION_BITS;
            long subtracted = (scaledUp - Math.min(estimate, scaledUp)) >>> SMOOTHING_BITS;
            signal[i] = Math.max(subtracted, floor);
        }
    }

    private void controlGain() {
        for (int i = 0; i < FEATURE_COUNT; i++) {
            long gain = wideDynamicFunction(noiseEstimate[i]) & UINT32_MASK;
            long snr = ((signal[i] * gain) >>> PCAN_SNR_SHIFT) & UINT32_MASK;
            signal[i] =
                    snr < (2L << PCAN_SNR_BITS)
                            ? (snr * snr) >>> (2 + 2 * PCAN_SNR_BITS - PCAN_OUTPUT_BITS)
                            : (snr >>> (PCAN_SNR_BITS - PCAN_OUTPUT_BITS))
                                    - (1 << PCAN_OUTPUT_BITS);
        }
    }

    /** Piecewise quadratic approximation of the PCAN gain for a noise estimate. */
    private static short wideDynamicFunction(long x) {
        if (x <= 2) {
            return GAIN_LUT[(int) x];
        }
        int interval = msb32((int) x);
        int base = 4 * interval - 6;
        int frac = (int) ((interval < 11 ? x << (11 - interval) : x >>> (interval - 11)) & 0x3FF);
        int result = (GAIN_LUT[base + 2] * frac) >> 5;
        result += GAIN_LUT[base + 1] << 5;
        result *= frac;
        result = (result + (1 << 14)) >> 15;
        result += GAIN_LUT[base];
        return (short) result;
    }

    /** Natural log in fixed point, scaled by {@link #LOG_SCALE_SHIFT}. */
    private static long log(long x) {
        int integer = msb32((int) x) - 1;
        long log2 = (((long) integer << LOG_SCALE_BITS) + log2Fraction(x, integer)) & UINT32_MASK;
        long round = 1L << (LOG_SCALE_BITS - 1);
        long loge = ((LOG_COEFFICIENT * log2 + round) >>> LOG_SCALE_BITS) & UINT32_MASK;
        return ((((loge << LOG_SCALE_SHIFT) & UINT32_MASK) + round) & UINT32_MASK)
                >>> LOG_SCALE_BITS;
    }

    private static long log2Fraction(long x, int log2x) {
        int frac = (int) (x - (1L << log2x));
        if (log2x < LOG_SCALE_BITS) {
            frac <<= LOG_SCALE_BITS - log2x;
        } else {
            frac >>= log2x - LOG_SCALE_BITS;
        }
        int segment = frac >>> (LOG_SCALE_BITS - LOG_SEGMENTS_BITS);
        int c0 = LOG_LUT[segment];
        int c1 = LOG_LUT[segment + 1];
        int segmentBase = ((1 << LOG_SCALE_BITS) >> LOG_SEGMENTS_BITS) * segment;
        int relative = ((c1 - c0) * (frac - segmentBase)) >> LOG_SCALE_BITS;
        return (frac + c0 + relative) & UINT32_MASK;
    }

    /** Integer square root rounded to the nearest value, of an unsigned 64-bit value. */
    private static long sqrt64(long num) {
        if ((num >>> 32) == 0) {
            return sqrt32((int) num);
        }
        long result = 0;
        int maxBit = Long.numberOfLeadingZeros(num) | 1;
        long bit = 1L << (63 - maxBit);
        for (int i = (63 - maxBit) / 2 + 1; i > 0; i--) {
            if (Long.compareUnsigned(num, result + bit) >= 0) {
                num -= result + bit;
                result = (result >>> 1) + bit;
            } else {
                result >>>= 1;
            }
            bit >>>= 2;
        }
        if (Long.compareUnsigned(num, result) > 0 && result != UINT32_MASK) {
            result++;
        }
        return result & UINT32_MASK;
    }

    private static long sqrt32(int num) {
        if (num == 0) {
            return 0;
        }
        int result = 0;
        int maxBit = Integer.numberOfLeadingZeros(num) | 1;
        int bit = 1 << (31 - maxBit);
        for (int i = (31 - maxBit) / 2 + 1; i > 0; i--) {
            if (Integer.compareUnsigned(num, result + bit) >= 0) {
                num -= result + bit;
                result = (result >>> 1) + bit;
            } else {
                result >>>= 1;
            }
            bit >>>= 2;
        }
        if (Integer.compareUnsigned(num, result) > 0 && result != 0xFFFF) {
            result++;
        }
        return result & 0xFFFF;
    }

    /** Position of the highest set bit counting from 1, of an unsigned value. */
    private static int msb32(int n) {
        return 32 - Integer.numberOfLeadingZeros(n);
    }

    private static short[] window() {
        short[] coefficients = new short[WINDOW_SIZE];
        float arg = (float) Math.PI * 2f / WINDOW_SIZE;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float value = 0.5f - 0.5f * (float) StrictMath.cos(arg * (i + 0.5f));
            coefficients[i] = (short) Math.floor(value * (1 << WINDOW_BITS) + 0.5f);
        }
        return coefficients;
    }

    private static short[] gainLut() {
        int inputBits = SMOOTHING_BITS - CORRECTION_BITS;
        short[] lut = new short[4 * WIDE_DYNAMIC_FUNCTION_BITS - 3];
        lut[0] = gain(inputBits, 0);
        lut[1] = gain(inputBits, 1);
        for (int interval = 2; interval <= WIDE_DYNAMIC_FUNCTION_BITS; interval++) {
            long x0 = 1L << (interval - 1);
            long x1 = x0 + (x0 >> 1);
            long x2 = interval == WIDE_DYNAMIC_FUNCTION_BITS ? x0 + (x0 - 1) : 2 * x0;
            short y0 = gain(inputBits, x0);
            int diff1 = gain(inputBits, x1) - y0;
            int diff2 = gain(inputBits, x2) - y0;
            int a1 = 4 * diff1 - diff2;
            int a2 = diff2 - a1;
            int base = 4 * interval - 6;
            lut[base] = y0;
            lut[base + 1] = (short) a1;
            lut[base + 2] = (short) a2;
        }
        return lut;
    }

    private static short gain(int inputBits, long x) {
        float xAsFloat = (float) x / (float) (1L << inputBits);
        float gain =
                (float) (1L << PCAN_GAIN_BITS)
                        * (float) StrictMath.pow(xAsFloat + PCAN_OFFSET, -PCAN_STRENGTH);
        return gain > Short.MAX_VALUE ? Short.MAX_VALUE : (short) (int) (gain + 0.5f);
    }

    private static float freqToMel(float frequency) {
        return 1127f * (float) StrictMath.log1p(frequency / 700f);
    }

    /** Triangular mel filters, each sharing its slopes with the neighboring channels. */
    private static final class Filterbank {
        private static final int INDEX_ALIGNMENT = 2;
        private static final int CHANNEL_BLOCK_SIZE = 4;

        private final int[] frequencyStarts = new int[FEATURE_COUNT + 1];
        private final int[] weightStarts = new int[FEATURE_COUNT + 1];
        private final int[] widths = new int[FEATURE_COUNT + 1];
        private final short[] weights;
        private final short[] unweights;
        private final int startIndex;
        private final int endIndex;
        private final int energySize;

        Filterbank() {
            int channels = FEATURE_COUNT + 1;
            float[] centers = new float[channels];
            float melLow = freqToMel(LOWER_BAND_LIMIT);
            float melSpacing = (freqToMel(UPPER_BAND_LIMIT) - melLow) / channels;
            for (int i = 0; i < channels; i++) {
                centers[i] = melLow + melSpacing * (i + 1);
            }
            float hzPerBin = 0.5f * SAMPLE_RATE / (SPECTRUM_SIZE - 1f);
            startIndex = (int) (1.5f + LOWER_BAND_LIMIT / hzPerBin);

            int[] actualStarts = new int[channels];
            int[] actualWidths = new int[channels];
            int frequencyStart = startIndex;
            int weightStart = 0;
            boolean needsZeros = false;
            for (int channel = 0; channel < channels; channel++) {
                int frequency = frequencyStart;
                while (freqToMel(frequency * hzPerBin) <= centers[channel]) {
                    frequency++;
                }
                int width = frequency - frequencyStart;
                actualStarts[channel] = frequencyStart;
                actualWidths[channel] = width;
                if (width == 0) {
                    // an empty channel reads a block of zero weights
                    frequencyStarts[channel] = 0;
                    weightStarts[channel] = 0;
                    widths[channel] = CHANNEL_BLOCK_SIZE;
                    if (!needsZeros) {
                        needsZeros = true;
                        for (int j = 0; j < channel; j++) {
                            weightStarts[j] += CHANNEL_BLOCK_SIZE;
                        }
                        weightStart += CHANNEL_BLOCK_SIZE;
                    }
                } else {
                    int alignedStart = frequencyStart / INDEX_ALIGNMENT * INDEX_ALIGNMENT;
                    int alignedWidth = frequencyStart - alignedStart + width;
                    int paddedWidth =
                            ((alignedWidth - 1) / CHANNEL_BLOCK_SIZE + 1) * CHANNEL_BLOCK_SIZE;
                    frequencyStarts[channel] = alignedStart;
                    weightStarts[channel] = weightStart;
                    widths[channel] = paddedWidth;
                    weightStart += paddedWidth;
                }
                frequencyStart = frequency;
            }

            weights = new short[weightStart];
            unweights = new short[weightStart];
            int end = 0;
            int size = 0;
            for (int channel = 0; channel < channels; channel++) {
                int frequency = actualStarts[channel];
                int offset = frequency - frequencyStarts[channel];
                float denominator = channel == 0 ? melLow : centers[channel - 1];
                for (int j = 0; j < actualWidths[channel]; j++, frequency++) {
                    float weight =
                            (centers[channel] - freqToMel(frequency * hzPerBin))
                                    / (centers[channel] - denominator);
                    int index = weightStarts[channel] + offset + j;
                    weights[index] = (short) Math.floor(weight * (1 << FILTERBANK_BITS) + 0.5f);
                    unweights[index] =
                            (short) Math.floor((1f - weight) * (1 << FILTERBANK_BITS) + 0.5f);
                }
                end = Math.max(end, frequency);
                size = Math.max(size, frequencyStarts[channel] + widths[channel]);
            }
            if (end >= SPECTRUM_SIZE) {
                throw new IllegalStateException("Filterbank exceeds the spectrum");
            }
            endIndex = end;
            energySize = Math.max(size, SPECTRUM_SIZE);
        }

        /** Sums the weighted energies of every channel's rising and falling slopes. */
        void accumulate(int[] energy, long[] work) {
            long weighted = 0;
            long unweighted = 0;
            for (int channel = 0; channel <= FEATURE_COUNT; channel++) {
                int frequency = frequencyStarts[channel];
                int weight = weightStarts[channel];
                for (int j = 0; j < widths[channel]; j++) {
                    long magnitude = energy[frequency + j];
                    weighted += weights[weight + j] * magnitude;
                    unweighted += unweights[weight + j] * magnitude;
                }
                work[channel] = weighted;
                weighted = unweighted;
                unweighted = 0;
            }
        }
    }
}
