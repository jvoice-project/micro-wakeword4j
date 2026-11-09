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
package io.github.jvoiceproject.microwakeword.internal.tflite.core;

import java.io.IOException;

/**
 * A real number stored as a Q31 fixed-point value and a power of two exponent, used to rescale
 * quantized values with integer math only.
 *
 * <p>The conversion and the rounding match TensorFlow Lite's {@code QuantizeMultiplier} and its
 * default (double rounding) {@code MultiplyByQuantizedMultiplier}, so results are bit-exact with
 * its reference kernels.
 *
 * @param value Q31 significand
 * @param shift power of two exponent, positive for a left shift
 * @author Dan Cunningham - Initial contribution
 */
record Multiplier(int value, int shift) {

    /**
     * @throws IOException when the scale is negative, not finite, or too large to apply, which
     *     needs a shift of at most 30
     */
    static Multiplier of(double real) throws IOException {
        if (!(real >= 0) || real == Double.POSITIVE_INFINITY) {
            throw new IOException("Invalid quantization scales");
        }
        if (real == 0) {
            return new Multiplier(0, 0);
        }
        // frexp: real = q * 2^shift with 0.5 <= |q| < 1
        int shift = Math.getExponent(real) + 1;
        long fixed = (long) roundHalfAway(Math.scalb(real, -shift) * (1L << 31));
        if (fixed == 1L << 31) {
            fixed /= 2;
            shift++;
        }
        if (shift < -31) {
            return new Multiplier(0, 0);
        }
        if (shift > 30) {
            throw new IOException("Quantization scales are too far apart");
        }
        return new Multiplier((int) fixed, shift);
    }

    int apply(int x) {
        int left = Math.max(shift, 0);
        int right = Math.max(-shift, 0);
        return roundingDivideByPowerOfTwo(doublingHighMul(x * (1 << left), value), right);
    }

    /** Rounds halfway cases away from zero, like C's {@code round}. */
    static double roundHalfAway(double x) {
        double magnitude = Math.abs(x);
        double floor = Math.floor(magnitude);
        return Math.copySign(magnitude - floor >= 0.5 ? floor + 1 : floor, x);
    }

    /** Rounds halfway cases away from zero, like C's {@code roundf}. */
    static float roundHalfAway(float x) {
        return (float) roundHalfAway((double) x);
    }

    private static int doublingHighMul(int a, int b) {
        if (a == Integer.MIN_VALUE && b == Integer.MIN_VALUE) {
            return Integer.MAX_VALUE;
        }
        long product = (long) a * b;
        long nudge = product >= 0 ? 1L << 30 : 1 - (1L << 30);
        return (int) ((product + nudge) / (1L << 31));
    }

    private static int roundingDivideByPowerOfTwo(int x, int exponent) {
        int mask = (int) ((1L << exponent) - 1);
        int remainder = x & mask;
        int threshold = (mask >> 1) + (x < 0 ? 1 : 0);
        return (x >> exponent) + (remainder > threshold ? 1 : 0);
    }
}
