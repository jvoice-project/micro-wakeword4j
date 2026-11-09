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
package io.github.jvoiceproject.microwakeword.internal.tflite.frontend;

/**
 * Fixed-point real FFT for power of two sizes.
 *
 * <p>A port of the 16-bit {@code kiss_fftr} that the TensorFlow Lite Micro audio frontend uses.
 * Every stage scales its values down to stay within 16 bits, and every intermediate value is
 * truncated to 16 bits exactly like the C code, so the output is bit-exact.
 *
 * <p>Derived from KISS FFT, Copyright (c) 2003-2010, Mark Borgerding, under the BSD 3-clause
 * license in {@code LICENSE-kissfft}.
 *
 * @author Dan Cunningham - Initial contribution
 */
final class Fft {
    private static final int SAMP_MAX = 32767;

    private final int complexSize;
    private final int[] factors;
    private final short[] twiddleRe;
    private final short[] twiddleIm;
    private final short[] superTwiddleRe;
    private final short[] superTwiddleIm;

    /**
     * @param size number of real input values, a power of two of at least 4
     */
    Fft(int size) {
        if (size < 4 || Integer.bitCount(size) != 1) {
            throw new IllegalArgumentException("FFT size must be a power of two");
        }
        complexSize = size / 2;
        twiddleRe = new short[complexSize];
        twiddleIm = new short[complexSize];
        for (int i = 0; i < complexSize; i++) {
            double phase = -2 * Math.PI * i / complexSize;
            twiddleRe[i] = fixed(StrictMath.cos(phase));
            twiddleIm[i] = fixed(StrictMath.sin(phase));
        }
        superTwiddleRe = new short[complexSize / 2];
        superTwiddleIm = new short[complexSize / 2];
        for (int i = 0; i < complexSize / 2; i++) {
            double phase = -Math.PI * ((double) (i + 1) / complexSize + .5);
            superTwiddleRe[i] = fixed(StrictMath.cos(phase));
            superTwiddleIm[i] = fixed(StrictMath.sin(phase));
        }
        factors = factor(complexSize);
    }

    /** Scratch buffers of one caller, so a shared {@link Fft} needs no locking. */
    final class Workspace {
        private final short[] re = new short[complexSize];
        private final short[] im = new short[complexSize];
    }

    Workspace workspace() {
        return new Workspace();
    }

    /** Transforms {@code 2 * complexSize} real values into {@code complexSize + 1} complex bins. */
    void transform(Workspace w, short[] input, short[] outRe, short[] outIm) {
        short[] re = w.re;
        short[] im = w.im;
        work(re, im, 0, input, 0, 1, 0);

        // split the packed transform of even and odd samples into the real spectrum
        short dcRe = divide(re[0], 2);
        short dcIm = divide(im[0], 2);
        outRe[0] = (short) (dcRe + dcIm);
        outRe[complexSize] = (short) (dcRe - dcIm);
        outIm[0] = 0;
        outIm[complexSize] = 0;
        for (int k = 1; k <= complexSize / 2; k++) {
            short fpkRe = divide(re[k], 2);
            short fpkIm = divide(im[k], 2);
            short fpnkRe = divide(re[complexSize - k], 2);
            short fpnkIm = divide((short) -im[complexSize - k], 2);
            short f1kRe = (short) (fpkRe + fpnkRe);
            short f1kIm = (short) (fpkIm + fpnkIm);
            short f2kRe = (short) (fpkRe - fpnkRe);
            short f2kIm = (short) (fpkIm - fpnkIm);
            short twRe = mulRe(f2kRe, f2kIm, superTwiddleRe[k - 1], superTwiddleIm[k - 1]);
            short twIm = mulIm(f2kRe, f2kIm, superTwiddleRe[k - 1], superTwiddleIm[k - 1]);
            outRe[k] = (short) ((f1kRe + twRe) >> 1);
            outIm[k] = (short) ((f1kIm + twIm) >> 1);
            outRe[complexSize - k] = (short) ((f1kRe - twRe) >> 1);
            outIm[complexSize - k] = (short) ((twIm - f1kIm) >> 1);
        }
    }

    /** One stage of the recursive mixed radix FFT, reading interleaved real and imaginary input. */
    private void work(
            short[] re, short[] im, int out, short[] input, int in, int stride, int factor) {
        int radix = factors[factor];
        int m = factors[factor + 1];
        int end = out + radix * m;
        if (m == 1) {
            for (int o = out; o < end; o++, in += stride) {
                re[o] = input[2 * in];
                im[o] = input[2 * in + 1];
            }
        } else {
            for (int o = out; o < end; o += m, in += stride) {
                work(re, im, o, input, in, stride * radix, factor + 2);
            }
        }
        if (radix == 4) {
            butterfly4(re, im, out, stride, m);
        } else {
            butterfly2(re, im, out, stride, m);
        }
    }

    private void butterfly2(short[] re, short[] im, int out, int stride, int m) {
        for (int k = 0; k < m; k++) {
            int a = out + k;
            int b = a + m;
            re[a] = divide(re[a], 2);
            im[a] = divide(im[a], 2);
            re[b] = divide(re[b], 2);
            im[b] = divide(im[b], 2);
            int tw = k * stride;
            short tRe = mulRe(re[b], im[b], twiddleRe[tw], twiddleIm[tw]);
            short tIm = mulIm(re[b], im[b], twiddleRe[tw], twiddleIm[tw]);
            re[b] = (short) (re[a] - tRe);
            im[b] = (short) (im[a] - tIm);
            re[a] = (short) (re[a] + tRe);
            im[a] = (short) (im[a] + tIm);
        }
    }

    private void butterfly4(short[] re, short[] im, int out, int stride, int m) {
        for (int k = 0; k < m; k++) {
            int f0 = out + k;
            int f1 = f0 + m;
            int f2 = f0 + 2 * m;
            int f3 = f0 + 3 * m;
            for (int f = f0; f <= f3; f += m) {
                re[f] = divide(re[f], 4);
                im[f] = divide(im[f], 4);
            }
            int tw1 = k * stride;
            int tw2 = 2 * k * stride;
            int tw3 = 3 * k * stride;
            short s0Re = mulRe(re[f1], im[f1], twiddleRe[tw1], twiddleIm[tw1]);
            short s0Im = mulIm(re[f1], im[f1], twiddleRe[tw1], twiddleIm[tw1]);
            short s1Re = mulRe(re[f2], im[f2], twiddleRe[tw2], twiddleIm[tw2]);
            short s1Im = mulIm(re[f2], im[f2], twiddleRe[tw2], twiddleIm[tw2]);
            short s2Re = mulRe(re[f3], im[f3], twiddleRe[tw3], twiddleIm[tw3]);
            short s2Im = mulIm(re[f3], im[f3], twiddleRe[tw3], twiddleIm[tw3]);
            short s5Re = (short) (re[f0] - s1Re);
            short s5Im = (short) (im[f0] - s1Im);
            re[f0] = (short) (re[f0] + s1Re);
            im[f0] = (short) (im[f0] + s1Im);
            short s3Re = (short) (s0Re + s2Re);
            short s3Im = (short) (s0Im + s2Im);
            short s4Re = (short) (s0Re - s2Re);
            short s4Im = (short) (s0Im - s2Im);
            re[f2] = (short) (re[f0] - s3Re);
            im[f2] = (short) (im[f0] - s3Im);
            re[f0] = (short) (re[f0] + s3Re);
            im[f0] = (short) (im[f0] + s3Im);
            re[f1] = (short) (s5Re + s4Im);
            im[f1] = (short) (s5Im - s4Re);
            re[f3] = (short) (s5Re - s4Im);
            im[f3] = (short) (s5Im + s4Re);
        }
    }

    /** Radix and remaining length pairs, preferring radix 4 like {@code kf_factor}. */
    private static int[] factor(int n) {
        int stages = 0;
        for (int rest = n; rest > 1; rest /= rest % 4 == 0 ? 4 : 2) {
            stages++;
        }
        int[] factors = new int[2 * stages];
        for (int i = 0; n > 1; i += 2) {
            int radix = n % 4 == 0 ? 4 : 2;
            n /= radix;
            factors[i] = radix;
            factors[i + 1] = n;
        }
        return factors;
    }

    private static short fixed(double value) {
        return (short) Math.floor(.5 + SAMP_MAX * value);
    }

    private static short round(int value) {
        return (short) ((value + (1 << 14)) >> 15);
    }

    private static short divide(short value, int divisor) {
        return round(value * (SAMP_MAX / divisor));
    }

    private static short mulRe(short aRe, short aIm, short bRe, short bIm) {
        return round(aRe * bRe - aIm * bIm);
    }

    private static short mulIm(short aRe, short aIm, short bRe, short bIm) {
        return round(aRe * bIm + aIm * bRe);
    }
}
