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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

public class FftTest {

    @ParameterizedTest
    @ValueSource(ints = {4, 8, 16, 32, 64})
    public void matchesDiscreteFourierTransform(int size) {
        Fft fft = new Fft(size);
        short[] input = new short[size];
        for (int n = 0; n < size; n++) {
            double wave = 4000 * Math.cos(2 * Math.PI * n / size);
            input[n] = (short) (wave - 2500 * (n & 1) + 500 * n);
        }
        short[] outRe = new short[size / 2 + 1];
        short[] outIm = new short[size / 2 + 1];

        fft.transform(fft.workspace(), input, outRe, outIm);

        for (int k = 0; k <= size / 2; k++) {
            double expectedRe = 0;
            double expectedIm = 0;
            for (int n = 0; n < size; n++) {
                double angle = -2 * Math.PI * k * n / size;
                expectedRe += input[n] * Math.cos(angle);
                expectedIm += input[n] * Math.sin(angle);
            }
            assertEquals(expectedRe / size, outRe[k], 4.0, "re bin " + k);
            assertEquals(expectedIm / size, outIm[k], 4.0, "im bin " + k);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {-4, 0, 2, 6, 12})
    public void rejectsInvalidSizes(int size) {
        assertThrows(IllegalArgumentException.class, () -> new Fft(size));
    }
}
