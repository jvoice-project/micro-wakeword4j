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

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Models and manifests made to break the loader, which must fail with an IOException. */
public class MalformedModelTest {
    private static final int CONV_2D = 3;
    private static final int CALL_ONCE = 129;

    @TempDir Path directory;

    @Test
    public void loadsTheUnchangedModel() {
        assertDoesNotThrow(() -> load(model -> {}));
    }

    @Test
    public void rejectsVectorsLongerThanTheModel() {
        assertThrows(
                IOException.class,
                () -> load(model -> model.setInt(model.inputShape(), 0x7FFFFFF0)));
    }

    @Test
    public void rejectsNegativeDimensions() {
        assertThrows(
                IOException.class, () -> load(model -> model.setInt(model.inputShape() + 8, -1)));
    }

    @Test
    public void rejectsTensorsThatAreTooLarge() {
        assertThrows(
                IOException.class,
                () -> load(model -> model.setInt(model.inputShape() + 8, 1 << 21)));
        assertThrows(
                IOException.class,
                () -> load(model -> model.setInt(model.inputShape() + 8, Integer.MAX_VALUE)));
    }

    @Test
    public void rejectsInvalidScales() {
        for (float scale : new float[] {0, -1, Float.NaN, Float.POSITIVE_INFINITY}) {
            assertThrows(
                    IOException.class,
                    () -> load(model -> model.setFloat(model.inputQuantization(2) + 4, scale)),
                    "scale " + scale);
        }
    }

    @Test
    public void rejectsZeroPointsOutsideTheType() {
        assertThrows(
                IOException.class,
                () -> load(model -> model.setLong(model.inputQuantization(3) + 4, 128)));
    }

    @Test
    public void rejectsOptionsOfAnotherOperator() {
        assertThrows(
                IOException.class,
                () ->
                        load(
                                model -> {
                                    int callOnce = model.operator(0);
                                    assertEquals(CALL_ONCE, model.opcode(callOnce));
                                    model.setByte(model.field(callOnce, 3), 1);
                                }));
    }

    @Test
    public void rejectsModelsThatNeedTooMuchMemory() {
        assertThrows(
                IOException.class,
                () -> load(model -> model.setInt(model.inputShape() + 8, 26_200)));
    }

    @Test
    public void rejectsTensorsWithMultipleWriters() {
        assertThrows(
                IOException.class,
                () ->
                        load(
                                model -> {
                                    int firstOut =
                                            model.buffer.getInt(
                                                    model.vector(model.operator(1), 2) + 4);
                                    model.setInt(model.vector(model.operator(2), 2) + 4, firstOut);
                                }));
    }

    @Test
    public void rejectsInvalidConvolutionStride() {
        for (int stride : new int[] {0, -1, 1025, Integer.MAX_VALUE}) {
            assertThrows(
                    IOException.class,
                    () ->
                            load(
                                    model -> {
                                        int conv = model.findOperator(CONV_2D);
                                        int options = model.indirect(model.field(conv, 4));
                                        model.setInt(model.field(options, 1), stride);
                                    }),
                    "stride " + stride);
        }
    }

    @Test
    public void rejectsNonZeroConvolutionFilterZeroPoints() {
        assertThrows(
                IOException.class,
                () ->
                        load(
                                model -> {
                                    int conv = model.findOperator(CONV_2D);
                                    int filterIndex =
                                            model.buffer.getInt(model.vector(conv, 1) + 8);
                                    int filter =
                                            model.element(
                                                    model.vector(model.mainGraph(), 0),
                                                    filterIndex);
                                    int zeroPoints =
                                            model.vector(model.indirect(model.field(filter, 4)), 3);
                                    model.setLong(zeroPoints + 4, 1);
                                }));
    }

    @Test
    public void rejectsDeeplyNestedManifests() {
        assertThrows(IOException.class, () -> Json.parse("[".repeat(200_000)));
        assertDoesNotThrow(() -> Json.parse("[".repeat(64) + "]".repeat(64)));
    }

    @Test
    public void rejectsSignedUnicodeEscapes() throws IOException {
        assertThrows(IOException.class, () -> Json.parse("\"\\u-123\""));
        assertThrows(IOException.class, () -> Json.parse("\"\\u+123\""));
        assertEquals("é", Json.parse("\"\\u00e9\""));
    }

    @Test
    public void parsesAndRejectsJsonCornerCases() throws IOException {
        assertEquals(Map.of(), Json.parse("{}"));
        assertEquals(List.of(true, false), Json.parse("[true, false]"));
        assertEquals(Arrays.asList(new Object[] {null}), Json.parse("[null]"));
        assertEquals("\"\\/\b\f\n\r\t", Json.parse("\"\\\"\\\\\\/\\b\\f\\n\\r\\t\""));

        for (String invalid :
                List.of(
                        "",
                        "{} []",
                        "tru",
                        "fals",
                        "nul",
                        "-",
                        "1.2.3",
                        "\"unterminated",
                        "\"\\",
                        "\"\\x\"",
                        "\"\\u12\"",
                        "\"\\u123g\"",
                        "{1: 2}",
                        "{\"a\" 1}",
                        "{\"a\": 1",
                        "[1")) {
            assertThrows(IOException.class, () -> Json.parse(invalid), invalid);
        }
    }

    @Test
    public void rejectsModelNamesThatAreNotPaths() throws IOException {
        Path manifest = directory.resolve("model.json");
        Files.writeString(manifest, manifest().replace("hey_jarvis.tflite", "a\\u0000b"));

        assertThrows(IOException.class, () -> WakeWordModel.load(manifest));
    }

    private static WakeWordModel load(Consumer<Model> change) throws IOException {
        Model model = new Model(resource("hey_jarvis.tflite"));
        change.accept(model);
        return WakeWordModel.load(manifest(), model.buffer.array());
    }

    private static String manifest() throws IOException {
        return new String(resource("hey_jarvis.json"), StandardCharsets.UTF_8);
    }

    private static byte[] resource(String name) throws IOException {
        try (InputStream in = WakeWordModel.class.getResourceAsStream(name)) {
            return in.readAllBytes();
        }
    }

    /** A model to change in place, with the positions of the FlatBuffer fields to change. */
    private static final class Model {
        final ByteBuffer buffer;

        Model(byte[] bytes) {
            buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        }

        /** The length of the shape vector of the main input, followed by its dimensions. */
        int inputShape() {
            return vector(input(), 0);
        }

        /** The length of a vector of the quantization of the main input. */
        int inputQuantization(int field) {
            return vector(indirect(field(input(), 4)), field);
        }

        int operator(int index) {
            return element(vector(mainGraph(), 3), index);
        }

        int opcode(int operator) {
            int code = element(vector(indirect(0), 1), getInt(operator, 0));
            return Math.max(getByte(code, 0), getInt(code, 3));
        }

        int findOperator(int opcode) {
            int operators = vector(mainGraph(), 3);
            int length = buffer.getInt(operators);
            for (int i = 0; i < length; i++) {
                int operator = element(operators, i);
                if (opcode(operator) == opcode) {
                    return operator;
                }
            }
            throw new IllegalStateException("No operator with opcode " + opcode);
        }

        int field(int table, int field) {
            int offset = offset(table, field);
            if (offset == 0) {
                throw new IllegalStateException("Field " + field + " is absent");
            }
            return table + offset;
        }

        /** An int field, which FlatBuffers leave out when it has the default value 0. */
        private int getInt(int table, int field) {
            int offset = offset(table, field);
            return offset == 0 ? 0 : buffer.getInt(table + offset);
        }

        private int getByte(int table, int field) {
            int offset = offset(table, field);
            return offset == 0 ? 0 : buffer.get(table + offset);
        }

        private int offset(int table, int field) {
            int vtable = table - buffer.getInt(table);
            int entry = 4 + 2 * field;
            return entry < Short.toUnsignedInt(buffer.getShort(vtable))
                    ? Short.toUnsignedInt(buffer.getShort(vtable + entry))
                    : 0;
        }

        void setByte(int position, int value) {
            buffer.put(position, (byte) value);
        }

        void setInt(int position, int value) {
            buffer.putInt(position, value);
        }

        void setLong(int position, long value) {
            buffer.putLong(position, value);
        }

        void setFloat(int position, float value) {
            buffer.putFloat(position, value);
        }

        private int mainGraph() {
            return element(vector(indirect(0), 2), 0);
        }

        private int input() {
            int index = buffer.getInt(vector(mainGraph(), 1) + 4);
            return element(vector(mainGraph(), 0), index);
        }

        private int vector(int table, int field) {
            return indirect(field(table, field));
        }

        private int element(int vector, int index) {
            return indirect(vector + 4 + 4 * index);
        }

        private int indirect(int position) {
            return position + buffer.getInt(position);
        }
    }
}
