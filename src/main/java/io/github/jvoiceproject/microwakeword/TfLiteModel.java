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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * The parts of a TensorFlow Lite model file that inference needs, read from its FlatBuffer
 * encoding.
 *
 * <p>Field numbers follow the TensorFlow Lite {@code schema.fbs}.
 *
 * @author Dan Cunningham - Initial contribution
 */
record TfLiteModel(List<Subgraph> subgraphs) {

    static final int TYPE_INT32 = 2;
    static final int TYPE_UINT8 = 3;
    static final int TYPE_INT8 = 9;
    static final int TYPE_RESOURCE = 13;

    /** The most elements a tensor may have; the largest tensor of the bundled models has 6,400. */
    static final int MAX_TENSOR_SIZE = 1 << 20;

    /**
     * @param shape dimensions, outermost first
     * @param type TensorFlow Lite {@code TensorType}
     * @param scale quantization scales, one per channel or a single one, empty when not quantized
     * @param zeroPoint quantization zero points, matching {@code scale}
     * @param quantizedDimension the channel dimension of per channel quantization
     * @param data constant contents, {@code null} when computed at runtime
     */
    record Tensor(
            String name,
            int[] shape,
            int type,
            float[] scale,
            long[] zeroPoint,
            int quantizedDimension,
            byte[] data) {

        int size() {
            int size = 1;
            for (int dimension : shape) {
                size *= dimension;
            }
            return size;
        }

        boolean isConstant() {
            return data != null;
        }
    }

    /**
     * @param opcode TensorFlow Lite {@code BuiltinOperator}
     * @param inputs tensor indices, -1 for an omitted optional input
     * @param optionsType TensorFlow Lite {@code BuiltinOptions} type of {@code options}
     * @param options the operator's builtin options table, {@code null} when absent
     */
    record Operator(int opcode, int[] inputs, int[] outputs, int optionsType, Table options) {}

    record Subgraph(List<Tensor> tensors, int[] inputs, int[] outputs, List<Operator> operators) {}

    /**
     * Reads a model. Options tables are read later, so their reads can throw {@link
     * RuntimeException}s on malformed models too.
     *
     * @throws IOException when the model can't be read or uses external buffers
     */
    static TfLiteModel parse(byte[] bytes) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        if (bytes.length < 8
                || !"TFL3".equals(new String(bytes, 4, 4, StandardCharsets.US_ASCII))) {
            throw new IOException("Not a TensorFlow Lite model");
        }
        Table model = new Table(buffer, buffer.getInt(0));
        List<Table> buffers = model.tables(4);
        List<Table> codes = model.tables(1);
        List<Subgraph> subgraphs = new ArrayList<>();
        for (Table subgraph : model.tables(2)) {
            subgraphs.add(subgraph(subgraph, codes, buffers));
        }
        return new TfLiteModel(List.copyOf(subgraphs));
    }

    private static Subgraph subgraph(Table subgraph, List<Table> codes, List<Table> buffers)
            throws IOException {
        List<Tensor> tensors = new ArrayList<>();
        for (Table tensor : subgraph.tables(0)) {
            tensors.add(tensor(tensor, buffers));
        }
        List<Operator> operators = new ArrayList<>();
        for (Table operator : subgraph.tables(3)) {
            Table code = codes.get(operator.getInt(0, 0));
            // models written before builtin_code was added only set the deprecated byte field
            int opcode = Math.max(code.getByte(0, 0), code.getInt(3, 0));
            operators.add(
                    new Operator(
                            opcode,
                            operator.ints(1),
                            operator.ints(2),
                            operator.getByte(3, 0) & 0xFF,
                            operator.table(4)));
        }
        return new Subgraph(
                List.copyOf(tensors), subgraph.ints(1), subgraph.ints(2), List.copyOf(operators));
    }

    private static Tensor tensor(Table tensor, List<Table> buffers) throws IOException {
        Table quantization = tensor.table(4);
        float[] scale = quantization == null ? new float[0] : quantization.floats(2);
        long[] zeroPoint = quantization == null ? new long[0] : quantization.longs(3);
        int quantizedDimension = quantization == null ? 0 : quantization.getInt(6, 0);
        Table buffer = buffers.get(tensor.getInt(2, 0));
        if (buffer.getLong(1, 0) > 1) {
            throw new IOException("Models with external buffers are not supported");
        }
        byte[] data = buffer.bytes(0);
        String name = tensor.string(3);
        int[] shape = tensor.ints(0);
        // bounding the product of the dimensions bounds every partial product of them too
        long size = 1;
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IOException("Tensor " + name + " has a negative dimension");
            }
            size *= Math.max(dimension, 1);
            if (size > MAX_TENSOR_SIZE) {
                throw new IOException("Tensor " + name + " is too large");
            }
        }
        return new Tensor(
                name,
                shape,
                tensor.getByte(1, 0),
                scale,
                zeroPoint,
                quantizedDimension,
                data.length == 0 ? null : data);
    }

    /** A FlatBuffer table, read through its vtable. */
    static final class Table {
        private final ByteBuffer buffer;
        private final int position;

        private Table(ByteBuffer buffer, int position) {
            this.buffer = buffer;
            this.position = position;
        }

        int getInt(int field, int defaultValue) {
            int offset = offset(field);
            return offset == 0 ? defaultValue : buffer.getInt(position + offset);
        }

        int getByte(int field, int defaultValue) {
            int offset = offset(field);
            return offset == 0 ? defaultValue : buffer.get(position + offset);
        }

        long getLong(int field, long defaultValue) {
            int offset = offset(field);
            return offset == 0 ? defaultValue : buffer.getLong(position + offset);
        }

        boolean getBoolean(int field) {
            return getByte(field, 0) != 0;
        }

        Table table(int field) {
            int offset = offset(field);
            return offset == 0 ? null : new Table(buffer, indirect(position + offset));
        }

        String string(int field) {
            int start = vector(field);
            if (start < 0) {
                return "";
            }
            byte[] bytes = new byte[length(start, 1)];
            buffer.get(start + 4, bytes);
            return new String(bytes, StandardCharsets.UTF_8);
        }

        byte[] bytes(int field) {
            int start = vector(field);
            byte[] values = new byte[length(start, 1)];
            if (values.length > 0) {
                buffer.get(start + 4, values);
            }
            return values;
        }

        int[] ints(int field) {
            int start = vector(field);
            int[] values = new int[length(start, 4)];
            for (int i = 0; i < values.length; i++) {
                values[i] = buffer.getInt(start + 4 + 4 * i);
            }
            return values;
        }

        long[] longs(int field) {
            int start = vector(field);
            long[] values = new long[length(start, 8)];
            for (int i = 0; i < values.length; i++) {
                values[i] = buffer.getLong(start + 4 + 8 * i);
            }
            return values;
        }

        float[] floats(int field) {
            int start = vector(field);
            float[] values = new float[length(start, 4)];
            for (int i = 0; i < values.length; i++) {
                values[i] = buffer.getFloat(start + 4 + 4 * i);
            }
            return values;
        }

        List<Table> tables(int field) {
            int start = vector(field);
            int length = length(start, 4);
            List<Table> tables = new ArrayList<>(length);
            for (int i = 0; i < length; i++) {
                tables.add(new Table(buffer, indirect(start + 4 + 4 * i)));
            }
            return tables;
        }

        /** Position of a vector's length prefix, or -1 when the field is absent. */
        private int vector(int field) {
            int offset = offset(field);
            return offset == 0 ? -1 : indirect(position + offset);
        }

        /** Length of the vector at {@code start}, checked against the rest of the model. */
        private int length(int start, int elementSize) {
            if (start < 0) {
                return 0;
            }
            int length = buffer.getInt(start);
            if (length < 0 || length > (buffer.capacity() - start - 4) / elementSize) {
                throw new IndexOutOfBoundsException("Vector at " + start + " exceeds the model");
            }
            return length;
        }

        private int offset(int field) {
            int vtable = position - buffer.getInt(position);
            int vtableSize = Short.toUnsignedInt(buffer.getShort(vtable));
            int entry = 4 + 2 * field;
            return entry < vtableSize ? Short.toUnsignedInt(buffer.getShort(vtable + entry)) : 0;
        }

        private int indirect(int position) {
            int target = position + buffer.getInt(position);
            // keeps -1 meaning an absent vector
            if (target < 0) {
                throw new IndexOutOfBoundsException("Offset at " + position + " is out of range");
            }
            return target;
        }
    }
}
