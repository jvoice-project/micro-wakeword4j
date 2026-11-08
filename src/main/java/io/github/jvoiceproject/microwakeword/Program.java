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

import static io.github.jvoiceproject.microwakeword.TfLiteModel.TYPE_INT32;
import static io.github.jvoiceproject.microwakeword.TfLiteModel.TYPE_INT8;
import static io.github.jvoiceproject.microwakeword.TfLiteModel.TYPE_RESOURCE;
import static io.github.jvoiceproject.microwakeword.TfLiteModel.TYPE_UINT8;

import io.github.jvoiceproject.microwakeword.TfLiteModel.Operator;
import io.github.jvoiceproject.microwakeword.TfLiteModel.Subgraph;
import io.github.jvoiceproject.microwakeword.TfLiteModel.Table;
import io.github.jvoiceproject.microwakeword.TfLiteModel.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A quantized TensorFlow Lite model compiled into integer kernels.
 *
 * <p>Supports the operators of microWakeWord streaming models with int8 activations. The kernels
 * follow TensorFlow Lite's reference implementations, so outputs are bit-exact with its {@code
 * BUILTIN_REF} interpreter. All parameters are computed here once, so a program can be shared by
 * any number of {@link Interpreter}s.
 *
 * @author Dan Cunningham - Initial contribution
 */
final class Program {

    // TensorFlow Lite BuiltinOperator codes
    private static final int CONCATENATION = 2;
    private static final int CONV_2D = 3;
    private static final int DEPTHWISE_CONV_2D = 4;
    private static final int FULLY_CONNECTED = 9;
    private static final int LOGISTIC = 14;
    private static final int RESHAPE = 22;
    private static final int STRIDED_SLICE = 45;
    private static final int SPLIT_V = 102;
    private static final int QUANTIZE = 114;
    private static final int CALL_ONCE = 129;
    private static final int VAR_HANDLE = 142;
    private static final int READ_VARIABLE = 143;
    private static final int ASSIGN_VARIABLE = 144;

    // TensorFlow Lite BuiltinOptions types
    private static final int CONV_2D_OPTIONS = 1;
    private static final int DEPTHWISE_CONV_2D_OPTIONS = 2;
    private static final int FULLY_CONNECTED_OPTIONS = 8;
    private static final int CONCATENATION_OPTIONS = 10;
    private static final int STRIDED_SLICE_OPTIONS = 32;
    private static final int CALL_ONCE_OPTIONS = 103;
    private static final int VAR_HANDLE_OPTIONS = 111;

    /** The most bytes of tensors and variables per stream; the bundled models use under 30 KB. */
    private static final long MAX_STATE_SIZE = 1 << 20;

    // TensorFlow Lite ActivationFunctionType and Padding values
    private static final int ACTIVATION_NONE = 0;
    private static final int ACTIVATION_RELU = 1;
    private static final int ACTIVATION_RELU_N1_TO_1 = 2;
    private static final int ACTIVATION_RELU6 = 3;
    private static final int PADDING_SAME = 0;
    private static final int PADDING_VALID = 1;

    /** A compiled operator, running on the tensors of one subgraph. */
    interface Op {
        void run(Interpreter interpreter, byte[][] tensors);
    }

    final List<List<Op>> operators;
    final List<byte[][]> constants;
    final List<int[]> bufferSizes;
    final int[] variableSizes;
    final int inputTensor;
    final int outputTensor;
    final int[] inputShape;
    final float inputScale;
    final int inputZeroPoint;
    final int outputZeroPoint;
    final boolean unsignedOutput;

    private Program(Compiler compiler, Tensor input, Tensor output, Subgraph main) {
        this.operators = List.copyOf(compiler.operators);
        this.constants = List.copyOf(compiler.constants);
        this.bufferSizes = List.copyOf(compiler.bufferSizes);
        this.variableSizes = compiler.variableSizes.stream().mapToInt(Integer::intValue).toArray();
        this.inputTensor = main.inputs()[0];
        this.outputTensor = main.outputs()[0];
        this.inputShape = input.shape().clone();
        this.inputScale = input.scale()[0];
        this.inputZeroPoint = (int) input.zeroPoint()[0];
        this.outputZeroPoint = (int) output.zeroPoint()[0];
        this.unsignedOutput = output.type() == TYPE_UINT8;
    }

    /**
     * Compiles a model with one int8 input and a single int8 or uint8 output value.
     *
     * @throws IOException when the model is invalid or uses unsupported features
     */
    static Program compile(TfLiteModel model) throws IOException {
        if (model.subgraphs().isEmpty()) {
            throw new IOException("Model has no subgraphs");
        }
        Subgraph main = model.subgraphs().get(0);
        if (main.inputs().length != 1 || main.outputs().length != 1) {
            throw new IOException("Model must have exactly one input and one output");
        }
        Tensor input = tensor(main, main.inputs()[0]);
        Tensor output = tensor(main, main.outputs()[0]);
        requireQuantized(input, TYPE_INT8);
        requireQuantized(output, TYPE_INT8, TYPE_UINT8);
        // a constant input would be shared, and written, by every interpreter
        if (input.isConstant()) {
            throw new IOException("Model input must not be constant");
        }
        if (output.size() != 1) {
            throw new IOException("Model output must be a single value");
        }
        Compiler compiler = new Compiler(model);
        for (int i = 0; i < model.subgraphs().size(); i++) {
            compiler.compileSubgraph(i);
        }
        for (int size : compiler.variableSizes) {
            compiler.reserve(Math.max(size, 0));
        }
        return new Program(compiler, input, output, main);
    }

    private static final class Compiler {
        private final TfLiteModel model;
        private final List<List<Op>> operators = new ArrayList<>();
        private final List<byte[][]> constants = new ArrayList<>();
        private final List<int[]> bufferSizes = new ArrayList<>();
        private final Map<List<String>, Integer> variableIds = new HashMap<>();
        private final List<Integer> variableSizes = new ArrayList<>();
        private final List<Integer> variableTypes = new ArrayList<>();
        private long stateSize;

        private Compiler(TfLiteModel model) {
            this.model = model;
        }

        private void compileSubgraph(int index) throws IOException {
            Subgraph graph = model.subgraphs().get(index);
            int count = graph.tensors().size();
            byte[][] data = new byte[count][];
            int[] sizes = new int[count];
            for (int i = 0; i < count; i++) {
                Tensor tensor = graph.tensors().get(i);
                int width = width(tensor);
                if (tensor.isConstant()
                        && width > 0
                        && tensor.data().length != tensor.size() * width) {
                    throw new IOException("Constant " + tensor.name() + " has the wrong size");
                }
                data[i] = tensor.data();
                boolean activation =
                        !tensor.isConstant()
                                && (tensor.type() == TYPE_INT8 || tensor.type() == TYPE_UINT8);
                sizes[i] = activation ? tensor.size() : -1;
                reserve(Math.max(sizes[i], 0));
            }
            // with a single writer per tensor, the copies that operators compile into are
            // bounded by the size of the tensors
            boolean[] written = new boolean[count];
            for (Operator operator : graph.operators()) {
                for (int output : operator.outputs()) {
                    if (output >= 0 && output < count) {
                        if (written[output]) {
                            throw new IOException("Tensor " + output + " has several writers");
                        }
                        written[output] = true;
                    }
                }
            }
            Map<Integer, Integer> resources = new HashMap<>();
            List<Op> ops = new ArrayList<>();
            for (Operator operator : graph.operators()) {
                Op op = compileOperator(index, graph, operator, resources);
                if (op != null) {
                    ops.add(op);
                }
            }
            operators.add(List.copyOf(ops));
            constants.add(data);
            bufferSizes.add(sizes);
        }

        private Op compileOperator(
                int subgraph, Subgraph graph, Operator operator, Map<Integer, Integer> resources)
                throws IOException {
            return switch (operator.opcode()) {
                case CONV_2D -> convolution(graph, operator);
                case DEPTHWISE_CONV_2D -> depthwiseConvolution(graph, operator);
                case FULLY_CONNECTED -> fullyConnected(graph, operator);
                case LOGISTIC -> logistic(graph, operator);
                case QUANTIZE -> requantize(graph, operator);
                case CONCATENATION -> concatenation(graph, operator);
                case STRIDED_SLICE -> stridedSlice(graph, operator);
                case SPLIT_V -> splitV(graph, operator);
                case RESHAPE -> reshape(graph, operator);
                case VAR_HANDLE -> {
                    resources.put(output(graph, operator, 0), variable(operator));
                    yield null;
                }
                case READ_VARIABLE -> readVariable(graph, operator, resources);
                case ASSIGN_VARIABLE -> assignVariable(graph, operator, resources);
                case CALL_ONCE -> callOnce(subgraph, operator);
                default -> throw new IOException("Unsupported operator code " + operator.opcode());
            };
        }

        private Op convolution(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor filter = inputTensor(graph, operator, 1);
            Tensor output = outputTensor(graph, operator, 0);
            requireQuantized(input, TYPE_INT8);
            requireQuantized(output, TYPE_INT8);
            requireConstant(filter, TYPE_INT8, 4);
            Table options = options(operator, CONV_2D_OPTIONS);
            Geometry geometry =
                    Geometry.of(
                            input,
                            filter,
                            output,
                            options.getByte(0, PADDING_SAME),
                            options.getInt(2, 0),
                            options.getInt(1, 0),
                            options.getInt(5, 1),
                            options.getInt(4, 1));
            int filterDepth = filter.shape()[3];
            if (filter.shape()[0] != geometry.outDepth()
                    || filterDepth == 0
                    || geometry.inDepth() % filterDepth != 0
                    || geometry.outDepth() % (geometry.inDepth() / filterDepth) != 0) {
                throw new IOException("Convolution filter does not match its input and output");
            }
            return new Convolution(
                    operator.inputs()[0],
                    operator.inputs()[1],
                    operator.outputs()[0],
                    geometry,
                    filterDepth,
                    bias(graph, operator, geometry.outDepth()),
                    Requantization.of(
                            input, filter, output, options.getByte(3, 0), geometry.outDepth()));
        }

        private Op depthwiseConvolution(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor filter = inputTensor(graph, operator, 1);
            Tensor output = outputTensor(graph, operator, 0);
            requireQuantized(input, TYPE_INT8);
            requireQuantized(output, TYPE_INT8);
            requireConstant(filter, TYPE_INT8, 4);
            Table options = options(operator, DEPTHWISE_CONV_2D_OPTIONS);
            Geometry geometry =
                    Geometry.of(
                            input,
                            filter,
                            output,
                            options.getByte(0, PADDING_SAME),
                            options.getInt(2, 0),
                            options.getInt(1, 0),
                            options.getInt(6, 1),
                            options.getInt(5, 1));
            if (filter.shape()[0] != 1
                    || filter.shape()[3] != geometry.outDepth()
                    || geometry.outDepth() % geometry.inDepth() != 0) {
                throw new IOException(
                        "Depthwise convolution filter does not match its input and output");
            }
            return new DepthwiseConvolution(
                    operator.inputs()[0],
                    operator.inputs()[1],
                    operator.outputs()[0],
                    geometry,
                    bias(graph, operator, geometry.outDepth()),
                    Requantization.of(
                            input, filter, output, options.getByte(4, 0), geometry.outDepth()));
        }

        private Op fullyConnected(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor filter = inputTensor(graph, operator, 1);
            Tensor output = outputTensor(graph, operator, 0);
            requireQuantized(input, TYPE_INT8);
            requireQuantized(output, TYPE_INT8);
            requireConstant(filter, TYPE_INT8, 2);
            Table options = options(operator, FULLY_CONNECTED_OPTIONS);
            if (options.getByte(1, 0) != 0) {
                throw new IOException(
                        "Only the default fully connected weights format is supported");
            }
            int depth = filter.shape()[0];
            int accumulation = filter.shape()[1];
            if (accumulation == 0
                    || input.size() % accumulation != 0
                    || output.size() != input.size() / accumulation * depth) {
                throw new IOException("Fully connected filter does not match its input and output");
            }
            boolean perChannel = filter.scale().length > 1;
            int filterOffset = perChannel ? 0 : -(int) filter.zeroPoint()[0];
            Requantization requantization =
                    perChannel
                            ? Requantization.of(input, filter, output, options.getByte(0, 0), depth)
                            : Requantization.ofProduct(
                                    input, filter, output, options.getByte(0, 0));
            return new FullyConnected(
                    operator.inputs()[0],
                    operator.inputs()[1],
                    operator.outputs()[0],
                    input.size() / accumulation,
                    depth,
                    accumulation,
                    filterOffset,
                    bias(graph, operator, depth),
                    requantization);
        }

        private Op logistic(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor output = outputTensor(graph, operator, 0);
            requireQuantized(input, TYPE_INT8);
            requireQuantized(output, TYPE_INT8);
            if (output.scale()[0] != 1f / 256) {
                throw new IOException("Logistic output must have a scale of 1/256");
            }
            if (input.size() != output.size()) {
                throw new IOException("Logistic input and output sizes differ");
            }
            float inputScale = input.scale()[0];
            int inputZeroPoint = (int) input.zeroPoint()[0];
            float inverseOutputScale = 1 / output.scale()[0];
            int outputZeroPoint = (int) output.zeroPoint()[0];
            byte[] table = new byte[256];
            for (int value = -128; value <= 127; value++) {
                float dequantized = inputScale * (value - inputZeroPoint);
                float transformed = 1f / (1f + (float) Math.exp(-dequantized));
                float rescaled = Multiplier.roundHalfAway(transformed * inverseOutputScale);
                int quantized = (int) (rescaled + outputZeroPoint);
                table[value & 0xFF] = (byte) Math.clamp(quantized, -128, 127);
            }
            return new Lookup(operator.inputs()[0], operator.outputs()[0], table);
        }

        private Op requantize(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor output = outputTensor(graph, operator, 0);
            requireQuantized(input, TYPE_INT8, TYPE_UINT8);
            requireQuantized(output, TYPE_INT8, TYPE_UINT8);
            if (input.size() != output.size()) {
                throw new IOException("Quantize input and output sizes differ");
            }
            boolean unsigned = output.type() == TYPE_UINT8;
            return new Requantize(
                    operator.inputs()[0],
                    operator.outputs()[0],
                    input.type() == TYPE_UINT8,
                    (int) input.zeroPoint()[0],
                    (int) output.zeroPoint()[0],
                    unsigned ? 0 : -128,
                    unsigned ? 255 : 127,
                    Multiplier.of((double) input.scale()[0] / (double) output.scale()[0]));
        }

        private Op concatenation(Subgraph graph, Operator operator) throws IOException {
            Tensor output = outputTensor(graph, operator, 0);
            Table options = options(operator, CONCATENATION_OPTIONS);
            if (options.getByte(1, 0) != ACTIVATION_NONE) {
                throw new IOException("Concatenation with an activation is not supported");
            }
            int rank = output.shape().length;
            int axis = options.getInt(0, 0);
            axis = axis < 0 ? axis + rank : axis;
            if (axis < 0 || axis >= rank) {
                throw new IOException("Invalid concatenation axis");
            }
            int outer = product(output.shape(), 0, axis);
            int outputChunk = product(output.shape(), axis, rank);
            int[] chunks = new int[operator.inputs().length];
            long total = 0;
            for (int i = 0; i < chunks.length; i++) {
                Tensor input = inputTensor(graph, operator, i);
                requireSameQuantization(input, output);
                for (int d = 0; d < rank; d++) {
                    if (input.shape().length != rank
                            || (d != axis && input.shape()[d] != output.shape()[d])) {
                        throw new IOException("Concatenation input shapes do not match");
                    }
                }
                chunks[i] = product(input.shape(), axis, rank);
                total += chunks[i];
            }
            // checked before the copies are listed, which could otherwise grow without bound
            if (total != outputChunk) {
                throw new IOException("Concatenation output shape does not match its inputs");
            }
            Copy.Builder copy = new Copy.Builder();
            int offset = 0;
            for (int i = 0; i < chunks.length; i++) {
                if (chunks[i] == 0) {
                    continue;
                }
                for (int o = 0; o < outer; o++) {
                    copy.add(
                            operator.inputs()[i],
                            o * chunks[i],
                            operator.outputs()[0],
                            o * outputChunk + offset,
                            chunks[i]);
                }
                offset += chunks[i];
            }
            return copy.build();
        }

        private Op stridedSlice(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor output = outputTensor(graph, operator, 0);
            requireSameQuantization(input, output);
            int[] begin = constantInts(inputTensor(graph, operator, 1));
            int[] end = constantInts(inputTensor(graph, operator, 2));
            int[] strides = constantInts(inputTensor(graph, operator, 3));
            Table options = options(operator, STRIDED_SLICE_OPTIONS);
            int beginMask = options.getInt(0, 0);
            int endMask = options.getInt(1, 0);
            int shrinkMask = options.getInt(4, 0);
            boolean offset = options.getBoolean(5);
            if (options.getInt(2, 0) != 0 || options.getInt(3, 0) != 0) {
                throw new IOException(
                        "Strided slice ellipsis and new axis masks are not supported");
            }
            int[] shape = input.shape();
            int rank = shape.length;
            if (rank == 0
                    || begin.length != end.length
                    || begin.length != strides.length
                    || begin.length > rank) {
                throw new IOException("Invalid strided slice indices");
            }
            int[] start = new int[rank];
            int[] stop = new int[rank];
            int[] step = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                boolean given = axis < begin.length;
                step[axis] = given ? strides[axis] : 1;
                if (step[axis] == 0) {
                    throw new IOException("Strided slice stride must not be zero");
                }
                boolean fromStart = !given || (beginMask & 1 << axis) != 0;
                boolean toEnd = !given || (endMask & 1 << axis) != 0;
                start[axis] =
                        sliceStart(shape[axis], given ? begin[axis] : 0, step[axis], fromStart);
                if (given && (shrinkMask & 1 << axis) != 0) {
                    stop[axis] = start[axis] >= shape[axis] ? start[axis] : start[axis] + 1;
                } else {
                    int last = given ? end[axis] + (offset ? start[axis] : 0) : 0;
                    stop[axis] = sliceStop(shape[axis], last, step[axis], toEnd);
                }
            }
            long count = 1;
            for (int axis = 0; axis < rank; axis++) {
                long distance =
                        step[axis] > 0
                                ? (long) stop[axis] - start[axis]
                                : (long) start[axis] - stop[axis];
                long stride = Math.abs((long) step[axis]);
                count *= Math.max(0, (distance + stride - 1) / stride);
            }
            if (count != output.size()) {
                throw new IOException("Strided slice output shape does not match its indices");
            }
            Copy.Builder copy = new Copy.Builder();
            int[] index = start.clone();
            int written = 0;
            while (inRange(index, stop, step, rank)) {
                int source = 0;
                for (int axis = 0; axis < rank; axis++) {
                    source = source * shape[axis] + index[axis];
                }
                copy.add(operator.inputs()[0], source, operator.outputs()[0], written++, 1);
                for (int axis = rank - 1; axis >= 0; axis--) {
                    index[axis] += step[axis];
                    if (axis == 0 || inRange(index[axis], stop[axis], step[axis])) {
                        break;
                    }
                    index[axis] = start[axis];
                }
            }
            return copy.build();
        }

        private Op splitV(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            int[] splits = constantInts(inputTensor(graph, operator, 1));
            int[] axisValue = constantInts(inputTensor(graph, operator, 2));
            int rank = input.shape().length;
            if (axisValue.length != 1 || splits.length != operator.outputs().length) {
                throw new IOException("Invalid split");
            }
            int axis = axisValue[0] < 0 ? axisValue[0] + rank : axisValue[0];
            if (axis < 0 || axis >= rank) {
                throw new IOException("Invalid split axis");
            }
            int inferred = -1;
            long total = 0;
            for (int i = 0; i < splits.length; i++) {
                if (splits[i] == -1 && inferred < 0) {
                    inferred = i;
                } else if (splits[i] < 0) {
                    throw new IOException("Invalid split sizes");
                } else {
                    total += splits[i];
                }
            }
            if (inferred >= 0 && total <= input.shape()[axis]) {
                splits[inferred] = (int) (input.shape()[axis] - total);
            } else if (inferred >= 0) {
                throw new IOException("Split sizes do not match the input");
            } else if (total != input.shape()[axis]) {
                throw new IOException("Split sizes do not match the input");
            }
            int outer = product(input.shape(), 0, axis);
            int inner = product(input.shape(), axis + 1, rank);
            int inputChunk = input.shape()[axis] * inner;
            Copy.Builder copy = new Copy.Builder();
            int offset = 0;
            for (int i = 0; i < splits.length; i++) {
                Tensor output = outputTensor(graph, operator, i);
                requireSameQuantization(input, output);
                int chunk = splits[i] * inner;
                if (output.size() != outer * chunk) {
                    throw new IOException("Split output shape does not match its size");
                }
                for (int o = 0; chunk > 0 && o < outer; o++) {
                    copy.add(
                            operator.inputs()[0],
                            o * inputChunk + offset,
                            operator.outputs()[i],
                            o * chunk,
                            chunk);
                }
                offset += chunk;
            }
            return copy.build();
        }

        private Op reshape(Subgraph graph, Operator operator) throws IOException {
            Tensor input = inputTensor(graph, operator, 0);
            Tensor output = outputTensor(graph, operator, 0);
            requireSameQuantization(input, output);
            if (input.size() != output.size()) {
                throw new IOException("Reshape input and output sizes differ");
            }
            return new Copy.Builder()
                    .add(operator.inputs()[0], 0, operator.outputs()[0], 0, input.size())
                    .build();
        }

        private int variable(Operator operator) throws IOException {
            Table options = operator.options();
            if (options != null && operator.optionsType() != VAR_HANDLE_OPTIONS) {
                throw new IOException("Variable handle has the wrong options");
            }
            List<String> name =
                    options == null
                            ? List.of("", "")
                            : List.of(options.string(0), options.string(1));
            return variableIds.computeIfAbsent(
                    name,
                    key -> {
                        variableSizes.add(-1);
                        variableTypes.add(-1);
                        return variableSizes.size() - 1;
                    });
        }

        private Op readVariable(Subgraph graph, Operator operator, Map<Integer, Integer> resources)
                throws IOException {
            int variable = resource(graph, operator, resources);
            Tensor output = outputTensor(graph, operator, 0);
            requireQuantized(output, TYPE_INT8, TYPE_UINT8);
            typeVariable(variable, output);
            return new ReadVariable(variable, operator.outputs()[0]);
        }

        private Op assignVariable(
                Subgraph graph, Operator operator, Map<Integer, Integer> resources)
                throws IOException {
            int variable = resource(graph, operator, resources);
            Tensor value = inputTensor(graph, operator, 1);
            requireQuantized(value, TYPE_INT8, TYPE_UINT8);
            typeVariable(variable, value);
            return new AssignVariable(variable, operator.inputs()[1]);
        }

        private Op callOnce(int subgraph, Operator operator) throws IOException {
            // initialization subgraphs that call further ones could nest deeper than the stack
            if (subgraph != 0) {
                throw new IOException("Only the main subgraph may call once");
            }
            int init = options(operator, CALL_ONCE_OPTIONS).getInt(0, -1);
            if (init <= 0 || init >= model.subgraphs().size()) {
                throw new IOException("Invalid call once subgraph " + init);
            }
            return new CallOnce(init);
        }

        private int resource(Subgraph graph, Operator operator, Map<Integer, Integer> resources)
                throws IOException {
            Tensor handle = inputTensor(graph, operator, 0);
            Integer variable = resources.get(operator.inputs()[0]);
            if (handle.type() != TYPE_RESOURCE || variable == null) {
                throw new IOException("Variable used before its handle was created");
            }
            return variable;
        }

        private void typeVariable(int variable, Tensor tensor) throws IOException {
            int size = variableSizes.get(variable);
            int type = variableTypes.get(variable);
            if ((size >= 0 && size != tensor.size()) || (type >= 0 && type != tensor.type())) {
                throw new IOException("Variable is used with different sizes or types");
            }
            variableSizes.set(variable, tensor.size());
            variableTypes.set(variable, tensor.type());
        }

        /** Counts memory that each interpreter of the program allocates. */
        private void reserve(long bytes) throws IOException {
            stateSize += bytes;
            if (stateSize > MAX_STATE_SIZE) {
                throw new IOException("Model needs too much memory");
            }
        }

        private int[] bias(Subgraph graph, Operator operator, int depth) throws IOException {
            if (operator.inputs().length < 3 || operator.inputs()[2] < 0) {
                return null;
            }
            int[] bias = constantInts(inputTensor(graph, operator, 2));
            if (bias.length != depth) {
                throw new IOException("Bias does not match the output depth");
            }
            return bias;
        }

        private static Table options(Operator operator, int type) throws IOException {
            if (operator.options() == null || operator.optionsType() != type) {
                throw new IOException(
                        "Operator " + operator.opcode() + " is missing its options or has others");
            }
            return operator.options();
        }
    }

    /** Spatial parameters of a convolution over NHWC tensors. */
    private record Geometry(
            int batches,
            int inHeight,
            int inWidth,
            int inDepth,
            int outHeight,
            int outWidth,
            int outDepth,
            int filterHeight,
            int filterWidth,
            int strideHeight,
            int strideWidth,
            int dilationHeight,
            int dilationWidth,
            int padHeight,
            int padWidth) {

        static Geometry of(
                Tensor input,
                Tensor filter,
                Tensor output,
                int padding,
                int strideHeight,
                int strideWidth,
                int dilationHeight,
                int dilationWidth)
                throws IOException {
            int[] in = input.shape();
            int[] out = output.shape();
            int[] kernel = filter.shape();
            if (in.length != 4 || out.length != 4 || in[0] != out[0]) {
                throw new IOException("Convolution tensors must have NHWC shapes");
            }
            if (in[3] == 0) {
                throw new IOException("Convolution input has no channels");
            }
            if (strideHeight <= 0
                    || strideWidth <= 0
                    || dilationHeight <= 0
                    || dilationWidth <= 0) {
                throw new IOException("Invalid convolution stride or dilation");
            }
            if (padding != PADDING_SAME && padding != PADDING_VALID) {
                throw new IOException("Unsupported convolution padding");
            }
            if (outputSize(padding, in[1], kernel[1], strideHeight, dilationHeight) != out[1]
                    || outputSize(padding, in[2], kernel[2], strideWidth, dilationWidth)
                            != out[2]) {
                throw new IOException("Convolution output shape does not match its parameters");
            }
            return new Geometry(
                    in[0],
                    in[1],
                    in[2],
                    in[3],
                    out[1],
                    out[2],
                    out[3],
                    kernel[1],
                    kernel[2],
                    strideHeight,
                    strideWidth,
                    dilationHeight,
                    dilationWidth,
                    padding(strideHeight, dilationHeight, in[1], kernel[1], out[1]),
                    padding(strideWidth, dilationWidth, in[2], kernel[2], out[2]));
        }

        private static int outputSize(
                int padding, int size, int filterSize, int stride, int dilation) {
            int effectiveFilter = (filterSize - 1) * dilation + 1;
            return padding == PADDING_SAME
                    ? (size + stride - 1) / stride
                    : (size + stride - effectiveFilter) / stride;
        }

        private static int padding(
                int stride, int dilation, int size, int filterSize, int outputSize) {
            int effectiveFilter = (filterSize - 1) * dilation + 1;
            return Math.max((outputSize - 1) * stride + effectiveFilter - size, 0) / 2;
        }
    }

    /**
     * Rescales int32 accumulators to the output quantization and applies the fused activation.
     *
     * @param multipliers one per output channel, or a single one for all channels
     */
    private record Requantization(
            int inputOffset, int outputOffset, int min, int max, Multiplier[] multipliers) {

        /** Per channel scales, as TensorFlow Lite computes them for convolutions. */
        static Requantization of(
                Tensor input, Tensor filter, Tensor output, int activation, int channels)
                throws IOException {
            float[] filterScales = filter.scale();
            if (filterScales.length != 1 && filterScales.length != channels) {
                throw new IOException(
                        "Filter " + filter.name() + " has the wrong number of scales");
            }
            Multiplier[] multipliers = new Multiplier[filterScales.length];
            for (int i = 0; i < filterScales.length; i++) {
                double scale =
                        (double) input.scale()[0]
                                * (double) filterScales[i]
                                / (double) output.scale()[0];
                multipliers[i] = Multiplier.of(scale);
            }
            return of(input, output, activation, multipliers);
        }

        /** A single scale from the float product of the input and filter scales. */
        static Requantization ofProduct(Tensor input, Tensor filter, Tensor output, int activation)
                throws IOException {
            double scale = (double) (input.scale()[0] * filter.scale()[0]) / output.scale()[0];
            return of(input, output, activation, new Multiplier[] {Multiplier.of(scale)});
        }

        private static Requantization of(
                Tensor input, Tensor output, int activation, Multiplier[] multipliers)
                throws IOException {
            float scale = output.scale()[0];
            int zeroPoint = (int) output.zeroPoint()[0];
            int min = -128;
            int max = 127;
            switch (activation) {
                case ACTIVATION_NONE -> {}
                case ACTIVATION_RELU -> min = Math.max(min, quantize(scale, zeroPoint, 0));
                case ACTIVATION_RELU6 -> {
                    min = Math.max(min, quantize(scale, zeroPoint, 0));
                    max = Math.min(max, quantize(scale, zeroPoint, 6));
                }
                case ACTIVATION_RELU_N1_TO_1 -> {
                    min = Math.max(min, quantize(scale, zeroPoint, -1));
                    max = Math.min(max, quantize(scale, zeroPoint, 1));
                }
                default -> throw new IOException("Unsupported activation " + activation);
            }
            return new Requantization(
                    -(int) input.zeroPoint()[0], zeroPoint, min, max, multipliers);
        }

        /** Clamped to the int8 range, where tiny scales can't overflow it. */
        private static int quantize(float scale, int zeroPoint, float value) {
            double quantized = zeroPoint + (double) Multiplier.roundHalfAway(value / scale);
            return (int) Math.clamp(quantized, -128, 127);
        }

        Multiplier multiplier(int channel) {
            return multipliers[multipliers.length == 1 ? 0 : channel];
        }

        int apply(int accumulator, int channel) {
            int value = multiplier(channel).apply(accumulator) + outputOffset;
            return Math.clamp(value, min, max);
        }
    }

    private record Convolution(
            int input,
            int filter,
            int output,
            Geometry g,
            int filterDepth,
            int[] bias,
            Requantization q)
            implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] in = tensors[input];
            byte[] weights = tensors[filter];
            byte[] out = tensors[output];
            int filtersPerGroup = g.outDepth() / (g.inDepth() / filterDepth);
            int inputOffset = q.inputOffset();
            for (int b = 0; b < g.batches(); b++) {
                for (int outY = 0; outY < g.outHeight(); outY++) {
                    int inYOrigin = outY * g.strideHeight() - g.padHeight();
                    for (int outX = 0; outX < g.outWidth(); outX++) {
                        int inXOrigin = outX * g.strideWidth() - g.padWidth();
                        for (int channel = 0; channel < g.outDepth(); channel++) {
                            int group = channel / filtersPerGroup;
                            int acc = 0;
                            for (int fy = 0; fy < g.filterHeight(); fy++) {
                                int inY = inYOrigin + g.dilationHeight() * fy;
                                if (inY < 0 || inY >= g.inHeight()) {
                                    continue;
                                }
                                for (int fx = 0; fx < g.filterWidth(); fx++) {
                                    int inX = inXOrigin + g.dilationWidth() * fx;
                                    if (inX < 0 || inX >= g.inWidth()) {
                                        continue;
                                    }
                                    int inBase =
                                            ((b * g.inHeight() + inY) * g.inWidth() + inX)
                                                            * g.inDepth()
                                                    + group * filterDepth;
                                    int weightBase =
                                            ((channel * g.filterHeight() + fy) * g.filterWidth()
                                                            + fx)
                                                    * filterDepth;
                                    for (int c = 0; c < filterDepth; c++) {
                                        acc +=
                                                weights[weightBase + c]
                                                        * (in[inBase + c] + inputOffset);
                                    }
                                }
                            }
                            if (bias != null) {
                                acc += bias[channel];
                            }
                            out[
                                            ((b * g.outHeight() + outY) * g.outWidth() + outX)
                                                            * g.outDepth()
                                                    + channel] =
                                    (byte) q.apply(acc, channel);
                        }
                    }
                }
            }
        }
    }

    private record DepthwiseConvolution(
            int input, int filter, int output, Geometry g, int[] bias, Requantization q)
            implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] in = tensors[input];
            byte[] weights = tensors[filter];
            byte[] out = tensors[output];
            int multiplier = g.outDepth() / g.inDepth();
            int inputOffset = q.inputOffset();
            for (int b = 0; b < g.batches(); b++) {
                for (int outY = 0; outY < g.outHeight(); outY++) {
                    int inYOrigin = outY * g.strideHeight() - g.padHeight();
                    for (int outX = 0; outX < g.outWidth(); outX++) {
                        int inXOrigin = outX * g.strideWidth() - g.padWidth();
                        for (int inChannel = 0; inChannel < g.inDepth(); inChannel++) {
                            for (int m = 0; m < multiplier; m++) {
                                int channel = m + inChannel * multiplier;
                                int acc = 0;
                                for (int fy = 0; fy < g.filterHeight(); fy++) {
                                    int inY = inYOrigin + g.dilationHeight() * fy;
                                    if (inY < 0 || inY >= g.inHeight()) {
                                        continue;
                                    }
                                    for (int fx = 0; fx < g.filterWidth(); fx++) {
                                        int inX = inXOrigin + g.dilationWidth() * fx;
                                        if (inX < 0 || inX >= g.inWidth()) {
                                            continue;
                                        }
                                        int value =
                                                in[
                                                        ((b * g.inHeight() + inY) * g.inWidth()
                                                                                + inX)
                                                                        * g.inDepth()
                                                                + inChannel];
                                        int weight =
                                                weights[
                                                        (fy * g.filterWidth() + fx) * g.outDepth()
                                                                + channel];
                                        acc += weight * (value + inputOffset);
                                    }
                                }
                                if (bias != null) {
                                    acc += bias[channel];
                                }
                                out[
                                                ((b * g.outHeight() + outY) * g.outWidth() + outX)
                                                                * g.outDepth()
                                                        + channel] =
                                        (byte) q.apply(acc, channel);
                            }
                        }
                    }
                }
            }
        }
    }

    private record FullyConnected(
            int input,
            int filter,
            int output,
            int batches,
            int depth,
            int accumulation,
            int filterOffset,
            int[] bias,
            Requantization q)
            implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] in = tensors[input];
            byte[] weights = tensors[filter];
            byte[] out = tensors[output];
            int inputOffset = q.inputOffset();
            for (int b = 0; b < batches; b++) {
                for (int channel = 0; channel < depth; channel++) {
                    int acc = 0;
                    for (int d = 0; d < accumulation; d++) {
                        acc +=
                                (weights[channel * accumulation + d] + filterOffset)
                                        * (in[b * accumulation + d] + inputOffset);
                    }
                    if (bias != null) {
                        acc += bias[channel];
                    }
                    out[channel + depth * b] = (byte) q.apply(acc, channel);
                }
            }
        }
    }

    /** Maps every int8 value through a precomputed table. */
    private record Lookup(int input, int output, byte[] table) implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] in = tensors[input];
            byte[] out = tensors[output];
            for (int i = 0; i < in.length; i++) {
                out[i] = table[in[i] & 0xFF];
            }
        }
    }

    private record Requantize(
            int input,
            int output,
            boolean unsignedInput,
            int inputZeroPoint,
            int outputZeroPoint,
            int min,
            int max,
            Multiplier multiplier)
            implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] in = tensors[input];
            byte[] out = tensors[output];
            for (int i = 0; i < in.length; i++) {
                int value = unsignedInput ? in[i] & 0xFF : in[i];
                int scaled = multiplier.apply(value - inputZeroPoint) + outputZeroPoint;
                out[i] = (byte) Math.clamp(scaled, min, max);
            }
        }
    }

    /** Moves bytes between tensors, for the operators that only rearrange data. */
    private record Copy(int[] from, int[] fromOffset, int[] to, int[] toOffset, int[] length)
            implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            for (int i = 0; i < from.length; i++) {
                System.arraycopy(
                        tensors[from[i]], fromOffset[i], tensors[to[i]], toOffset[i], length[i]);
            }
        }

        /** Collects copies, merging those that continue the previous one. */
        static final class Builder {
            private final List<int[]> runs = new ArrayList<>();

            Builder add(int from, int fromOffset, int to, int toOffset, int length) {
                if (length == 0) {
                    return this;
                }
                int[] last = runs.isEmpty() ? null : runs.get(runs.size() - 1);
                if (last != null
                        && last[0] == from
                        && last[2] == to
                        && last[1] + last[4] == fromOffset
                        && last[3] + last[4] == toOffset) {
                    last[4] += length;
                } else {
                    runs.add(new int[] {from, fromOffset, to, toOffset, length});
                }
                return this;
            }

            Copy build() {
                int[][] columns = new int[5][runs.size()];
                for (int i = 0; i < runs.size(); i++) {
                    for (int c = 0; c < 5; c++) {
                        columns[c][i] = runs.get(i)[c];
                    }
                }
                return new Copy(columns[0], columns[1], columns[2], columns[3], columns[4]);
            }
        }
    }

    private record ReadVariable(int variable, int output) implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] value = interpreter.variable(variable);
            System.arraycopy(value, 0, tensors[output], 0, value.length);
        }
    }

    private record AssignVariable(int variable, int input) implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            byte[] value = interpreter.variable(variable);
            System.arraycopy(tensors[input], 0, value, 0, value.length);
        }
    }

    private record CallOnce(int subgraph) implements Op {

        @Override
        public void run(Interpreter interpreter, byte[][] tensors) {
            interpreter.callOnce(subgraph);
        }
    }

    private static int sliceStart(int size, int begin, int stride, boolean fromStart) {
        if (fromStart) {
            return stride > 0 ? 0 : size - 1;
        }
        int start = begin < 0 ? begin + size : begin;
        return stride > 0 ? Math.clamp(start, 0, size) : Math.clamp(start, -1, size - 1);
    }

    private static int sliceStop(int size, int end, int stride, boolean toEnd) {
        if (toEnd) {
            return stride > 0 ? size : -1;
        }
        int stop = end < 0 ? end + size : end;
        return stride > 0 ? Math.clamp(stop, 0, size) : Math.clamp(stop, -1, size - 1);
    }

    private static boolean inRange(int index, int stop, int stride) {
        return stride > 0 ? index < stop : index > stop;
    }

    private static boolean inRange(int[] index, int[] stop, int[] stride, int rank) {
        for (int axis = 0; axis < rank; axis++) {
            if (!inRange(index[axis], stop[axis], stride[axis])) {
                return false;
            }
        }
        return true;
    }

    /** Bytes per element, 0 for types that this program never reads. */
    private static int width(Tensor tensor) {
        return switch (tensor.type()) {
            case TYPE_INT8, TYPE_UINT8 -> 1;
            case TYPE_INT32 -> 4;
            default -> 0;
        };
    }

    private static int product(int[] shape, int from, int to) {
        int product = 1;
        for (int i = from; i < to; i++) {
            product *= shape[i];
        }
        return product;
    }

    private static Tensor tensor(Subgraph graph, int index) throws IOException {
        if (index < 0 || index >= graph.tensors().size()) {
            throw new IOException("Invalid tensor index " + index);
        }
        return graph.tensors().get(index);
    }

    private static Tensor inputTensor(Subgraph graph, Operator operator, int input)
            throws IOException {
        if (input >= operator.inputs().length) {
            throw new IOException("Operator is missing input " + input);
        }
        return tensor(graph, operator.inputs()[input]);
    }

    private static Tensor outputTensor(Subgraph graph, Operator operator, int output)
            throws IOException {
        return tensor(graph, output(graph, operator, output));
    }

    private static int output(Subgraph graph, Operator operator, int output) throws IOException {
        if (output >= operator.outputs().length) {
            throw new IOException("Operator is missing output " + output);
        }
        int index = operator.outputs()[output];
        Tensor tensor = tensor(graph, index);
        if (tensor.isConstant()) {
            throw new IOException("Operator output must not be constant");
        }
        return index;
    }

    private static void requireQuantized(Tensor tensor, int... types) throws IOException {
        if (Arrays.stream(types).noneMatch(type -> type == tensor.type())) {
            throw new IOException("Unsupported type " + tensor.type() + " of " + tensor.name());
        }
        if (tensor.scale().length == 0 || tensor.zeroPoint().length == 0) {
            throw new IOException("Tensor " + tensor.name() + " is not quantized");
        }
        for (float scale : tensor.scale()) {
            if (!(scale > 0) || scale == Float.POSITIVE_INFINITY) {
                throw new IOException("Tensor " + tensor.name() + " has an invalid scale");
            }
        }
        int min = tensor.type() == TYPE_UINT8 ? 0 : -128;
        for (long zeroPoint : tensor.zeroPoint()) {
            if (zeroPoint < min || zeroPoint > min + 255) {
                throw new IOException("Tensor " + tensor.name() + " has an invalid zero point");
            }
        }
    }

    private static void requireConstant(Tensor tensor, int type, int rank) throws IOException {
        requireQuantized(tensor, type);
        if (!tensor.isConstant() || tensor.shape().length != rank) {
            throw new IOException("Expected a constant filter of rank " + rank);
        }
    }

    private static void requireSameQuantization(Tensor input, Tensor output) throws IOException {
        requireQuantized(input, TYPE_INT8, TYPE_UINT8);
        requireQuantized(output, input.type());
        if (input.scale()[0] != output.scale()[0]
                || input.zeroPoint()[0] != output.zeroPoint()[0]) {
            throw new IOException("Requantizing " + input.name() + " is not supported");
        }
    }

    private static int[] constantInts(Tensor tensor) throws IOException {
        if (tensor.type() != TYPE_INT32 || !tensor.isConstant()) {
            throw new IOException("Expected constant int32 values in " + tensor.name());
        }
        int[] values = new int[tensor.size()];
        ByteBuffer.wrap(tensor.data()).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(values);
        return values;
    }
}
