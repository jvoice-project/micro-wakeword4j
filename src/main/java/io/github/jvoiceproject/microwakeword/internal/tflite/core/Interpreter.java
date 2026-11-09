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

import java.util.Arrays;

/**
 * Runs a {@link Program} and holds the state of one audio stream: the intermediate tensors and the
 * variables in which streaming models keep earlier frames.
 *
 * @author Dan Cunningham - Initial contribution
 */
public final class Interpreter {
    private final Program program;
    private final byte[][][] tensors;
    private final byte[][] variables;
    private final boolean[] called;

    public Interpreter(Program program) {
        this.program = program;
        int subgraphs = program.operators.size();
        tensors = new byte[subgraphs][][];
        for (int s = 0; s < subgraphs; s++) {
            byte[][] constants = program.constants.get(s);
            int[] sizes = program.bufferSizes.get(s);
            tensors[s] = new byte[constants.length][];
            for (int t = 0; t < constants.length; t++) {
                // constants are shared between interpreters and never written
                tensors[s][t] = sizes[t] >= 0 ? new byte[sizes[t]] : constants[t];
            }
        }
        variables = new byte[program.variableSizes.length][];
        for (int v = 0; v < variables.length; v++) {
            variables[v] = new byte[Math.max(program.variableSizes[v], 0)];
        }
        called = new boolean[subgraphs];
    }

    /** The quantized model input, to be filled before {@link #invoke()}. */
    public byte[] input() {
        return tensors[0][program.inputTensor];
    }

    /** Runs the main subgraph and returns its quantized output value. */
    public int invoke() {
        run(0);
        byte output = tensors[0][program.outputTensor][0];
        return program.unsignedOutput ? output & 0xFF : output;
    }

    /**
     * Forgets the stream, so the next {@link #invoke()} initializes the variables again. They are
     * also cleared, for models without an initialization subgraph.
     */
    public void reset() {
        Arrays.fill(called, false);
        for (byte[] variable : variables) {
            Arrays.fill(variable, (byte) 0);
        }
    }

    byte[] variable(int id) {
        return variables[id];
    }

    void callOnce(int subgraph) {
        if (!called[subgraph]) {
            called[subgraph] = true;
            run(subgraph);
        }
    }

    private void run(int subgraph) {
        for (Program.Op op : program.operators.get(subgraph)) {
            op.run(this, tensors[subgraph]);
        }
    }
}
