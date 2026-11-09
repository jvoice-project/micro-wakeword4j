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
package io.github.jvoiceproject.microwakeword.internal;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small JSON reader for model manifests.
 *
 * <p>Objects become {@link Map}s, arrays {@link List}s, numbers {@link Double}s.
 *
 * @author Dan Cunningham - Initial contribution
 */
public final class Json {
    /** Deeper nesting than manifests use, and shallow enough to stay well within the stack. */
    private static final int MAX_DEPTH = 64;

    private final String text;
    private int position;

    private Json(String text) {
        this.text = text;
    }

    public static Object parse(String text) throws IOException {
        Json json = new Json(text);
        Object value = json.value(0);
        json.skipWhitespace();
        if (json.position != text.length()) {
            throw json.error("Unexpected content");
        }
        return value;
    }

    private Object value(int depth) throws IOException {
        skipWhitespace();
        if (position >= text.length()) {
            throw error("Unexpected end");
        }
        char c = text.charAt(position);
        if ((c == '{' || c == '[') && depth >= MAX_DEPTH) {
            throw error("Too deeply nested");
        }
        return switch (c) {
            case '{' -> object(depth + 1);
            case '[' -> array(depth + 1);
            case '"' -> string();
            case 't' -> literal("true", Boolean.TRUE);
            case 'f' -> literal("false", Boolean.FALSE);
            case 'n' -> literal("null", null);
            default -> number();
        };
    }

    private Map<String, Object> object(int depth) throws IOException {
        Map<String, Object> object = new LinkedHashMap<>();
        position++;
        skipWhitespace();
        if (consume('}')) {
            return object;
        }
        do {
            skipWhitespace();
            if (position >= text.length() || text.charAt(position) != '"') {
                throw error("Expected a key");
            }
            String key = string();
            skipWhitespace();
            if (!consume(':')) {
                throw error("Expected ':'");
            }
            object.put(key, value(depth));
            skipWhitespace();
        } while (consume(','));
        if (!consume('}')) {
            throw error("Expected '}'");
        }
        return object;
    }

    private List<Object> array(int depth) throws IOException {
        List<Object> array = new ArrayList<>();
        position++;
        skipWhitespace();
        if (consume(']')) {
            return array;
        }
        do {
            array.add(value(depth));
            skipWhitespace();
        } while (consume(','));
        if (!consume(']')) {
            throw error("Expected ']'");
        }
        return array;
    }

    private String string() throws IOException {
        StringBuilder value = new StringBuilder();
        position++;
        while (position < text.length()) {
            char c = text.charAt(position++);
            if (c == '"') {
                return value.toString();
            }
            if (c != '\\') {
                value.append(c);
                continue;
            }
            if (position >= text.length()) {
                break;
            }
            char escaped = text.charAt(position++);
            switch (escaped) {
                case '"', '\\', '/' -> value.append(escaped);
                case 'b' -> value.append('\b');
                case 'f' -> value.append('\f');
                case 'n' -> value.append('\n');
                case 'r' -> value.append('\r');
                case 't' -> value.append('\t');
                case 'u' -> {
                    if (position + 4 > text.length()) {
                        throw error("Invalid unicode escape");
                    }
                    String hex = text.substring(position, position + 4);
                    // parseInt alone would also take a sign
                    for (int i = 0; i < hex.length(); i++) {
                        if ("0123456789abcdefABCDEF".indexOf(hex.charAt(i)) < 0) {
                            throw error("Invalid unicode escape");
                        }
                    }
                    value.append((char) Integer.parseInt(hex, 16));
                    position += 4;
                }
                default -> throw error("Invalid escape");
            }
        }
        throw error("Unterminated string");
    }

    private Double number() throws IOException {
        int start = position;
        while (position < text.length() && "+-.eE0123456789".indexOf(text.charAt(position)) >= 0) {
            position++;
        }
        try {
            return Double.valueOf(text.substring(start, position));
        } catch (NumberFormatException e) {
            position = start;
            throw error("Invalid value");
        }
    }

    private Object literal(String word, Object value) throws IOException {
        if (!text.startsWith(word, position)) {
            throw error("Invalid value");
        }
        position += word.length();
        return value;
    }

    private boolean consume(char c) {
        if (position < text.length() && text.charAt(position) == c) {
            position++;
            return true;
        }
        return false;
    }

    private void skipWhitespace() {
        while (position < text.length() && Character.isWhitespace(text.charAt(position))) {
            position++;
        }
    }

    private IOException error(String message) {
        return new IOException(message + " at position " + position + " of model manifest");
    }
}
