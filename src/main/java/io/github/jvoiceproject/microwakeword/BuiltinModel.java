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

/**
 * The wake word models bundled with this library: v2 models of the <a
 * href="https://github.com/esphome/micro-wake-word-models">ESPHome model collection</a>, and models
 * trained by the jVoice project with the pipeline in {@code training/}.
 *
 * @author Dan Cunningham - Initial contribution
 */
public enum BuiltinModel {
    /** "Alexa" */
    ALEXA("alexa"),
    /** "Hey Jarvis" */
    HEY_JARVIS("hey_jarvis"),
    /** "Hey Jeeves", trained by the jVoice project */
    HEY_JEEVES("hey_jeeves"),
    /** "Hey Mycroft" */
    HEY_MYCROFT("hey_mycroft"),
    /** "Hey openHAB", trained by the jVoice project */
    HEY_OPENHAB("hey_openhab");

    private final String id;

    BuiltinModel(String id) {
        this.id = id;
    }

    /** Name of the model's manifest and model files, without extension. */
    public String getId() {
        return id;
    }
}
