#!/bin/sh
# Trains the given wake word ids, or every one in wake_words.json, skipping those that already have a model.
# Each logs to free/<id>/training.log. Run prepare.py first.
cd "$(dirname "$0")" || exit 1
ids=${*:-$(.venv-train/bin/python -c "import json; print(' '.join(w['id'] for w in json.load(open('wake_words.json'))))")}
for id in $ids; do
    if [ -f "free/$id/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite" ]; then
        continue
    fi
    .venv-train/bin/python -m microwakeword.model_train_eval \
        --training_config="free/$id/training_parameters.yaml" \
        --train 1 \
        --restore_checkpoint 1 \
        --test_tf_nonstreaming 0 \
        --test_tflite_nonstreaming 0 \
        --test_tflite_nonstreaming_quantized 0 \
        --test_tflite_streaming 0 \
        --test_tflite_streaming_quantized 1 \
        --use_weights best_weights \
        mixednet \
        --pointwise_filters 64,64,64,64 \
        --repeat_in_block 1,1,1,1 \
        --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
        --residual_connection 0,0,0,0 \
        --first_conv_filters 32 \
        --first_conv_kernel_size 5 \
        --stride 3 \
        > "free/$id/training.log" 2>&1
done
