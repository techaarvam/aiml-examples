#!/bin/bash
# Example run script for language RNN
# Usage: ./ex_run_lang.sh          — train + generate
#        ./ex_run_lang.sh --nogen  — load saved model, generate only

MODEL_FILE="lang/raw_data/combined.txt.model.pt"

if [[ "$1" == "--nogen" ]]; then
    python lang/rnn.py \
        --num-layers 3 \
        --window-size 8 \
        --hidden-dim 128 \
        --optimizer adam \
        --lr 0.001 \
        --epoch 100 \
        --input lang/raw_data/combined.txt \
        --output-mode softmax \
        --output-size 100 \
        --batch-size 1024 \
        --model-file "$MODEL_FILE"
else
    python lang/rnn.py \
        --num-layers 3 \
        --window-size 8 \
        --hidden-dim 128 \
        --optimizer adam \
        --lr 0.001 \
        --epoch 100 \
        --input lang/raw_data/combined.txt \
        --output-mode softmax \
        --output-size 100 \
        --batch-size 1024
fi
