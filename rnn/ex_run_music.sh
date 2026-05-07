#!/bin/bash
# Example run script for music RNN
# Usage: ./ex_run_music.sh          — train + generate + synthesise
#        ./ex_run_music.sh --nogen  — load saved model, generate + synthesise

MODEL_FILE="music/music_dataset.npz.model.pt"

if [[ "$1" == "--nogen" ]]; then
    python music/rnn.py \
        --num-layers 3 \
        --input music/music_dataset.npz \
        --batch-size 1024 \
        --epoch 500 \
        --lr 0.001 \
        --optimizer adam \
        --model-file "$MODEL_FILE" \
        --seq_len 512 \
        --num_samples 40
else
    python music/rnn.py \
        --num-layers 3 \
        --input music/music_dataset.npz \
        --batch-size 1024 \
        --epoch 500 \
        --lr 0.001 \
        --optimizer adam \
        --seq_len 512 \
        --num_samples 40
fi

python music/synth.py \
    --dataset music/generated.npz \
    --outdir music/gen \
    --n 40
