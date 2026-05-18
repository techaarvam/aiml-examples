#!/bin/bash
# Hourly backup of w128 server run — pulls pth + vocab + onnx
# Run dir : /root/aiml-examples/transformer/runs/w128_20260518_162121/
# ONNX conversion runs on server CPU (CUDA_VISIBLE_DEVICES="") so GPU training is unaffected.

SERVER="root@194.228.55.129"
PORT=35288
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes"
REMOTE_BASE="/root/aiml-examples/transformer"
REMOTE_RUN="$REMOTE_BASE/runs/w128_20260518_162121"
LOCAL_BASE="/home/rambala/work/learn/aiml/transformer/runs/w128_server_bkp"
LOG="$LOCAL_BASE/backup.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

do_backup() {
    local ts
    ts=$(date '+%Y%m%d_%H%M%S')
    local dest="$LOCAL_BASE/$ts"
    mkdir -p "$dest"

    log "======== checkpoint $ts ========"

    # progress snapshot
    ssh $SSH_OPTS -p "$PORT" "$SERVER" "cat $REMOTE_RUN/progress.txt" 2>&1 | tee -a "$LOG"

    # ONNX conversion on server CPU (won't touch the GPU)
    log "Converting pth → onnx on server (CPU)..."
    ssh $SSH_OPTS -p "$PORT" "$SERVER" "
        cd $REMOTE_BASE
        CUDA_VISIBLE_DEVICES='' /venv/main/bin/python pth2onnx.py \
            --embedding_type learned \
            --vecDims 512 \
            --num_heads 8 \
            --num_layers 8 \
            --window_size 128 \
            --output_type indices \
            --vocab_file  $REMOTE_RUN/vocab.json \
            --model_file  $REMOTE_RUN/model.pth \
            --output      $REMOTE_RUN/model.onnx
    " >> "$LOG" 2>&1

    if [ $? -ne 0 ]; then
        log "ERROR: ONNX conversion failed — skipping download for this checkpoint."
        return 1
    fi
    log "ONNX conversion done."

    # pull the three files
    log "Downloading model.pth, vocab.json, model.onnx..."
    scp $SSH_OPTS -P "$PORT" \
        "$SERVER:$REMOTE_RUN/model.pth" \
        "$SERVER:$REMOTE_RUN/vocab.json" \
        "$SERVER:$REMOTE_RUN/model.onnx" \
        "$dest/" >> "$LOG" 2>&1

    if [ $? -eq 0 ]; then
        log "Saved to $dest/"
        ls -lh "$dest/" | tee -a "$LOG"
    else
        log "ERROR: scp download failed."
        return 1
    fi
}

log "Backup daemon started. Interval: 60 min."
log "Remote: $REMOTE_RUN"
log "Local : $LOCAL_BASE"

do_backup

while true; do
    log "Next backup in 60 min..."
    sleep 3600
    do_backup
done
