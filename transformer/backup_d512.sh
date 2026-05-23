#!/bin/bash
# Hourly backup of d512 run — pulls model.pth + train.log
# Remote: /root/transformer/runs/btm_d512_cont_20260522_203334
# Heartbeat every 2 min, backup every 60 min, resilient to missing model.pth

SERVER="root@207.102.87.207"
PORT=52853
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"
REMOTE_RUN="/root/transformer/runs/btm_d512_cont_20260522_203334"
LOCAL_DIR="/home/rambala/work/learn/aiml/transformer/btm_r2_backups/d512_20260522"
LOG="$LOCAL_DIR/backup.log"

BACKUP_INTERVAL=60   # minutes
HEARTBEAT_INTERVAL=2 # minutes

mkdir -p "$LOCAL_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

heartbeat() {
    local progress
    progress=$(ssh $SSH_OPTS -p "$PORT" "$SERVER" "cat $REMOTE_RUN/progress.txt 2>/dev/null" 2>/dev/null)
    if [ -n "$progress" ]; then
        log "--- heartbeat ---"
        echo "$progress" | grep -E "Epoch|Batch|Loss|Rate|ETA" | while read line; do
            echo "  $line" | tee -a "$LOG"
        done
    else
        log "--- heartbeat (no progress file yet) ---"
    fi
}

do_backup() {
    local ts
    ts=$(date '+%Y%m%d_%H%M%S')
    log "======== backup $ts ========"

    # check model.pth exists on server before pulling
    if ssh $SSH_OPTS -p "$PORT" "$SERVER" "test -f $REMOTE_RUN/model.pth" 2>/dev/null; then
        scp -C $SSH_OPTS -P "$PORT" \
            "$SERVER:$REMOTE_RUN/model.pth" \
            "$LOCAL_DIR/model_$ts.pth" >> "$LOG" 2>&1
        if [ $? -eq 0 ]; then
            # keep only latest as model_latest.pth
            cp "$LOCAL_DIR/model_$ts.pth" "$LOCAL_DIR/model_latest.pth"
            log "model.pth → model_$ts.pth ($(du -h "$LOCAL_DIR/model_$ts.pth" | cut -f1))"
        else
            log "ERROR: scp model.pth failed"
        fi
    else
        log "model.pth not present yet — skipping"
    fi

    # train.log: always try, not fatal if missing
    scp -C $SSH_OPTS -P "$PORT" \
        "$SERVER:$REMOTE_RUN/train.log" \
        "$LOCAL_DIR/train.log" >> "$LOG" 2>&1 \
        && log "train.log pulled" || log "train.log not available yet"
}

log "Backup daemon started — heartbeat every ${HEARTBEAT_INTERVAL}m, backup every ${BACKUP_INTERVAL}m"
log "Remote : $REMOTE_RUN"
log "Local  : $LOCAL_DIR"

do_backup

minutes=0
while true; do
    sleep $(( HEARTBEAT_INTERVAL * 60 ))
    minutes=$(( minutes + HEARTBEAT_INTERVAL ))
    heartbeat
    if [ $(( minutes % BACKUP_INTERVAL )) -eq 0 ]; then
        do_backup
    fi
done
