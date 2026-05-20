#!/bin/bash
# BTM Round 2 backup — pulls model.pth + train log from all 4 machines every 120 min
# Double-buffer: downloads to /tmp first, then atomically overwrites local copy

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15"
LOCAL_BASE="/home/rambala/work/learn/aiml/transformer/btm_r2_backups"
LOG="$LOCAL_BASE/backup.log"

declare -A HOST PORT LOG_PATH
HOST[1]="74.48.78.46";   PORT[1]=61077
HOST[2]="61.228.34.3";   PORT[2]=31302
HOST[3]="82.65.196.171";   PORT[3]=40095
HOST[4]="207.102.87.207"; PORT[4]=52853

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

backup_machine() {
    local m=$1
    local host="${HOST[$m]}"
    local port="${PORT[$m]}"

    # find latest model.pth
    local remote_model
    remote_model=$(ssh $SSH_OPTS -p "$port" "root@$host" \
        "ls -t /root/transformer/runs/btm_w64_*/model.pth 2>/dev/null | head -1")

    # model.pth
    if [ -n "$remote_model" ]; then
        scp $SSH_OPTS -P "$port" "root@$host:$remote_model" "/tmp/btm_m${m}_model.pth" 2>>"$LOG"
        if [ $? -eq 0 ]; then
            cp "/tmp/btm_m${m}_model.pth" "$LOCAL_BASE/model_m${m}.pth"
            log "M${m} model.pth OK ($(du -sh "$LOCAL_BASE/model_m${m}.pth" | cut -f1))"
        else
            log "M${m} model.pth FAILED"
        fi
    else
        log "M${m} no model.pth found yet"
    fi

    # train log
    scp $SSH_OPTS -P "$port" \
        "root@$host:/root/transformer/raw_data/btm_machines/machine${m}/train_m${m}.log" \
        "/tmp/btm_m${m}_train.log" 2>>"$LOG"
    if [ $? -eq 0 ]; then
        cp "/tmp/btm_m${m}_train.log" "$LOCAL_BASE/train_m${m}.log"
        log "M${m} train log OK"
    else
        log "M${m} train log FAILED"
    fi
}

do_backup() {
    log "======== backup start ========"
    for m in 1 2 3 4; do
        backup_machine "$m" &
    done
    wait
    log "======== backup done ========"
}

log "BTM r2 backup daemon started — interval 120 min"
log "Local: $LOCAL_BASE"

do_backup

while true; do
    log "Next backup in 120 min..."
    sleep 7200
    do_backup
done
