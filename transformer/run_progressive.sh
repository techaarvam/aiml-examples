#!/bin/bash
# --------------------------------------------------
# Tech Aarvam — Progressive Expansion Training
# 256 → 384 → 512 → 640 → 768 → 896 → 1024
#
# Each stage runs to Chinchilla 100% for its model size.
# Between stages: extend_dims_sigma.py expands the checkpoint.
# Every 5 epochs: model.pth saved to CHECKPOINT_DIR, entropy appended to CSV.
#
# Usage:
#   bash run_progressive.sh
#       Full run from d256.
#
#   bash run_progressive.sh --start-from d512
#       Skip d256/d384; extend checkpoint to d512 and train fresh.
#       Use this when a previous stage just completed cleanly.
#
#   bash run_progressive.sh --start-from d512 --no-extend
#       Skip d256/d384; resume d512 training from the shared checkpoint.
#       Use this when d512 was interrupted mid-run (extend already done).
#
#   bash run_progressive.sh --start-from d512 --no-extend --resume-epoch 23
#       Same as above but explicitly override start epoch.
# --------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT_DIR="runs/progressive/checkpoint"
ENTROPY_CSV="runs/progressive/entropy.csv"

DIMS=(   256  384  512  640  768  896 1024)
EPOCHS=( 32   40   51   66   83  104  127)

# ── Parse args ─────────────────────────────────────────────────────────────
START_FROM=""
NO_EXTEND=0
RESUME_EPOCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from)
            START_FROM="$2"; shift 2 ;;
        --no-extend)
            NO_EXTEND=1; shift ;;
        --resume-epoch)
            RESUME_EPOCH="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--start-from d<N>] [--no-extend] [--resume-epoch <N>]"
            exit 1 ;;
    esac
done

# ── Resolve start index ────────────────────────────────────────────────────
START_IDX=0
if [[ -n "$START_FROM" ]]; then
    DIM_LABEL="${START_FROM#d}"   # strip leading 'd' if present
    for i in "${!DIMS[@]}"; do
        if [[ "${DIMS[$i]}" == "$DIM_LABEL" ]]; then
            START_IDX=$i
            break
        fi
    done
    if [[ $START_IDX -eq 0 && "${DIMS[0]}" != "$DIM_LABEL" ]]; then
        echo "ERROR: --start-from '$START_FROM' not found in stage list: ${DIMS[*]}"
        exit 1
    fi
    echo "Starting from stage d${DIMS[$START_IDX]} (index $START_IDX)"
    if [[ $NO_EXTEND -eq 1 ]]; then
        echo "  --no-extend: skipping dimension expansion, resuming from checkpoint"
    fi
fi

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$(dirname "$ENTROPY_CSV")"

# ── Stage loop ─────────────────────────────────────────────────────────────
for i in "${!DIMS[@]}"; do
    if [[ $i -lt $START_IDX ]]; then
        echo "  Skipping stage d${DIMS[$i]} (before --start-from)"
        continue
    fi

    DIM=${DIMS[$i]}
    EPS=${EPOCHS[$i]}
    PROFILE="progressive_d${DIM}"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Stage d${DIM}  |  ${EPS} epochs  |  profile: ${PROFILE}"
    echo "══════════════════════════════════════════════════════"

    if [[ $i -eq 0 && $START_IDX -eq 0 ]]; then
        # ── First stage: cold start ───────────────────────────────────────
        systemd-inhibit --what=idle:sleep --why="progressive d${DIM}" \
        python runner.py runs.toml "$PROFILE" \
            --shared_checkpoint_dir "$CHECKPOINT_DIR" \
            --entropy_csv "$ENTROPY_CSV"

    elif [[ $i -eq $START_IDX && $NO_EXTEND -eq 1 ]]; then
        # ── Resume interrupted stage: skip extend, resume from checkpoint ─
        RESUME_FLAGS="--resume --model_file ${CHECKPOINT_DIR}/model.pth"
        if [[ -n "$RESUME_EPOCH" ]]; then
            RESUME_FLAGS="$RESUME_FLAGS --start_epoch $RESUME_EPOCH"
        fi
        systemd-inhibit --what=idle:sleep --why="progressive d${DIM}" \
        python runner.py runs.toml "$PROFILE" \
            $RESUME_FLAGS \
            --shared_checkpoint_dir "$CHECKPOINT_DIR" \
            --entropy_csv "$ENTROPY_CSV"

    else
        # ── New stage: extend checkpoint then train fresh ─────────────────
        PREV_DIM=${DIMS[$((i-1))]}
        echo "  Extending d${PREV_DIM} → d${DIM} ..."
        TEMP_PTH="${CHECKPOINT_DIR}/model_d${DIM}_tmp.pth"
        python extend_dims_sigma.py \
            "${CHECKPOINT_DIR}/model.pth" \
            "$TEMP_PTH" \
            "$DIM"
        mv "$TEMP_PTH" "${CHECKPOINT_DIR}/model.pth"
        echo "  Extended checkpoint ready → ${CHECKPOINT_DIR}/model.pth"

        systemd-inhibit --what=idle:sleep --why="progressive d${DIM}" \
        python runner.py runs.toml "$PROFILE" \
            --resume \
            --start_epoch 1 \
            --model_file "${CHECKPOINT_DIR}/model.pth" \
            --shared_checkpoint_dir "$CHECKPOINT_DIR" \
            --entropy_csv "$ENTROPY_CSV"
    fi

    # After first skipped-to stage, subsequent stages are always fresh extends
    START_IDX=$((START_IDX - 1))   # ensure future stages fall into the else branch
    NO_EXTEND=0                     # only applies to the first resumed stage

    echo "  Stage d${DIM} complete."
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Progressive training complete!"
echo "  Entropy log : $ENTROPY_CSV"
echo "  Final model : ${CHECKPOINT_DIR}/model.pth"
echo "══════════════════════════════════════════════════════"
