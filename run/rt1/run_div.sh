#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd models/main_models/rt1

SPLIT_TYPE='cluster'
# LOW_DIV='--low_div' $LOW_DIV
EPOCHS=1
CHECKPOINT_DIR='checkpoints/cluster_high_5'
EVAL_FREQ=50
TRAIN_SUBBATCH=5

python main_ft.py --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --train-subbatch "$TRAIN_SUBBATCH"