#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1

module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

CHECKPOINT_DIR='/users/ajaafar/scratch/rt1_pretrain_ckpts17'
CHECKPOINT_FREQ=0
EVAL_FREQ=80
VAL_LOSS_DIR='val_losses/rt1_pretrain_new17'
TRAIN_BATCH=6
EVAL_BATCH=6
LR=1e-15
LR_SCHED="plateau"
GAMMA=0.999
FACTOR=0.9
PATIENCE=1


python main_pretrain.py --checkpoint-dir "$CHECKPOINT_DIR" --val_loss_dir "$VAL_LOSS_DIR" --eval-freq "$EVAL_FREQ" --wandb --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr "$LR" --lr_sched "$LR_SCHED" --gamma "$GAMMA" --factor "$FACTOR" --patience "$PATIENCE" --checkpoint-freq "$CHECKPOINT_FREQ"