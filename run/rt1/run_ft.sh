#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
# source /mnt/miniconda3/etc/profile.d/conda.sh

conda activate rt1

# module load cuda/11.8.0-lpttyok
# module load cudnn/8.7.0.84-11.8-lg2dpd5

cd ../../models/main_models/rt1

HP=9
SPLIT_TYPE="k_fold_scene"
SUBSET_AMT=''
TEST_SCENE=4
# LOAD_CHECKPOINT='/mnt/ahmed/rt1/pretrain_ckpts/checkpoint_best.pt'
LOAD_CHECKPOINT="/users/ajaafar/data/shared/lanmp/pretrained_rt1_ckpt/checkpoint_best.pt"
CHECKPOINT_DIR="results/checkpoints/train-rt1-nodist-${SPLIT_TYPE}-${SUBSET_AMT}-scene${TEST_SCENE}-HP${HP}"
VAL_LOSS_DIR="results/val_losses/train-rt1-nodist-${SPLIT_TYPE}-${SUBSET_AMT}-scene${TEST_SCENE}-HP${HP}"
EPOCHS=30
EVAL_FREQ=50
CHECKPOINT_FREQ=0
TRAIN_BATCH=3
EVAL_BATCH=3
TRAIN_SUBBATCH=10
EVAL_SUBBATCH=10
LR=1e-4
LR_SCHED="plateau"
FACTOR=0.5
GAMMA=0.8
PATIENCE=1
# LOW_DIV='--low_div' $LOW_DIV

python main_ft.py --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --val_loss_dir "$VAL_LOSS_DIR" --wandb --checkpoint-freq "$CHECKPOINT_FREQ" --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr "$LR" --lr_sched "$LR_SCHED" --gamma "$GAMMA" --factor "$FACTOR" --patience "$PATIENCE" --train-subbatch "$TRAIN_SUBBATCH" --eval-subbatch "$EVAL_SUBBATCH" --test-scene "$TEST_SCENE" # --subset-amt "$SUBSET_AMT" #--load-checkpoint "$LOAD_CHECKPOINT" #--freeze #--use-dist 