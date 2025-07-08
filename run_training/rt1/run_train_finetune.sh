#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate rt1

# module load cuda/12.2.0-4lgnkrh
# module load cudnn/8.9.6.50-12-56zgdoa

cd ../../models/main_models/rt1

HP=2
SPLIT_TYPE="task_split"
ARCH_TYPE="transformer" #transformer, lstm, mamba
DIST="no" #yes, no
SUBSET_AMT='75' #25, 50, 75
TEST_SCENE=''
DATASET_PATH="/oscar/data/stellex/shared/lanmp/lanmp_dataset_newest_compressed_img.hdf5" #wherever you stored the simualted LAMBDA dataset
LOAD_CHECKPOINT="/users/ajaafar/data/shared/lanmp/pretrained_rt1_ckpt/checkpoint_best.pt" #wherever you stored the pretrained rt-1 pt file
CHECKPOINT_DIR="results/checkpoints/train_${ARCH_TYPE}-${DIST}_dist-${SPLIT_TYPE}-subset_${SUBSET_AMT}-scene_${TEST_SCENE}-HP_${HP}"
VAL_LOSS_DIR="results/val_losses/train_${ARCH_TYPE}-${DIST}_dist-${SPLIT_TYPE}-subset_${SUBSET_AMT}-scene_${TEST_SCENE}-HP_${HP}"

EPOCHS=30
EVAL_FREQ=50
CHECKPOINT_FREQ=0
TRAIN_BATCH=5
EVAL_BATCH=5
TRAIN_SUBBATCH=8
EVAL_SUBBATCH=8
LR=1e-4
LR_SCHED="plateau"
FACTOR=0.9
GAMMA=0.8
PATIENCE=1
# LOW_DIV='--low_div' $LOW_DIV

python main_train_ft.py --dataset-path "$DATASET_PATH" --split-type "$SPLIT_TYPE" --epochs "$EPOCHS" --checkpoint-dir "$CHECKPOINT_DIR" --eval-freq "$EVAL_FREQ" --val_loss_dir "$VAL_LOSS_DIR" --wandb --checkpoint-freq "$CHECKPOINT_FREQ" --train-batch-size "$TRAIN_BATCH" --eval-batch-size "$EVAL_BATCH" --lr "$LR" --lr_sched "$LR_SCHED" --gamma "$GAMMA" --factor "$FACTOR" --patience "$PATIENCE" --train-subbatch "$TRAIN_SUBBATCH" --eval-subbatch "$EVAL_SUBBATCH" --test-scene "$TEST_SCENE" --arch-type "$ARCH_TYPE" --subset-amt "$SUBSET_AMT" #--load-checkpoint "$LOAD_CHECKPOINT" #--freeze #--use-dist 