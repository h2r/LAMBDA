#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate motionglot

cd ../../models/main_models/rt1/motionglot

DATASET='tokenized_dataset_pickles/lambda_task_gen_0.pkl'
TOKENIZER='lambda_tokenizer/lambda_task_gen_0'
OUTPUT_DIR='task_gen_ft_trash'
MODEL_CKPT='../results/checkpoints/motionglot-nodist-task_split' #set to modify default for finetuning


python train_lambda.py --dataset "$DATASET" --tokenizer_path "$TOKENIZER" --output_dir "$OUTPUT_DIR" --pre_train_model "$MODEL_CKPT" #--freeze_all --unfreeze_head