#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate motionglot

cd ../models/main_models/rt1/motionglot

DATASET='tokenized_dataset_pickles/lambda_task_gen_0.pkl'
TOKENIZER='lambda_tokenizer/lambda_task_gen_0'
OUTPUT_DIR='task_gen_ft_freeze_all'
MODEL_CKPT='/users/ajaafar/data/ajaafar/LaNMP-Dataset/models/main_models/rt1/motionglot/motionglot_ckpt/checkpoint-30700'


python train_lambda.py --dataset "$DATASET" --tokenizer_path "$TOKENIZER" --output_dir "$OUTPUT_DIR" --pre_train_model "$MODEL_CKPT" #--freeze_all --unfreeze_head