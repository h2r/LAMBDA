#!/bin/bash

set -e
set -u

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda activate motionglot

cd ../../models/main_models/rt1/motionglot


SPLIT_TYPE="task_split"
TEST_SCENE=''
DATASET='tokenized_dataset_pickles/lambda_task_gen.pkl'
TOKENIZER='lambda_tokenizer/lambda_task_gen'


python main_ft.py --split_type "$SPLIT_TYPE" --test_scene "$TEST_SCENE" --dataset "$DATASET" --tokenizer_path $TOKENIZER$