#!/bin/bash

set -e
set -u

cd ..
source alfred_env2/bin/activate
cd LaNMP-Dataset
cd models/main_models
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
export ALFRED_ROOT=$(pwd)/alfred
cd alfred
NUM="5b"
DIV="high_scene_low_num"
DOUT="exp/model:seq2seq_im_mask_discrete_relative_${DIV}_${NUM}"
PP_DATA="data/feats_discrete_relative_${DIV}_${NUM}"
SPLIT_KEYS="data/splits/split_keys_discrete_relative_${DIV}_${NUM}.json"

python models/train/train_seq2seq.py --model seq2seq_im_mask --dout "$DOUT" --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --class_mode --preprocess --relative --pp_data "$PP_DATA" --split_keys "$SPLIT_KEYS" --splits_folds "div"