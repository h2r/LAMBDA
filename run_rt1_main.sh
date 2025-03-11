#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH -p gpu --gres=gpu:1
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate rt1_pytorch2

#try 1
#module load cuda/12.2.0-4lgnkrh
#module load cudnn/8.9.6.50-12-56zgdoa

#try working
#module load cuda/11.8.0-lpttyok
#module load cudnn/8.7.0.84-11.8-lg2dpd5

#try 2:
#module load  cuda/10.2.89-xnfjmrt
#module load cudnn/8.7.0.84-11.8-lg2dpd5
cd rt1-pytorch
python main.py --datasets bridge --train-split "train[:500]" --eval-split "train[:500]" --train-batch-size 8 --eval-batch-size 8 --eval-freq 100 --checkpoint-freq 100 --checkpoint-dir /users/sjulian2/data/sjulian2/rt1-pytorch/checkpoints/bridge
#python main.py --dataset "jaco_play"  --train-split "train[:500]" --eval-split "train[:500]" --train-batch-size 8 --eval-batch-size 8 --eval-freq 100 --checkpoint-freq 100
echo $PATH
