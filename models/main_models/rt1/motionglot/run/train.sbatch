#!/bin/bash
#SBATCH -n 30
#SBATCH -t 72:00:00
#SBATCH -p 3090-gcondo --gres=gpu:8 --gres-flags=enforce-binding
#SBATCH --mem=120Gb

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e ft_outputs/new_data/tasks_train/%J.err
#SBATCH -o ft_outputs/new_data/tasks_train/%J.out

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=ahmed_jaafar@brown.edu

mkdir -p train_outputs

./run_train.sh