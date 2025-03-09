#!/bin/bash
#SBATCH -n 25
#SBATCH -t 30:00:00
#SBATCH -p 3090-gcondo --gres=gpu:6 --gres-flags=enforce-binding
#SBATCH --mem=75Gb

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e train_outputs_ft/task_gen/%J.err
#SBATCH -o train_outputs_ft/task_gen/%J.out

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=ahmed_jaafar@brown.edu

mkdir -p train_outputs

./run_train.sh