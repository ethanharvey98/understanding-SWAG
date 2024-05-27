#!/bin/bash
#SBATCH -n 16
#SBATCH -p gpu
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-9%1

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
    "python ../src/ImageNet_v2_main.py --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2' --model_name='ImageNet_v2_random_state=6001' --random_state=6001 --wandb --wandb_project='ImageNet'"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate