#!/bin/bash
#SBATCH -n 16
#SBATCH -p gpu
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-3

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/swag_ImageNet_v2_main.py --checkpoint_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2/ImageNet_v2_random_state=1001.pt' --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/swag_ImageNet_v2' --K=20 --lr=0.01 --model_name='swag_epochs=30_K=20_lr=0.01_no_cov_factor=False_random_state=1001' --random_state=1001 --wandb --wandb_project='understanding-SWAG'"
    "python ../src/swag_ImageNet_v2_main.py --checkpoint_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2/ImageNet_v2_random_state=1001.pt' --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/swag_ImageNet_v2' --K=20 --lr=0.005 --model_name='swag_epochs=30_K=20_lr=0.005_no_cov_factor=False_random_state=1001' --random_state=1001 --wandb --wandb_project='understanding-SWAG'"
    "python ../src/swag_ImageNet_v2_main.py --checkpoint_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2/ImageNet_v2_random_state=1001.pt' --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/swag_ImageNet_v2' --K=20 --lr=0.001 --model_name='swag_epochs=30_K=20_lr=0.001_no_cov_factor=False_random_state=1001' --random_state=1001 --wandb --wandb_project='understanding-SWAG'"
    "python ../src/swag_ImageNet_v2_main.py --checkpoint_path='/cluster/home/eharve06/understanding-SWAG/experiments/ImageNet_v2/ImageNet_v2_random_state=1001.pt' --experiments_path='/cluster/home/eharve06/understanding-SWAG/experiments/swag_ImageNet_v2' --K=20 --lr=0.0005 --model_name='swag_epochs=30_K=20_lr=0.0005_no_cov_factor=False_random_state=1001' --random_state=1001 --wandb --wandb_project='understanding-SWAG'"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate