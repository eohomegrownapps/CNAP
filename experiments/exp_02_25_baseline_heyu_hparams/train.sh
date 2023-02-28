#!/bin/bash
#SBATCH -A DRAGE-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:00

# Submit with sbatch --array=0-239:1 train.sh

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1
source /home/elyro2/rds/rds-t2-cs163-0cKEKVse28g/nar-env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/software/cuda/11.2/extras/CUPTI/lib64/

cd /home/elyro2/nar/cnap/cnap/training

experiment="exp_02_25_baseline_heyu_hparams"

params=(           "baseline"  "xlvin-no_exe" "xlvin" "baseline-pretrained_transe"  "xlvin-no_exe-pretrained_transe" "xlvin-pretrained_transe")
include_executor=( "False"     "True"         "True"  "False"                       "True"                           "True")
include_transe=(   "False"     "False"        "False" "True"                        "True"                           "True")
run_executor=(     "False"     "False"        "True"  "False"                       "False"                          "True")

n_epochs=("1" "10")

n_epochs_idx=$((($SLURM_ARRAY_TASK_ID/${#params[@]})%${#n_epochs[@]}))
params_idx=$(($SLURM_ARRAY_TASK_ID%${#params[@]}))
seed=$(($SLURM_ARRAY_TASK_ID)) # $(($SLURM_ARRAY_TASK_ID/${#params[@]}))

echo "Params ${params[$params_idx]}"
echo "PPO epochs ${n_epochs[$n_epochs_idx]}"
echo "Seed $seed"

python classic_control.py \
--env=acrobot \
--seed=$seed \
--graph_type=erdos-renyi \
--gnn_steps=2 \
`# PPO` \
--lr=0.0003 \
--gamma=0.99 \
--value_loss_coef=0.5 \
--entropy_coef=0.01 \
--transe_loss_coef=0.001 \
--gae_lambda=0.95 \
--max_grad_norm=0.5 \
--clip_param=0.2 \
--ppo_epoch=${n_epochs[$n_epochs_idx]} \
`# XLVIN policy` \
--freeze_encoder=False \
--freeze_executor=True \
--include_transe=${include_transe[$params_idx]} \
--include_executor=${include_executor[$params_idx]} \
--run_executor=${run_executor[$params_idx]} \
`# Saving` \
--device=cuda \
--save_model=True \
--codebase_backup_dir=/home/elyro2/nar/cnap/experiments/${experiment}/codebases \
--save_dir=/home/elyro2/nar/cnap/experiments/${experiment}/models \
--log_dir=/home/elyro2/nar/cnap/experiments/${experiment}/logs \
