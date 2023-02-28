#!/bin/bash
#SBATCH -A COMPUTERLAB-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00

# Submit with sbatch --array=0-59:1 train.sh

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1
source /home/elyro2/rds/rds-t2-cs163-0cKEKVse28g/nar-env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/software/cuda/11.2/extras/CUPTI/lib64/

cd /home/elyro2/nar/cnap/cnap/training

params=(           "ppo-baseline" "transe-baseline" "xlvin")
include_executor=( "False"        "True"            "True")
run_executor=(     "False"        "False"           "True")

params_idx=$(($SLURM_ARRAY_TASK_ID%${#params[@]}))
seed=$(($SLURM_ARRAY_TASK_ID/${#params[@]}))

echo "Params ${params[$params_idx]}"
echo "Seed $seed"

python classic_control.py \
--env=acrobot \
--seed=$seed \
--graph_type=erdos-renyi \
--gnn_steps=1 \
--value_loss_coef=1 \
--lr=0.001 \
--include_transe=True \
--include_executor=${include_executor[$params_idx]} \
--run_executor=${run_executor[$params_idx]} \
--device=cuda \
--save_model=True \
--codebase_backup_dir=/home/elyro2/nar/cnap/experiments/exp_02_24_acrobot_baseline \
