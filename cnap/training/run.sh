#!/bin/bash
module purge
module unload cuda/8.0
module load rhel7/default-gpu
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1
source /home/elyro2/rds/rds-t2-cs163-0cKEKVse28g/nar-env/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/software/cuda/11.2/extras/CUPTI/lib64/
pwd

python classic_control.py \
--env=acrobot \
--seed=2 \
--graph_type=erdos-renyi \
--gnn_steps=1 \
--value_loss_coef=1 \
--lr=0.001 \
--include_transe=True \
--include_executor=True \
--device=cuda
