#!/usr/bin/env bash
#SBATCH --job-name=dnadiffusion
#SBATCH --account openbioml
#SBATCH --partition=a40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=/weka/ssenan/dnadiffusion/training/logs/%A_%a.out
#SBATCH --cpus-per-gpu=10

#export NCCL_PROTO=simple

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export NCCL_TREE_THRESHOLD=0
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_pml="^cm"
export OMPI_MCA_btl="^openib"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export WORLD_SIZE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export JOB_COMMENT="Key=Monitoring,Value=ON"


echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

#export WANDB_DIR="/fsx/ssenan/dnadiffusion/script/outputs"
#export WANDB_CACHE_DIR="/fsx/ssenan/.cache"
# export WANDB_MODE="online"
# WANDB_API_KEY=8070036ce797b0d26291b5ed1a132e3bd36ea27d
# export WANDB_API_KEY


echo go COUNT_NODE=$COUNT_NODE
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo WORLD_SIZE=$WORLD_SIZE

export PATH=/weka/ssenan/micromamba/:$PATH
#source /fsx/ssenan/mambaforge/bin/activate dnadiffusion
eval "$(micromamba shell hook --shell bash)"
micromamba activate dnadiffusion

cd /weka/ssenan/dnadiffusion/training

accelerate launch \
    --num_processes $(( 8 * $COUNT_NODE )) \
    --num_machines "$COUNT_NODE" \
    --machine_rank "$THEID" \
    --multi_gpu \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision 'bf16' \
    train.py
    # --dynamo_backend 'no' \