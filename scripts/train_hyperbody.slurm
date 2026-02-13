#!/bin/bash
#SBATCH --job-name=hyperbody
#SBATCH --partition=short
#SBATCH --nodelist=gpu20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=100G
#SBATCH --output=logs/hyperbody_%j.out
#SBATCH --error=logs/hyperbody_%j.err

# ---------- Environment ----------
source ~/.bashrc
conda activate nnunet

PROJECT_ROOT="/home/comp/csrkzhu/code/HyperBodyV2"
DATA_ROOT="${PROJECT_ROOT}/nnUnet/nnUNet_data"

export nnUNet_raw="${DATA_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${DATA_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${DATA_ROOT}/nnUNet_results"

# ---------- Training config ----------
DATASET="Dataset501_HyperBody"
CONFIG="3d_fullres"
FOLD=0
TRAINER="nnUNetTrainerHyperBody"
NUM_GPUS=2

# nnUNet data augmentation workers (per GPU)
export nnUNet_n_proc_DA=12

# ---------- Launch DDP training ----------
echo "=== HyperBody Training ==="
echo "Node: $(hostname)"
echo "GPUs: ${NUM_GPUS}x L40S"
echo "Dataset: ${DATASET}"
echo "Trainer: ${TRAINER}"
echo "Fold: ${FOLD}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================="

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    $(which nnUNetv2_train) \
    ${DATASET} ${CONFIG} ${FOLD} \
    -tr ${TRAINER}
