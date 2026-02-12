#!/bin/bash
#SBATCH --job-name=nnunet_501
#SBATCH --partition=short
#SBATCH --nodelist=gpu20
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=slurm/logs/nnunet_%j.log
#SBATCH --error=slurm/logs/nnunet_%j.err

set -e

# ========== Environment ==========
source activate nnunet
cd /home/comp/csrkzhu/code/Compare/nnUNet

# Disable Python stdout buffering so print() output appears in SLURM log in real time
export PYTHONUNBUFFERED=1

export nnUNet_raw="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_results"

DATASET=501
CONFIG=3d_fullres
TRAINER=nnUNetTrainerNoDA
FOLD=0

echo "=========================================="
echo "nnUNet Training: Dataset ${DATASET}, Fold ${FOLD}"
echo "Trainer: ${TRAINER}, 2-GPU DDP"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"
echo "Start time: $(date)"
echo "=========================================="

# ========== Train Fold 0 (2-GPU DDP) ==========
nnUNetv2_train ${DATASET} ${CONFIG} ${FOLD} --npz -tr ${TRAINER} -num_gpus 2
echo "[$(date)] Fold ${FOLD} training completed"

# ========== Inference on Test Set ==========
echo ""
echo "[$(date)] Running inference on test set..."

PRED_DIR="${nnUNet_results}/Dataset${DATASET}_HyperBody/${TRAINER}__nnUNetPlans__${CONFIG}/predictions"

nnUNetv2_predict \
    -i ${nnUNet_raw}/Dataset${DATASET}_HyperBody/imagesTs \
    -o ${PRED_DIR} \
    -d ${DATASET} -c ${CONFIG} -tr ${TRAINER} -f ${FOLD}

# ========== Evaluation ==========
echo ""
echo "[$(date)] Running evaluation..."

python evaluate_nnunet_predictions.py \
    --pred_dir ${PRED_DIR}/ \
    --gt_dir ${nnUNet_raw}/Dataset${DATASET}_HyperBody/labelsTs/ \
    --output_dir evaluation_results/ \
    --npz_dir Dataset/voxel_data/ \
    --original_space

echo ""
echo "=========================================="
echo "All done! End time: $(date)"
echo "=========================================="
