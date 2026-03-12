#!/bin/bash

#SBATCH --job-name=ablate_attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/train_%j.log
#SBATCH --time=24:00:00

echo "Starting job"

source ~/plop/grah/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/home/users/medha156/.config/gcloud/application_default_credentials.json"
python train.py \
    --project_id plop-486317 \
    --dataset_id binding_data \
    --protein_dir "/home/users/medha156/plop/proteins_smol/proteins-smol/proteins-{000000..000011}.tar" \
    --ligand_dir "/home/users/medha156/plop/shardmmen/shardmmen/ligands-{000000..000032}.tar" \
    --model_dir model_-atn \
    --atn_layers 0

echo "done"
