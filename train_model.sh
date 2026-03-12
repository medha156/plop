#!/bin/bash

#SBATCH --job-name=plop_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

ulimit -n 65535
source ~/plop_env/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/home/users/saarlip/plop/key.json"
python train.py \
    --project_id plop-486317 \
    --dataset_id binding_data \
    --protein_dir "/home/users/saarlip/proteins_smol/proteins-{000000..000011}.tar" \
    --ligand_dir "/home/users/shardmmen/ligands-{000000..000032}.tar" \
    --model_dir ~/plop_model
