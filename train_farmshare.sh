#!/bin/bash

#SBATCH --job-name=tutorial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

python train.py \
    --project_id plop-486317 \
    --dataset_id binding_data \
    --protein_dir /gcs/protein-ligand-outcome-prediction/proteins_smol/proteins-{000000..000011}.tar \
    --ligand_dir /gcs/protein-ligand-outcome-prediction/shardmmen/ligands-{000000..000032}.tar \
    --model_dir model_main \
