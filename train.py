# Standard library
import os
import time
import random
import argparse
import subprocess
from datetime import timedelta

# Third-party
import torch
import pandas as pd
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from google.cloud import bigquery
from torch_geometric.data import Batch
import hypertune
import logging
import re

# Local modules
from dataset import *
from model import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    hpt = hypertune.HyperTune()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    client = bigquery.Client(project=args.project_id)
    query = f"""
    SELECT COUNT(*) as n
    FROM `{args.project_id}.{args.dataset_id}.binding_cleaned_clustered`
    WHERE polymerid IS NOT NULL
    """
    total = client.query(query).to_dataframe().iloc[0]['n']

    indices = list(range(total))
    random.seed(42)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    if args.downsample is not None:
        train_idx = random.sample(train_idx, min(args.downsample, len(train_idx)))
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    if args.downsample is not None:
        valid_idx = random.sample(valid_idx, min(args.downsample, len(valid_idx)))

    train_dataset = BindingDBDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        protein_urls=args.protein_dir,
        ligand_urls=args.ligand_dir,
        batch_size=args.batch_size,
        split_indices=train_idx,        # <-- filters metadata + protein cache
    )
    valid_dataset = BindingDBDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        protein_urls=args.protein_dir,
        ligand_urls=args.ligand_dir,
        batch_size=args.batch_size,
        split_indices=valid_idx,
    )

    train_loader = DataLoader(
        train_dataset.dataset,          # .dataset is the wds pipeline
        batch_size=args.batch_size,
        collate_fn=binding_collate,
        num_workers=16,                  # wds supports multi-worker streaming
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_dataset.dataset,
        batch_size=args.batch_size,
        collate_fn=binding_collate,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    model = BasicModel(
        ligand_node_in=9,
        ligand_edge_in=3,
        ligand_node_embed=args.node_embed,
        ligand_edge_embed=args.edge_embed,
        protein_embed=480,
        gnn_num_layers=args.gnn_layers,
        attn_num_layers=args.atn_layers,
        mlp_num_layers=args.mlp_layers,
        num_heads_protein=args.atn_protein_heads,
        num_heads_ligand=args.atn_ligand_heads,
        out=3,
        dropout_rate=args.dropout_rate
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = MSELoss(reduction='none') # Important for manual masking

    log_interval = 50

    for epoch in range(args.epochs):
        start_time = time.time()  # Start the clock
        # TRAIN
        model.train()
        train_sum_loss = torch.zeros(3, device=device)
        train_count = torch.zeros(3, device=device)

        for batch_idx, (ligand_batch, proteins, protein_mask, labels) in enumerate(train_loader):
            ligand_batch = ligand_batch.to(device)
            ligand_batch.edge_attr = ligand_batch.edge_attr.float()  # Ensure edge attributes are float
            proteins = proteins.to(device)
            protein_mask = protein_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            mask = (labels != -1.0).float()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds, valid = model(ligand_batch, proteins, protein_mask)

                raw_loss = criterion(preds, labels)

                masked_loss_per_element = raw_loss * mask * valid.float()
                column_sums = masked_loss_per_element.sum(dim=0)
                column_counts = mask.sum(dim=0).clamp(min=1) # Avoid division by zero
    
                mean_loss_per_type = column_sums / column_counts
    
                # Final loss is the average of the three types
                # This ensures Ki, Kd, and IC50 contribute 1/3 each to the gradient
                loss = mean_loss_per_type.mean()
            
            if not valid.all().item():
                logger.warning(f"Batch {batch_idx} contains {(~valid).sum().item()} invalid samples after attention pooling. This may indicate an issue with the data or model architecture.")

            loss.backward()
            optimizer.step()

            train_sum_loss += column_sums.detach()
            train_count += column_counts.detach()

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Train Epoch: {epoch} [{batch_idx} / {len(train_loader)}] Loss: {loss.item():.6f}"
                )
        
        # VALIDATE
        model.eval()
        valid_sum_loss = torch.zeros(3, device=device)
        valid_count = torch.zeros(3, device=device)
        
        # torch.no_grad() prevents memory buildup and speeds up the loop
        with torch.no_grad():
            for ligand_batch, proteins, protein_mask, labels in valid_loader:
                ligand_batch = ligand_batch.to(device)
                proteins = proteins.to(device)
                protein_mask = protein_mask.to(device)
                labels = labels.to(device)

                mask = (labels != -1.0).float()

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds = model(ligand_batch, proteins, protein_mask)

                    raw_loss = criterion(preds, labels)

                    masked_loss_per_element = raw_loss * mask
                    column_sums = masked_loss_per_element.sum(dim=0)
                    column_counts = mask.sum(dim=0).clamp(min=1) # Avoid division by zero

                valid_sum_loss += column_sums.detach()
                valid_count += column_counts.detach()
        
        train_loss_column = train_sum_loss / train_count
        train_loss_total = train_loss_column.mean()
        valid_loss_column = valid_sum_loss / valid_count
        valid_loss_total = valid_loss_column.mean()

        epoch_duration = time.time() - start_time
        duration_str = str(timedelta(seconds=int(epoch_duration)))

        summary = f"""
{'='*40}
Epoch {epoch:03d} Summary | Time: {duration_str}
{'-'*40}
TRAIN | Total: {train_loss_total:.4f} | Ki: {train_loss_column[0]:.4f} | Kd: {train_loss_column[1]:.4f} | IC50: {train_loss_column[2]:.4f}
VALID | Total: {valid_loss_total:.4f} | Ki: {valid_loss_column[0]:.4f} | Kd: {valid_loss_column[1]:.4f} | IC50: {valid_loss_column[2]:.4f}
{'='*40}
"""
        logger.info(summary)

        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_loss',
            metric_value=valid_loss_total.item(), 
            global_step=epoch
        )

    # Save to GCS-mounted directory
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))

if __name__ == "__main__":
    print("RUNNING TRAIN FILE VERSION UPDATED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--protein_dir", type=str, required=True) # Path to protein shards
    parser.add_argument("--ligand_dir", type=str, required=True) # Path to ligand shards
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("AIP_MODEL_DIR", "./model"))
    model_dir = os.environ.get("AIP_MODEL_DIR")
    parser.add_argument("--node_embed", type=int, default=256)
    parser.add_argument("--edge_embed", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=4)
    parser.add_argument("--atn_layers", type=int, default=2)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--atn_protein_heads", type=int, default=8)
    parser.add_argument("--atn_ligand_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--downsample", type=int, default=None)

    args = parser.parse_args()
    train(args)