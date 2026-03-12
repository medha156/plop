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
import pandas

# Local modules
from dataset import *
from model import *

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args):
    # hpt = hypertune.HyperTune()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("pairs.csv")
    total = len(df)
    #client = bigquery.Client(project=args.project_id)
    #query = f"""
    #SELECT COUNT(*) as n
    #FROM `{args.project_id}.{args.dataset_id}.binding_cleaned_clustered`
    #WHERE polymerid IS NOT NULL
    #"""
    #total = client.query(query).to_dataframe().iloc[0]['n']

    indices = list(range(total))
    random.seed(42)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    if args.downsample is not None:
        train_idx = random.sample(train_idx, min(args.downsample, len(train_idx)))
    
    train_dataset = BindingDBDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        # protein_urls=args.protein_dir,
        ligand_urls=args.ligand_dir,
        split_indices=train_idx,        # <-- filters metadata + protein cache
    )
    test_dataset = BindingDBDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        # protein_urls=args.protein_dir,
        ligand_urls=args.ligand_dir,
        split_indices=test_idx,         # <-- filters metadata + protein cache
    )
    BindingDBDataset.cache_proteins(args.protein_dir, train_dataset, test_dataset)

    train_loader = DataLoader(
        train_dataset.dataset,          # .dataset is the wds pipeline
        batch_size=args.batch_size,
        collate_fn=binding_collate,
        num_workers=16,                  # wds supports multi-worker streaming
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset.dataset,
        batch_size=args.batch_size,
        collate_fn=binding_collate,
        num_workers=16,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=4
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
        dropout_rate=args.dropout_rate,
        pooling=args.pool
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
            if ligand_batch is None:
                logger.warning(f"Batch {batch_idx} contains no valid samples after collate filtering. Skipping.")
                continue

            ligand_batch = ligand_batch.to(device)
            proteins = proteins.to(device)
            protein_mask = protein_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            mask = (labels != -1.0).float()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds, valid_mask = model(ligand_batch, proteins, protein_mask)

                raw_loss = criterion(preds, labels)

                full_mask = mask * valid_mask.float().unsqueeze(1)

                masked_loss_per_element = raw_loss * full_mask
                column_sums = masked_loss_per_element.sum(dim=0)
                column_counts = full_mask.sum(dim=0).clamp(min=1) # Avoid division by zero
    
                mean_loss_per_type = column_sums / column_counts
    
                # Final loss is the average of the three types
                # This ensures Ki, Kd, and IC50 contribute 1/3 each to the gradient
                loss = mean_loss_per_type.mean()
            
            if not valid_mask.all().item():
                logger.warning(f"Training batch {batch_idx} contains {(~valid_mask).sum().item()} invalid samples after attention pooling. This may indicate an issue with the data or model architecture.")

            loss.backward()
            optimizer.step()

            train_sum_loss += column_sums.detach()
            train_count += column_counts.detach()

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Train Epoch: {epoch} Batch: {batch_idx} x {args.batch_size} Loss: {loss.item():.6f}"
                )
        
        # VALIDATE
        model.eval()
        test_sum_loss = torch.zeros(3, device=device)
        test_count = torch.zeros(3, device=device)
        
        # torch.no_grad() prevents memory buildup and speeds up the loop
        with torch.no_grad():
            for batch_idx, (ligand_batch, proteins, protein_mask, labels) in enumerate(test_loader):
                if ligand_batch is None:
                    logger.warning(f"Validation batch {batch_idx} contains no valid samples after collate filtering. Skipping.")
                    continue

                ligand_batch = ligand_batch.to(device)
                proteins = proteins.to(device)
                protein_mask = protein_mask.to(device)
                labels = labels.to(device)

                mask = (labels != -1.0).float()

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds, valid_mask = model(ligand_batch, proteins, protein_mask)

                    raw_loss = criterion(preds, labels)

                    masked_loss_per_element = raw_loss * mask * valid_mask.float().unsqueeze(1)
                    column_sums = masked_loss_per_element.sum(dim=0)
                    column_counts = mask.sum(dim=0).clamp(min=1) # Avoid division by zero

                if not valid_mask.all().item():
                    logger.warning(f"Validation batch {batch_idx} contains {(~valid_mask).sum().item()} invalid samples after attention pooling. This may indicate an issue with the data or model architecture.")

                test_sum_loss += column_sums.detach()
                test_count += column_counts.detach()
                
                if batch_idx % log_interval == 0:
                    interim_test_loss = (test_sum_loss / test_count.clamp(min=1)).mean().item()
                    logger.info(
                        f"Test Epoch: {epoch} Batch: {batch_idx} x {args.batch_size} Interim Test Loss: {interim_test_loss:.6f}"
                    )
        
        train_loss_column = train_sum_loss / train_count
        train_loss_total = train_loss_column.mean()
        test_loss_column = test_sum_loss / test_count
        test_loss_total = test_loss_column.mean()

        epoch_duration = time.time() - start_time
        duration_str = str(timedelta(seconds=int(epoch_duration)))

        summary = f"""
{'='*40}
Epoch {epoch:03d} Summary | Time: {duration_str}
{'-'*40}
TRAIN | Total: {train_loss_total:.4f} | Ki: {train_loss_column[0]:.4f} | Kd: {train_loss_column[1]:.4f} | IC50: {train_loss_column[2]:.4f}
TEST | Total: {test_loss_total:.4f} | Ki: {test_loss_column[0]:.4f} | Kd: {test_loss_column[1]:.4f} | IC50: {test_loss_column[2]:.4f}
{'='*40}
"""
        logger.info(summary)

        # hpt.report_hyperparameter_tuning_metric(
        #     hyperparameter_metric_tag='val_loss',
        #     metric_value=valid_loss_total.item(), 
        #     global_step=epoch
        # )

        # Save to GCS-mounted directory
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"model_e{epoch}.pt"))

if __name__ == "__main__":
    print("RUNNING TRAIN FILE VERSION UPDATED")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--protein_dir", type=str, required=True)
    parser.add_argument("--ligand_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32) # Tuned by HPO
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=7e-4) # Tuned by HPO
    parser.add_argument("--model_dir", type=str, default=os.environ.get("AIP_MODEL_DIR", "./model"))
    parser.add_argument("--node_embed", type=int, default=64) # Tuned by HPO
    parser.add_argument("--edge_embed", type=int, default=64) # Tuned by HPO
    parser.add_argument("--gnn_layers", type=int, default=5) # Tuned by HPO
    parser.add_argument("--atn_layers", type=int, default=2) # Tuned by HPO
    parser.add_argument("--mlp_layers", type=int, default=2) # Tuned by HPO
    parser.add_argument("--atn_protein_heads", type=int, default=16) # Tuned by HPO
    parser.add_argument("--atn_ligand_heads", type=int, default=8) # Tuned by HPO
    parser.add_argument("--dropout_rate", type=float, default=0.1) # Tuned by HPO
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--pool", choices=["max", "mean", "sum"], default="max")

    args = parser.parse_args()
    train(args)
