import math

import torch
import webdataset as wds
from torch_geometric.data import Data, Batch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BindingDBDataset:
    def __init__(self, project_id, dataset_id, protein_urls, ligand_urls, split_indices=None):
        """
        Args:
            split_indices: A list or array of integers representing the rows 
                           from the BigQuery table to include in this split.
        """
        from google.cloud import bigquery
        self.client = bigquery.Client(project=project_id)
        
        # 1. Fetch Full Metadata
        query = f"""
            SELECT CAST(monomerid AS STRING) as mid, 
                   CAST(polymerid AS STRING) as pid, 
                   ki_value, kd_value, ic50_value
            FROM `{project_id}.{dataset_id}.binding_cleaned_clustered`
            WHERE polymerid IS NOT NULL
        """
        full_df = self.client.query(query).to_dataframe().fillna(-1)

        # 2. Apply the Split Indices
        if split_indices is not None:
            self.df = full_df.iloc[split_indices]
            logger.info(f"Split applied: {len(self.df)} samples.")
        else:
            self.df = full_df

        # 3. Create Allow-lists for the Stream
        # We only care about MIDs and PIDs present in THIS split
        self.active_mids = set(self.df['mid'].unique())
        self.active_pids = set(self.df['pid'].unique())
        
        # Mapping: mid -> list of (pid, labels) 
        # (Handling the case where one ligand appears multiple times in the split)
        self.lookup = {}
        for row in self.df.itertuples():
            if row.mid not in self.lookup:
                self.lookup[row.mid] = []
            def to_p(value):
                if value <= 0:
                    return -1
                return -math.log10(value) + 9
            labels = torch.tensor([to_p(row.ki_value), to_p(row.kd_value), to_p(row.ic50_value)], dtype=torch.float32)
            self.lookup[row.mid].append((row.pid, labels))

        # 4. Selective Protein Cache
        # We only store proteins that are actually used in this train or test split
        self.protein_cache = {}
        logger.info(f"Caching proteins for {len(self.active_pids)} unique targets...")
        prot_ds = wds.WebDataset(protein_urls, shardshuffle=False).decode()

        for sample in prot_ds:
            pid = sample["__key__"]
            if pid in self.active_pids:
                # Convert NumPy back to Torch
                tensor = torch.from_numpy(sample["pyd"]).float().share_memory_()  # Cache in shared memory for multi-worker access
                self.protein_cache[pid] = tensor
        
        # 5. The Ligand Stream
        # We use 'select' logic to drop ligands not in our split immediately
        self.dataset = (
            wds.WebDataset(ligand_urls, shardshuffle=7)
            .compose(wds.split_by_node)   # For multi-GPU
            .compose(wds.split_by_worker) # For num_workers > 0
            .decode()
            .compose(self._expand_pairs)
            .shuffle(20000)
            .with_epoch(len(self.df))
        )

    def __len__(self):
        return len(self.df)
    
    def _expand_pairs(self, source):
        for sample in source:
            mid = sample["__key__"]
            if mid not in self.active_mids:
                continue

            ligand_data = sample["pyd"]  # already a Data object, no conversion needed
            # ligand_data.edge_attr = ligand_data.edge_attr  # only this cast needed

            for pid, labels in self.lookup.get(mid, []):
                if pid in self.protein_cache:
                    yield (ligand_data, self.protein_cache[pid], labels)

def binding_collate(batch):
    """Flattens the list-of-lists and performs PyG batching + Protein padding."""
    # WebDataset returns a list of samples; our map returns a list of pairs per sample
    
    ligands, proteins, labels = zip(*batch)
    
    # Batch ligands (GNN)
    ligand_batch = Batch.from_data_list(ligands)

    # Pad proteins (Sequence Length x 480)
    max_len = max(p.shape[0] for p in proteins)
    embed_dim = proteins[0].shape[1]
    
    padded_proteins = torch.zeros(len(proteins), max_len, embed_dim)
    protein_mask = torch.zeros(len(proteins), max_len, dtype=torch.bool)

    for i, p in enumerate(proteins):
        padded_proteins[i, :p.shape[0]] = p
        protein_mask[i, :p.shape[0]] = 1

    # print("COLLATE FUNCTION:")
    # print("Ligand batch size:", ligand_batch.num_graphs)
    # print("Padded proteins shape:", padded_proteins.shape)
    # print("Protein mask shape:", protein_mask.shape)

    return ligand_batch, padded_proteins, protein_mask, torch.stack(labels)