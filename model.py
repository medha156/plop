import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch, dropout_edge

class BidirectionalAttentionLayer(nn.Module):
    def __init__(self, dim1, dim2, num_heads1, num_heads2, dropout_rate):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(dim1, num_heads = num_heads1, kdim=dim2, vdim=dim2, batch_first=True, dropout=min(dropout_rate, 0.1))
        self.attn2 = nn.MultiheadAttention(dim2, num_heads = num_heads2, kdim=dim1, vdim=dim1, batch_first=True, dropout=min(dropout_rate, 0.1))

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.ln1 = nn.LayerNorm(dim1)
        self.ln2 = nn.LayerNorm(dim2)
    
    def forward(self, x1, x2, mask1=None, mask2=None):
        # Cross-attention: x1 attends to x2

        out1, _ = self.attn1(x1, x2, x2, key_padding_mask=~mask2 if mask2 is not None else None)
        out1 = self.ln1(x1 + self.drop1(out1)) # Added residual + norm for stability
        
        # Cross-attention: x2 attends to x1
        out2, _ = self.attn2(x2, x1, x1, key_padding_mask=~mask1 if mask1 is not None else None)
        out2 = self.ln2(x2 + self.drop2(out2))

        return out1, out2

class GraphInteractionLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_rate):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_dim)
        )

        self.conv = gnn.GENConv(node_dim, node_dim, edge_dim=edge_dim, learn_t=True)
        self.msg_norm = gnn.MessageNorm(learn_scale=True)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index, edge_attr):
        # edge_index and edge_attr are already consistently masked by BasicModel
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr = self.edge_mlp(edge_input) + edge_attr
        
        h = self.conv(x, edge_index, edge_attr)
        h = self.msg_norm(x, h) # Standard GENConv practice
        x = self.node_norm(x + self.node_dropout(h)) # Residual + Norm

        return x, edge_attr

class BasicModel(nn.Module):
    def __init__(self, ligand_node_in, ligand_edge_in, ligand_node_embed, ligand_edge_embed,
                 protein_embed, 
                 gnn_num_layers,
                 attn_num_layers, num_heads_protein, num_heads_ligand,
                 mlp_num_layers, out,
                 dropout_rate):
        super().__init__()

        # Initial linear layer to move the graph into the expected dimensions
        self.node_init = torch.nn.Linear(ligand_node_in, ligand_node_embed)
        self.edge_init = torch.nn.Linear(ligand_edge_in, ligand_edge_embed)

        self.gnn = nn.ModuleList([GraphInteractionLayer(ligand_node_embed, ligand_edge_embed, dropout_rate) 
                                  for _ in range(gnn_num_layers)])

        self.attention = nn.ModuleList([BidirectionalAttentionLayer(protein_embed, ligand_node_embed, num_heads_protein, num_heads_ligand, dropout_rate) 
                                        for _ in range(attn_num_layers)])
        
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(ligand_node_embed + protein_embed, ligand_node_embed + protein_embed), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) for _ in range(mlp_num_layers - 1)])

        self.output_layer = nn.Linear(ligand_node_embed + protein_embed, out)
    
    def debug_check(self, tensor, name):
        if not torch.isfinite(tensor).all():
            print(f"!!! NAN DETECTED at: {name} !!!")
            print(f"Shape: {tensor.shape} | Min: {tensor.min()} | Max: {tensor.max()}")
            return True
        return False
    
    def masked_max_pool(self, x, mask, dim):
        """
        x:    (B, N, D)
        mask: (B, N) boolean
        """
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        x = torch.max(x, dim=dim).values
        x = torch.nan_to_num(x, neginf=0.0)
        return x

    def forward(self, ligand_data, protein, protein_mask):
        ligand = ligand_data.x
        edge_index = ligand_data.edge_index
        edge_attr = ligand_data.edge_attr
        batch = ligand_data.batch

        ligand = self.node_init(ligand)
        edge_attr = self.edge_init(edge_attr)
        self.debug_check(ligand, "Post-Node-Init")

        if self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.gnn[0].dropout_rate)
            edge_attr = edge_attr[edge_mask]

        for i, layer in enumerate(self.gnn):
            ligand, edge_attr = layer(ligand, edge_index, edge_attr)
            if self.debug_check(ligand, f"GNN Layer {i}"): break
        ligand, ligand_mask = to_dense_batch(ligand, batch)

        valid_mask = ligand_mask.any(dim=1) & protein_mask.any(dim=1)
        if not valid_mask.all().item():
            ligand_mask[~valid_mask, 0] = True
            protein_mask[~valid_mask, 0] = True

        for i, layer in enumerate(self.attention):
            protein, ligand = layer(protein, ligand, mask1=protein_mask, mask2=ligand_mask)
            if self.debug_check(protein, f"Attention Protein Layer {i}"): break
            if self.debug_check(ligand, f"Attention Ligand Layer {i}"): break
        
        ligand = self.masked_max_pool(ligand, ligand_mask, dim=1)
        protein = self.masked_max_pool(protein, protein_mask, dim=1)
        
        self.debug_check(ligand, "Post-Max-Pool Ligand")
        self.debug_check(protein, "Post-Max-Pool Protein")
        
        x = torch.cat([protein, ligand], dim=-1)

        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if self.debug_check(x, f"MLP Layer {i}"): break
        
        x = self.output_layer(x)

        valid_mask &= ~torch.isnan(x).any(dim=-1)

        return x, valid_mask