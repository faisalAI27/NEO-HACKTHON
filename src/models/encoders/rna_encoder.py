import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    """
    Simple MLP Encoder for vector data (RNA, Methylation, Clinical).
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.3, num_layers=2):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        curr_dim = hidden_dim
        # Hidden layers
        for _ in range(num_layers - 1):
            next_dim = curr_dim // 2
            layers.append(nn.Linear(curr_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = next_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(curr_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, return_attention=False):
        h = self.mlp(x)
        out = self.output_projection(h)
        if return_attention:
            return out, None
        return out

class PathwayAttentionEncoder(nn.Module):
    """
    Transformer-based encoder that treats genes as tokens.
    Learns interactions between genes (pathways).
    """
    def __init__(self, num_genes, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.num_genes = num_genes
        self.d_model = d_model
        
        # Embeddings
        # 1. Position embedding (Gene Identity) - learnable vector for each gene
        self.gene_pos_embed = nn.Parameter(torch.randn(1, num_genes, d_model))
        
        # 2. Value embedding: Project scalar gene expression value to d_model
        # We share this projection across all genes to learn "what does high/low expression mean generally"
        # Strategy: h_i = PosEmbed_i + Linear(val_i)
        self.val_proj = nn.Linear(1, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling (similar to MIL) to aggregate gene tokens into patient embedding
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Output Projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, num_genes)
        """
        # Ensure input has the correct number of genes
        if x.size(1) != self.num_genes:
            # Ideally we would check this, but dynamic batching might make this tricky if not careful.
            # Assuming x is correct shape [B, num_genes]
            pass
            
        # 1. Create tokens
        # x_expanded: (batch, num_genes, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Value embedding: (batch, num_genes, d_model)
        val_embed = self.val_proj(x_expanded)
        
        # Add position embedding (broadcasting over batch)
        # (batch, num_genes, d_model)
        h = val_embed + self.gene_pos_embed
        
        # 2. Transformer
        # (batch, num_genes, d_model)
        h = self.transformer(h)
        
        # 3. Attention Pooling
        # attn_scores: (batch, num_genes, 1)
        attn_logits = self.attention_net(h)
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # Weighted sum: (batch, num_genes, d_model) * (batch, num_genes, 1) -> sum over genes
        global_feat = torch.sum(h * attn_weights, dim=1)
        
        # 4. Output Project
        out = self.output_projection(global_feat)
        
        if return_attention:
            return out, attn_weights.squeeze(-1)
            
        return out
