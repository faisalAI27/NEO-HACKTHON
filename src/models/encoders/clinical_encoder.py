import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    """
    Encoder for Clinical data (Age, Gender, Stage, etc.).
    Input is a vector of continuous and one-hot encoded variables.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=256, dropout=0.1):
        super().__init__()
        
        # Clinical data is low-dimensional but high-value
        # We use a smaller MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, return_attention=False):
        """
        Args:
           x: (batch, input_dim)
           return_attention: Ignored for MLP.
        """
        h = self.encoder(x)
        out = self.output_projection(h)
        if return_attention:
            return out, None
        return out
