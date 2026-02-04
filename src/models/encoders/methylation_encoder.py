import torch
import torch.nn as nn

class MethylationEncoder(nn.Module):
    """
    Encoder for Methylation data (M-values).
    Uses a bottleneck MLP architecture to compress high-dimensional methylation data.
    """
    def __init__(self, input_dim, hidden_dim=1024, bottleneck_dim=512, output_dim=256, dropout=0.3):
        super().__init__()
        
        # 1. MLP with bottleneck
        # Input -> Hidden -> Bottleneck
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2 (Bottleneck)
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3 (Further extraction)
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.BatchNorm1d(bottleneck_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
         
        # 2. Output Projection
        self.output_projection = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, input_dim) - M-values vector
            return_attention: Ignored
        """
        # Encode
        encoded = self.encoder(x)
        
        # Project
        out = self.output_projection(encoded)
        
        if return_attention:
            return out, None
        return out
