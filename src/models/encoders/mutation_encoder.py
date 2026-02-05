import torch
import torch.nn as nn


class MutationEncoder(nn.Module):
    """
    Encoder for Mutation data.
    Combines a wide binary gene matrix with specific driver gene features and TMB (tumor mutation burden).
    """

    def __init__(self, num_genes, num_drivers=0, output_dim=256, dropout=0.3):
        """
        Args:
            num_genes: Total number of genes in the binary matrix.
            num_drivers: Number of driver genes (if handled separately).
                         If 0, driver branch is skipped.
            output_dim: Dimension of the final latent embedding.
        """
        super().__init__()

        # 1. Mutation Burden Encoder (Scalar -> Embedding)
        self.burden_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )

        # 2. Global Binary Matrix Encoder (Wide -> Narrow)
        # Using a deeper MLP to compress the sparse signal
        self.binary_encoder = nn.Sequential(
            nn.Linear(num_genes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.use_drivers = num_drivers > 0
        if self.use_drivers:
            # 3. Driver Gene Attention/Encoder
            # drivers are also binary (usually), but we might want to give them more capacity
            self.driver_encoder = nn.Sequential(
                nn.Linear(num_drivers, 128), nn.ReLU(), nn.Dropout(dropout)
            )
            fusion_dim = 64 + 512 + 128
        else:
            fusion_dim = 64 + 512

        # 4. Fusion & Output Projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x_binary, x_burden, x_drivers=None, return_attention=False):
        """
        Args:
            x_binary: (batch, num_genes)
            x_burden: (batch, 1)
            x_drivers: (batch, num_drivers) [Optional]
            return_attention: Ignored
        """
        # Encode burden
        h_burden = self.burden_encoder(x_burden)

        # Encode binary matrix
        h_binary = self.binary_encoder(x_binary)

        features = [h_burden, h_binary]

        if self.use_drivers and x_drivers is not None:
            h_drivers = self.driver_encoder(x_drivers)
            features.append(h_drivers)

        # Concatenate
        h_fused = torch.cat(features, dim=1)

        # Project
        out = self.output_projection(h_fused)

        if return_attention:
            return out, None

        return out
