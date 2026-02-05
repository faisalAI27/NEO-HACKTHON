import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling layer for Multiple Instance Learning.
    References: Ilse et al., 2018, "Attention-based Deep Multiple Instance Learning"
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(GatedAttentionPooling, self).__init__()

        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())

        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, n_tiles, input_dim)
               Note: Batch size is usually 1 during inference for massive WSIs,
                     but we support batching if tiles are pre-sampled or padded.
        Returns:
            pooled: (batch, input_dim)
            A: Attention scores (batch, n_tiles)
        """
        # Calculate attention scores
        # V: (batch, n_tiles, hidden_dim)
        A_V = self.attention_V(x)
        # U: (batch, n_tiles, hidden_dim)
        A_U = self.attention_U(x)

        # Gated mechanism: element-wise mult
        # (batch, n_tiles, hidden_dim)
        gated = A_V * A_U

        # (batch, n_tiles, 1)
        A = self.attention_weights(gated)

        # Softmax over tiles dimension (dim=1)
        # (batch, n_tiles, 1)
        A = torch.softmax(A, dim=1)

        # Weighted sum: (batch, 1, n_tiles) @ (batch, n_tiles, input_dim) -> (batch, 1, input_dim)
        # Transpose A for matmul: (batch, 1, n_tiles)
        M = torch.matmul(A.transpose(1, 2), x)

        # Remove singleton dimension -> (batch, input_dim)
        M = M.squeeze(1)

        return M, A.squeeze(2)


class WSIEncoder(nn.Module):
    """
    Feature extractor for Whole Slide Images using Attention MIL.
    Encodes a bag of tile features (e.g., from UNI) into a single slide-level embedding.
    """

    def __init__(self, input_dim=1024, output_dim=256, atomic_dim=512, dropout=0.25):
        super(WSIEncoder, self).__init__()

        # 1. Feature Projector: Projects high-dim WSI features to lower dim space
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, atomic_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # 2. Attention Pooling
        self.attention_net = GatedAttentionPooling(input_dim=atomic_dim, hidden_dim=128)

        # 3. Output Projection
        self.output_projection = nn.Sequential(
            nn.Linear(atomic_dim, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, n_tiles, input_dim)
            return_attention: bool
        """
        # Project features
        # x: (batch, n_tiles, atomic_dim)
        h = self.feature_projector(x)

        # Pool
        # global_feat: (batch, atomic_dim)
        # attn_scores: (batch, n_tiles)
        global_feat, attn_scores = self.attention_net(h)

        # Output projection
        # out: (batch, output_dim)
        out = self.output_projection(global_feat)

        if return_attention:
            return out, attn_scores
        return out
