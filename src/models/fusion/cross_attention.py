import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Multi-Head Cross-Modal Attention mechanism.
    Allows one modality (Query) to attend to another (Key/Value).
    """

    def __init__(self, dim_Q, dim_KV, dim_out, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads

        assert (
            self.head_dim * num_heads == dim_out
        ), "dim_out must be divisible by num_heads"

        self.W_q = nn.Linear(dim_Q, dim_out)
        self.W_k = nn.Linear(dim_KV, dim_out)
        self.W_v = nn.Linear(dim_KV, dim_out)

        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Args:
            query: (batch, n_query, dim_Q)
            key:   (batch, n_kv, dim_KV)
            value: (batch, n_kv, dim_KV)
            mask:  (batch, n_query, n_kv) or (batch, 1, 1, n_kv)
        """
        batch_size = query.size(0)

        # 1. Linear Projections
        # (batch, n, dim_out)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads
        # (batch, n, num_heads, head_dim) -> (batch, num_heads, n, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # (batch, num_heads, n_query, head_dim) @ (batch, num_heads, head_dim, n_kv)
        # -> (batch, num_heads, n_query, n_kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted Sum
        # (batch, num_heads, n_query, n_kv) @ (batch, num_heads, n_kv, head_dim)
        # -> (batch, num_heads, n_query, head_dim)
        out = torch.matmul(attn_weights, V)

        # 5. Concatenate heads
        # (batch, n_query, num_heads, head_dim)
        out = out.transpose(1, 2).contiguous()
        # (batch, n_query, dim_out)
        out = out.view(batch_size, -1, self.dim_out)

        # 6. Output Projection
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights

        return out
