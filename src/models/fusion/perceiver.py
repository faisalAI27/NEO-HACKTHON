import torch
import torch.nn as nn

from .cross_attention import CrossModalAttention


class PerceiverFusion(nn.Module):
    """
    Perceiver-based Fusion Module.
    Uses learnable latent queries to attend to multimodal inputs.
    Adapts Perceiver IO architecture: https://arxiv.org/abs/2107.14795
    """

    def __init__(
        self,
        modal_dims,
        num_latents=32,
        latent_dim=256,
        num_cross_heads=4,
        num_self_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        """
        Args:
            modal_dims: dict of input dimensions for each modality {'wsi': 256, 'rna': 256, ...}
            num_latents: number of latent query vectors
            latent_dim: dimension of latent vectors
            num_cross_heads: attention heads for cross-attention
            num_self_heads: attention heads for self-attention
            num_layers: number of self-attention processing layers
        """
        super().__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # 1. Learnable Latent Queries
        # (1, num_latents, latent_dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # 2. Cross-Attention Layers (ONE per modality or SHARED?)
        # Strategy: Concatenate all modalities into one big sequence?
        # OR: Attend to each modality separately and sum?
        # Simpler approach: Project all modalities to same dim, concat, then cross-attend.

        # We assume modalities are already encoded to some reasonable dimension/sequence.
        # But they might be different shapes?
        # If we just treat them as a bag of features:
        # We need to project them to a common key/value dimension if we want to concat them.
        # OR we let the cross attention handle different kv_dims.

        # Let's go with: Concatenate all features into a single sequence (batch, total_tokens, common_dim)
        # But wait, WSI is a vector (batch, 256), RNA is a vector (batch, 256).
        # Should we stack them? (batch, num_modalities, 256)
        # YES.

        # Input Projection to match latent_dim (which acts as KV dim)
        self.input_projections = nn.ModuleDict()
        for mod, dim in modal_dims.items():
            self.input_projections[mod] = nn.Linear(dim, latent_dim)

        # Cross Attention: Latents (Q) attend to Modalities (KV)
        self.cross_attn = CrossModalAttention(
            dim_Q=latent_dim,
            dim_KV=latent_dim,
            dim_out=latent_dim,
            num_heads=num_cross_heads,
            dropout=dropout,
        )
        self.cross_ln_q = nn.LayerNorm(latent_dim)
        self.cross_ln_kv = nn.LayerNorm(latent_dim)

        # 3. Self-Attention Processing Layers (Latents attend to Latents)
        self.self_layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = nn.ModuleDict(
                {
                    "attn": nn.MultiheadAttention(
                        embed_dim=latent_dim,
                        num_heads=num_self_heads,
                        batch_first=True,
                        dropout=dropout,
                    ),
                    "ln1": nn.LayerNorm(latent_dim),
                    "ff": nn.Sequential(
                        nn.Linear(latent_dim, latent_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(latent_dim * 2, latent_dim),
                    ),
                    "ln2": nn.LayerNorm(latent_dim),
                }
            )
            self.self_layers.append(layer)

        # 4. Output Pooling (Latents -> Single Vector)
        # Simple mean pooling or attention pooling
        self.output_pool = nn.Linear(
            latent_dim * num_latents, latent_dim
        )  # Flatten and project? Or just Mean?
        # Let's do Mean Pooling for robustness

        self.final_proj = nn.LayerNorm(latent_dim)

    def forward(self, modality_inputs, return_attention=False):
        """
        Args:
           modality_inputs: dict of tensors { 'wsi': (B, dim), 'rna': (B, dim), ... }
           return_attention: If True, return attention weights.
        """
        input_list = []
        batch_size = next(iter(modality_inputs.values())).size(0)

        # 1. Prepare Inputs
        # We need to track which token corresponds to which modality for visualization
        # modality_order is just keys order.

        for mod, x in modality_inputs.items():
            if mod in self.input_projections:
                # Project to common dim
                # x: (B, dim) -> (B, latent_dim)
                proj = self.input_projections[mod](x)

                # Add sequence dimension: (B, 1, latent_dim)
                proj = proj.unsqueeze(1)

                # Add modality encoding? (Optional, maybe later)

                input_list.append(proj)

        # Concat along sequence dim
        # (B, num_modalities, latent_dim)
        kv_input = torch.cat(input_list, dim=1)

        # 2. Cross-Attention
        # Query: Latents (B, num_latents, latent_dim)
        latents = self.latents.repeat(batch_size, 1, 1)

        q = self.cross_ln_q(latents)
        kv = self.cross_ln_kv(kv_input)

        # latents = latents + cross_attn(q, kv, kv)
        if return_attention:
            attn_out, attn_weights = self.cross_attn(q, kv, kv, return_attention=True)
        else:
            attn_out = self.cross_attn(q, kv, kv)

        latents = latents + attn_out

        # 3. Self-Attention Layers
        for layer in self.self_layers:
            # Pytorch MHA expects (B, S, E) if batch_first=True
            residual = latents
            latents_norm = layer["ln1"](latents)

            # (out, weights)
            attn_out, _ = layer["attn"](latents_norm, latents_norm, latents_norm)
            latents = residual + attn_out

            # Feed Forward
            residual = latents
            latents_norm = layer["ln2"](latents)
            ff_out = layer["ff"](latents_norm)
            latents = residual + ff_out

        # 4. Pooling
        # (B, num_latents, latent_dim) -> (B, latent_dim)
        # Global Average Pooling over query positions
        pooled = torch.mean(latents, dim=1)

        out = self.final_proj(pooled)

        if return_attention:
            return out, attn_weights

        return out
