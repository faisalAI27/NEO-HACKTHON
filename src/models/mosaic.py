import torch
import torch.nn as nn
from src.models.encoders.wsi_encoder import WSIEncoder
from src.models.encoders.rna_encoder import PathwayAttentionEncoder
from src.models.encoders.methylation_encoder import MethylationEncoder
from src.models.encoders.mutation_encoder import MutationEncoder
from src.models.encoders.clinical_encoder import ClinicalEncoder
from src.models.fusion.perceiver import PerceiverFusion

class MOSAIC(nn.Module):
    """
    MOSAIC: Multi-Omics Survival Analysis with Interpretable Cross-modal attention.
    Integrates WSI, RNA, Methylation, Mutation, and Clinical data.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Encoders
        self.encoders = nn.ModuleDict()
        input_dims = config['data_dims'] # Dict with input shapes e.g. {'rna': 3000, 'wsi': 1024}
        enc_config = config.get('encoders', {})
        
        # WSI
        if 'wsi' in input_dims:
            self.encoders['wsi'] = WSIEncoder(
                input_dim=input_dims['wsi'],
                output_dim=enc_config.get('wsi_out_dim', 256)
            )
            
        # RNA
        if 'rna' in input_dims:
            self.encoders['rna'] = PathwayAttentionEncoder(
                num_genes=input_dims['rna'],
                output_dim=enc_config.get('rna_out_dim', 256)
            )
            
        # Methylation
        if 'methylation' in input_dims:
            self.encoders['methylation'] = MethylationEncoder(
                input_dim=input_dims['methylation'],
                output_dim=enc_config.get('meth_out_dim', 256)
            )
            
        # Mutation
        if 'mutations' in input_dims:
            self.encoders['mutations'] = MutationEncoder(
                num_genes=input_dims['mutations'],
                num_drivers=input_dims.get('drivers', 0),
                output_dim=enc_config.get('mut_out_dim', 256)
            )
            
        # Clinical
        if 'clinical' in input_dims:
            self.encoders['clinical'] = ClinicalEncoder(
                input_dim=input_dims['clinical'],
                output_dim=enc_config.get('clin_out_dim', 256)
            )
            
        # 2. Modality Embeddings (Optional but suggested for Perceiver)
        # We add a learnable vector to each modality embedding to identify it
        self.modality_embeddings = nn.ParameterDict()
        for mod in self.encoders.keys():
            dim = enc_config.get(f'{mod}_out_dim', 256) # Assuming standardized keys
            # Actually simpler: assume all encoders output 'projection_dim' or fusion handles it
            # Let's assume encoders output what we configured. 
            self.modality_embeddings[mod] = nn.Parameter(torch.randn(1, dim))

        # 3. Fusion Module
        # We need to tell Perceiver what the input dims are (which are the OUTPUT dims of the encoders)
        fusion_input_dims = {}
        for mod in self.encoders.keys():
             fusion_input_dims[mod] = enc_config.get(f'{mod}_out_dim', 256)
             
        self.fusion = PerceiverFusion(
            modal_dims=fusion_input_dims,
            num_latents=config.get('fusion', {}).get('num_latents', 32),
            latent_dim=config.get('fusion', {}).get('latent_dim', 256),
            num_layers=config.get('fusion', {}).get('num_layers', 2),
            dropout=config.get('fusion', {}).get('dropout', 0.2)
        )
        
        # 4. Survival Head
        # Predicts hazard ratio (or risk score)
        latent_dim = config.get('fusion', {}).get('latent_dim', 256)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1) # Single output: log risk
        )
        
    def forward(self, batch, return_attention: bool = False):
        """
        Args:
            batch: dict of tensors. Keys must match encoder keys.
                   e.g. {'wsi': tensor, 'rna': tensor, ...}
            return_attention: If True, returns (logits, attn_weights_dict)
        """
        encoded_modalities = {}
        intra_modal_attn = {}
        
        # 1. Encode available modalities
        for mod, encoder in self.encoders.items():
            if mod in batch:
                x = batch[mod]
                emb = None
                attn = None
                
                # Prepare args
                kwargs = {'return_attention': return_attention}
                
                if mod == 'mutations' and isinstance(x, (list, tuple)):
                    res = encoder(*x, **kwargs)
                elif mod == 'mutations' and isinstance(x, dict):
                    res = encoder(**x, **kwargs)
                else:
                    res = encoder(x, **kwargs)
                
                # Unpack result
                if return_attention:
                    emb, attn = res
                    if attn is not None:
                        intra_modal_attn[mod] = attn
                else:
                    emb = res
                
                # Add modality type embedding
                if mod in self.modality_embeddings:
                    emb = emb + self.modality_embeddings[mod]
                
                encoded_modalities[mod] = emb
        
        # 2. Fuse
        # fused: (Batch, latent_dim)
        if return_attention:
            fused, cross_attn_weights = self.fusion(encoded_modalities, return_attention=True)
        else:
            fused = self.fusion(encoded_modalities)
        
        # 3. Predict
        logits = self.head(fused)
        
        if return_attention:
            # Combine all attention info
            all_attn = {
                'cross_modal': cross_attn_weights,
                **intra_modal_attn
            }
            return logits, all_attn
            
        return logits
