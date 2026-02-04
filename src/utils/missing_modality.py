"""
Missing modality handling for multi-modal survival prediction.

This module provides utilities for handling patients with missing modalities:
1. Modality masking during training
2. Mean/median imputation
3. Learned imputation networks
4. Dropout-based augmentation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    name: str
    dim: int
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    imputation_value: Optional[np.ndarray] = None
    missing_rate: float = 0.0  # Observed missing rate


@dataclass  
class ImputationConfig:
    """Configuration for imputation strategy."""
    strategy: str = 'mean'  # 'mean', 'median', 'zero', 'learned', 'dropout'
    dropout_rate: float = 0.1  # For dropout augmentation during training
    use_mask_embedding: bool = True  # Add learnable embedding for missing modality
    cross_modal_imputation: bool = False  # Use other modalities to impute


class MissingModalityHandler:
    """
    Handler for missing modality detection, masking, and imputation.
    
    Supports multiple strategies:
    - Zero imputation: Replace with zeros
    - Mean imputation: Replace with training set mean
    - Median imputation: Replace with training set median
    - Learned imputation: Use a neural network to predict missing values
    - Dropout augmentation: Randomly drop modalities during training
    
    Example:
        >>> handler = MissingModalityHandler(
        ...     modality_configs={
        ...         'rna': ModalityConfig(name='rna', dim=7923),
        ...         'meth': ModalityConfig(name='meth', dim=10057),
        ...     },
        ...     imputation_config=ImputationConfig(strategy='mean')
        ... )
        >>> handler.fit(train_data)
        >>> imputed_batch = handler.transform(batch, training=True)
    """
    
    def __init__(
        self,
        modality_configs: Dict[str, ModalityConfig],
        imputation_config: Optional[ImputationConfig] = None
    ):
        """
        Initialize the handler.
        
        Args:
            modality_configs: Dictionary of modality configurations
            imputation_config: Imputation strategy configuration
        """
        self.modality_configs = modality_configs
        self.config = imputation_config or ImputationConfig()
        self._fitted = False
        
        # Statistics computed from training data
        self._means = {}
        self._medians = {}
        self._stds = {}
    
    def fit(self, train_data: Dict[str, np.ndarray]) -> 'MissingModalityHandler':
        """
        Compute imputation statistics from training data.
        
        Args:
            train_data: Dictionary mapping modality names to data arrays
                       Shape: (n_samples, modality_dim)
                       
        Returns:
            Self for chaining
        """
        for modality, data in train_data.items():
            if modality not in self.modality_configs:
                continue
            
            # Handle missing values in data (represented as NaN or all zeros)
            valid_mask = ~np.isnan(data).any(axis=1)
            if np.sum(data, axis=1).min() == 0:
                # Also treat all-zero rows as missing
                valid_mask &= (np.abs(data).sum(axis=1) > 0)
            
            valid_data = data[valid_mask]
            
            if len(valid_data) > 0:
                self._means[modality] = np.mean(valid_data, axis=0)
                self._medians[modality] = np.median(valid_data, axis=0)
                self._stds[modality] = np.std(valid_data, axis=0) + 1e-8
            else:
                # No valid data, use zeros
                dim = self.modality_configs[modality].dim
                self._means[modality] = np.zeros(dim)
                self._medians[modality] = np.zeros(dim)
                self._stds[modality] = np.ones(dim)
            
            # Update config
            cfg = self.modality_configs[modality]
            cfg.mean = self._means[modality]
            cfg.std = self._stds[modality]
            cfg.missing_rate = 1 - (valid_mask.sum() / len(valid_mask))
            
            # Set imputation value based on strategy
            if self.config.strategy == 'mean':
                cfg.imputation_value = self._means[modality]
            elif self.config.strategy == 'median':
                cfg.imputation_value = self._medians[modality]
            elif self.config.strategy == 'zero':
                cfg.imputation_value = np.zeros(cfg.dim)
            else:
                cfg.imputation_value = self._means[modality]
        
        self._fitted = True
        return self
    
    def detect_missing(
        self,
        batch: Dict[str, torch.Tensor],
        threshold: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        """
        Detect which modalities are missing for each sample in batch.
        
        Args:
            batch: Dictionary of modality tensors (batch_size, modality_dim)
            threshold: Values below this are considered missing
            
        Returns:
            Dictionary of boolean masks (True = present, False = missing)
        """
        masks = {}
        
        for modality, tensor in batch.items():
            if modality not in self.modality_configs:
                continue
            
            # Check if entire modality is missing (all zeros or NaN)
            is_present = (tensor.abs().sum(dim=-1) > threshold)
            
            # Also check for NaN
            is_present &= ~torch.isnan(tensor).any(dim=-1)
            
            masks[modality] = is_present
        
        return masks
    
    def impute(
        self,
        batch: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Impute missing modalities.
        
        Args:
            batch: Dictionary of modality tensors
            masks: Dictionary of presence masks (True = present)
            
        Returns:
            Imputed batch
        """
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")
        
        imputed = {}
        device = next(iter(batch.values())).device
        
        for modality, tensor in batch.items():
            if modality not in self.modality_configs:
                imputed[modality] = tensor
                continue
            
            cfg = self.modality_configs[modality]
            mask = masks.get(modality, torch.ones(tensor.shape[0], dtype=torch.bool, device=device))
            
            if mask.all():
                # No missing values
                imputed[modality] = tensor
            else:
                # Create imputation tensor
                impute_values = torch.tensor(
                    cfg.imputation_value,
                    dtype=tensor.dtype,
                    device=device
                ).unsqueeze(0).expand(tensor.shape[0], -1)
                
                # Apply imputation where missing
                mask_expanded = mask.unsqueeze(-1).expand_as(tensor)
                imputed[modality] = torch.where(mask_expanded, tensor, impute_values)
        
        return imputed
    
    def apply_dropout(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply dropout augmentation during training.
        
        Randomly drops entire modalities to improve robustness
        to missing data at test time.
        
        Args:
            batch: Dictionary of modality tensors
            training: Whether in training mode
            
        Returns:
            Tuple of (augmented batch, dropout masks)
        """
        if not training or self.config.dropout_rate <= 0:
            # Return original batch with all-ones masks
            masks = {
                m: torch.ones(batch[m].shape[0], dtype=torch.bool, device=batch[m].device)
                for m in batch if m in self.modality_configs
            }
            return batch, masks
        
        batch_size = next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device
        
        augmented = {}
        masks = {}
        
        for modality, tensor in batch.items():
            if modality not in self.modality_configs:
                augmented[modality] = tensor
                continue
            
            # Random dropout mask
            keep_mask = torch.rand(batch_size, device=device) > self.config.dropout_rate
            masks[modality] = keep_mask
            
            if keep_mask.all():
                augmented[modality] = tensor
            else:
                # Impute dropped modalities
                cfg = self.modality_configs[modality]
                impute_values = torch.tensor(
                    cfg.imputation_value,
                    dtype=tensor.dtype,
                    device=device
                ).unsqueeze(0).expand(batch_size, -1)
                
                mask_expanded = keep_mask.unsqueeze(-1).expand_as(tensor)
                augmented[modality] = torch.where(mask_expanded, tensor, impute_values)
        
        return augmented, masks
    
    def transform(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full transformation pipeline: detect, impute, and optionally dropout.
        
        Args:
            batch: Dictionary of modality tensors
            training: Whether in training mode
            
        Returns:
            Transformed batch
        """
        # Detect missing
        masks = self.detect_missing(batch)
        
        # Impute missing
        imputed = self.impute(batch, masks)
        
        # Apply dropout augmentation during training
        if training and self.config.dropout_rate > 0:
            imputed, _ = self.apply_dropout(imputed, training=True)
        
        return imputed
    
    def get_modality_masks_tensor(
        self,
        masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert mask dictionary to stacked tensor for model input.
        
        Args:
            masks: Dictionary of modality masks
            
        Returns:
            Tensor of shape (batch_size, n_modalities)
        """
        modality_order = sorted(self.modality_configs.keys())
        
        mask_list = []
        for modality in modality_order:
            if modality in masks:
                mask_list.append(masks[modality].float().unsqueeze(-1))
            else:
                # Assume present if not in masks
                batch_size = next(iter(masks.values())).shape[0]
                device = next(iter(masks.values())).device
                mask_list.append(torch.ones(batch_size, 1, device=device))
        
        return torch.cat(mask_list, dim=-1)


class ModalityMaskCollator:
    """
    Custom collate function that handles missing modalities.
    
    Use with PyTorch DataLoader to properly batch samples
    with varying missing modalities.
    
    Example:
        >>> collator = ModalityMaskCollator(handler=handler)
        >>> loader = DataLoader(dataset, collate_fn=collator)
    """
    
    def __init__(
        self,
        handler: MissingModalityHandler,
        modality_keys: List[str] = None
    ):
        """
        Initialize the collator.
        
        Args:
            handler: MissingModalityHandler instance
            modality_keys: List of modality keys to handle
        """
        self.handler = handler
        self.modality_keys = modality_keys or list(handler.modality_configs.keys())
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Collated and imputed batch dictionary
        """
        collated = {}
        
        # Get all keys from first sample
        all_keys = batch[0].keys()
        
        for key in all_keys:
            if key in self.modality_keys:
                # Stack modality data
                tensors = [sample[key] for sample in batch]
                collated[key] = torch.stack(tensors)
            elif isinstance(batch[0][key], torch.Tensor):
                # Stack other tensors
                collated[key] = torch.stack([sample[key] for sample in batch])
            elif isinstance(batch[0][key], (int, float)):
                # Stack scalars
                collated[key] = torch.tensor([sample[key] for sample in batch])
            else:
                # Keep as list
                collated[key] = [sample[key] for sample in batch]
        
        # Detect missing modalities
        modality_batch = {k: collated[k] for k in self.modality_keys if k in collated}
        masks = self.handler.detect_missing(modality_batch)
        
        # Impute missing
        imputed = self.handler.impute(modality_batch, masks)
        
        # Update collated batch
        for k, v in imputed.items():
            collated[k] = v
        
        # Add mask tensor
        collated['modality_masks'] = self.handler.get_modality_masks_tensor(masks)
        
        return collated


class LearnedImputationNetwork(nn.Module):
    """
    Neural network for learning to impute missing modalities.
    
    Uses available modalities to predict missing ones.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        n_layers: int = 2
    ):
        """
        Initialize the imputation network.
        
        Args:
            modality_dims: Dictionary mapping modality names to dimensions
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modality_names = sorted(modality_dims.keys())
        
        # Encoder for each modality
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            for name, dim in modality_dims.items()
        })
        
        # Fusion network
        fusion_input = hidden_dim * len(modality_dims)
        fusion_layers = []
        for _ in range(n_layers):
            fusion_layers.extend([
                nn.Linear(fusion_input if len(fusion_layers) == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Decoder for each modality
        self.decoders = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim)
            for name, dim in modality_dims.items()
        })
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to impute missing modalities.
        
        Args:
            batch: Dictionary of modality tensors
            masks: Dictionary of presence masks (True = present)
            
        Returns:
            Dictionary with imputed modalities
        """
        # Encode available modalities
        encoded = []
        for name in self.modality_names:
            tensor = batch[name]
            mask = masks.get(name, torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device))
            
            # Encode
            enc = self.encoders[name](tensor)
            
            # Zero out missing modalities in encoding
            enc = enc * mask.unsqueeze(-1).float()
            encoded.append(enc)
        
        # Fuse
        fused = torch.cat(encoded, dim=-1)
        fused = self.fusion(fused)
        
        # Decode missing modalities
        imputed = {}
        for name in self.modality_names:
            tensor = batch[name]
            mask = masks.get(name, torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device))
            
            if mask.all():
                imputed[name] = tensor
            else:
                # Decode
                decoded = self.decoders[name](fused)
                
                # Replace missing with decoded
                mask_expanded = mask.unsqueeze(-1).expand_as(tensor)
                imputed[name] = torch.where(mask_expanded, tensor, decoded)
        
        return imputed
