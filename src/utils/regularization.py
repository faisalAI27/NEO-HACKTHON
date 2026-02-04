"""
Strong regularization techniques for small sample size survival prediction.

When training deep learning models on small survival datasets (~70-100 patients),
strong regularization is critical to prevent overfitting:
1. Weight decay (L2 regularization)
2. Dropout (already in model)
3. Data augmentation
4. Label smoothing for survival
5. Mixup/CutMix adapted for survival
6. Gradient clipping
7. Early stopping
8. Ensemble methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass


@dataclass
class RegularizationConfig:
    """Configuration for regularization techniques."""
    # Weight decay
    weight_decay: float = 0.01
    encoder_weight_decay: float = 0.001  # Lower for pretrained encoders
    
    # Dropout
    dropout_rate: float = 0.3
    attention_dropout: float = 0.2
    
    # Data augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_modality_dropout: bool = True
    modality_dropout_rate: float = 0.1
    
    # Label smoothing (time-aware)
    use_label_smoothing: bool = True
    label_smooth_sigma: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Feature noise
    use_feature_noise: bool = True
    feature_noise_std: float = 0.05
    
    # Stochastic depth
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1


class RegularizationModule:
    """
    Comprehensive regularization for survival prediction models.
    
    Provides multiple regularization techniques that can be applied
    during training to prevent overfitting on small datasets.
    
    Example:
        >>> reg = RegularizationModule(RegularizationConfig())
        >>> batch = reg.augment_batch(batch, training=True)
        >>> loss = reg.compute_regularized_loss(model, batch, criterion)
    """
    
    def __init__(self, config: RegularizationConfig):
        """
        Initialize regularization module.
        
        Args:
            config: RegularizationConfig with regularization parameters
        """
        self.config = config
    
    def mixup_data(
        self,
        batch: Dict[str, torch.Tensor],
        alpha: float = 0.2
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply Mixup augmentation adapted for survival data.
        
        For survival prediction, we need to be careful:
        - Mix features from two patients
        - Mix survival times proportionally
        - Keep event indicator for the patient with shorter time (more informative)
        
        Args:
            batch: Batch dictionary with features and targets
            alpha: Mixup interpolation strength
            
        Returns:
            Tuple of (mixed batch, lambda values)
        """
        if alpha <= 0:
            return batch, torch.ones(batch['time'].shape[0], device=batch['time'].device)
        
        batch_size = batch['time'].shape[0]
        device = batch['time'].device
        
        # Sample mixing coefficients
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        
        # Random permutation for mixing
        indices = torch.randperm(batch_size, device=device)
        
        mixed_batch = {}
        
        for key, value in batch.items():
            if key in ['time', 'event', 'patient_id', 'case_id']:
                continue
            
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                # Mix features
                lam_expanded = lam.view(-1, *([1] * (value.dim() - 1)))
                mixed_batch[key] = lam_expanded * value + (1 - lam_expanded) * value[indices]
            else:
                mixed_batch[key] = value
        
        # Handle survival targets carefully
        time1 = batch['time']
        time2 = batch['time'][indices]
        event1 = batch['event']
        event2 = batch['event'][indices]
        
        # For mixed samples, use weighted combination of times
        # and keep event indicator from shorter-lived patient
        mixed_batch['time'] = lam * time1 + (1 - lam) * time2
        
        # Event indicator: if either had event, mixed sample has event
        # Weight by proximity to each original sample
        mixed_batch['event'] = torch.where(
            lam > 0.5,
            event1,
            event2
        )
        
        # Alternative: Use event from patient with shorter time (more informative)
        shorter_idx = time1 < time2
        mixed_batch['event'] = torch.where(
            shorter_idx,
            event1,
            event2
        )
        
        return mixed_batch, lam
    
    def add_feature_noise(
        self,
        batch: Dict[str, torch.Tensor],
        modality_keys: List[str],
        noise_std: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to features for augmentation.
        
        Args:
            batch: Batch dictionary
            modality_keys: Keys of modalities to add noise to
            noise_std: Standard deviation of noise (relative to feature std)
            
        Returns:
            Batch with noisy features
        """
        noise_std = noise_std or self.config.feature_noise_std
        
        noisy_batch = batch.copy()
        
        for key in modality_keys:
            if key not in batch:
                continue
            
            tensor = batch[key]
            
            # Scale noise by feature standard deviation
            std = tensor.std(dim=0, keepdim=True) + 1e-8
            noise = torch.randn_like(tensor) * std * noise_std
            
            noisy_batch[key] = tensor + noise
        
        return noisy_batch
    
    def apply_modality_dropout(
        self,
        batch: Dict[str, torch.Tensor],
        modality_keys: List[str],
        dropout_rate: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Randomly zero out entire modalities.
        
        Args:
            batch: Batch dictionary
            modality_keys: Keys of modalities to potentially dropout
            dropout_rate: Probability of dropping each modality
            
        Returns:
            Batch with some modalities zeroed
        """
        dropout_rate = dropout_rate or self.config.modality_dropout_rate
        
        dropped_batch = batch.copy()
        batch_size = next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device
        
        for key in modality_keys:
            if key not in batch:
                continue
            
            # Random dropout mask per sample
            keep_mask = torch.rand(batch_size, device=device) > dropout_rate
            
            # Apply mask
            tensor = batch[key]
            keep_mask = keep_mask.view(-1, *([1] * (tensor.dim() - 1)))
            dropped_batch[key] = tensor * keep_mask.float()
        
        return dropped_batch
    
    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        modality_keys: List[str],
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all configured augmentations.
        
        Args:
            batch: Input batch
            modality_keys: Keys of modality tensors
            training: Whether in training mode
            
        Returns:
            Augmented batch
        """
        if not training:
            return batch
        
        # Mixup
        if self.config.use_mixup:
            batch, _ = self.mixup_data(batch, self.config.mixup_alpha)
        
        # Feature noise
        if self.config.use_feature_noise:
            batch = self.add_feature_noise(batch, modality_keys)
        
        # Modality dropout
        if self.config.use_modality_dropout:
            batch = self.apply_modality_dropout(batch, modality_keys)
        
        return batch


class SurvivalRegularizer(nn.Module):
    """
    Custom regularization losses for survival prediction.
    
    Adds regularization terms to the main Cox PH loss:
    1. L2 regularization on risk scores (shrinkage)
    2. Concordance penalty (soft C-index loss)
    3. Ranking margin loss
    4. Temporal smoothness penalty
    """
    
    def __init__(
        self,
        l2_weight: float = 0.01,
        ranking_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        margin: float = 0.1
    ):
        """
        Initialize the regularizer.
        
        Args:
            l2_weight: Weight for L2 regularization
            ranking_weight: Weight for ranking loss
            smoothness_weight: Weight for temporal smoothness
            margin: Margin for ranking loss
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.ranking_weight = ranking_weight
        self.smoothness_weight = smoothness_weight
        self.margin = margin
    
    def l2_regularization(self, risk_scores: torch.Tensor) -> torch.Tensor:
        """L2 regularization on risk scores to prevent extreme predictions."""
        return self.l2_weight * (risk_scores ** 2).mean()
    
    def ranking_loss(
        self,
        risk_scores: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Pairwise ranking loss for concordance.
        
        Penalizes pairs where patient with shorter survival time
        has lower risk score.
        """
        n = len(risk_scores)
        
        # Only consider comparable pairs (at least one event)
        loss = torch.tensor(0.0, device=risk_scores.device)
        n_pairs = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # i should have higher risk if time_i < time_j and event_i = 1
                if times[i] < times[j] and events[i] == 1:
                    # Risk_i should be > Risk_j
                    # Hinge loss with margin
                    pair_loss = F.relu(self.margin - (risk_scores[i] - risk_scores[j]))
                    loss = loss + pair_loss
                    n_pairs += 1
        
        if n_pairs > 0:
            loss = loss / n_pairs
        
        return self.ranking_weight * loss
    
    def temporal_smoothness(
        self,
        risk_scores: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage smooth risk score changes over time.
        
        Patients with similar survival times should have similar risk scores.
        """
        # Sort by time
        sorted_idx = torch.argsort(times)
        sorted_risks = risk_scores[sorted_idx]
        
        # Penalize large jumps between consecutive (time-sorted) patients
        diffs = (sorted_risks[1:] - sorted_risks[:-1]) ** 2
        
        return self.smoothness_weight * diffs.mean()
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total regularization loss.
        
        Args:
            risk_scores: Model risk score predictions
            times: Survival times
            events: Event indicators
            
        Returns:
            Total regularization loss
        """
        reg_loss = self.l2_regularization(risk_scores)
        
        # Ranking loss is expensive, sample if batch is large
        if len(risk_scores) <= 32:
            reg_loss = reg_loss + self.ranking_loss(risk_scores, times, events)
        
        reg_loss = reg_loss + self.temporal_smoothness(risk_scores, times)
        
        return reg_loss


class GradientClipping:
    """
    Utility for gradient clipping with monitoring.
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (default L2)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return original norm.
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Original gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type
        )
        
        self.grad_norms.append(total_norm.item())
        
        return total_norm.item()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get gradient norm statistics."""
        if not self.grad_norms:
            return {'mean': 0, 'max': 0, 'min': 0}
        
        return {
            'mean': np.mean(self.grad_norms),
            'max': np.max(self.grad_norms),
            'min': np.min(self.grad_norms),
            'clipped_ratio': np.mean([n > self.max_norm for n in self.grad_norms])
        }


class StochasticDepth(nn.Module):
    """
    Stochastic depth (layer dropout) for regularization.
    
    Randomly drops entire layers during training.
    """
    
    def __init__(self, drop_prob: float = 0.1, mode: str = 'row'):
        """
        Initialize stochastic depth.
        
        Args:
            drop_prob: Probability of dropping the layer
            mode: 'row' for per-sample, 'batch' for entire batch
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.mode = mode
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic depth.
        
        Args:
            x: Input tensor (to skip)
            residual: Residual tensor (to potentially drop)
            
        Returns:
            x + (potentially scaled/dropped) residual
        """
        if not self.training or self.drop_prob == 0:
            return x + residual
        
        if self.mode == 'row':
            # Per-sample dropout
            shape = [x.shape[0]] + [1] * (x.dim() - 1)
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor = random_tensor.floor_()
            return x + residual * random_tensor / keep_prob
        else:
            # Batch dropout
            if torch.rand(1).item() < self.drop_prob:
                return x
            else:
                return x + residual


def get_regularized_optimizer(
    model: nn.Module,
    config: RegularizationConfig,
    lr: float = 1e-4,
    encoder_lr_mult: float = 0.1
) -> torch.optim.Optimizer:
    """
    Create optimizer with differential weight decay.
    
    Applies different weight decay to:
    - Pretrained encoder parameters (lower)
    - Other parameters (higher)
    - Bias and LayerNorm (no decay)
    
    Args:
        model: Model to optimize
        config: Regularization configuration
        lr: Base learning rate
        encoder_lr_mult: Learning rate multiplier for encoders
        
    Returns:
        Configured optimizer
    """
    # Separate parameter groups
    decay_params = []
    no_decay_params = []
    encoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if encoder parameter
        is_encoder = any(enc in name.lower() for enc in ['encoder', 'uni', 'backbone'])
        
        # Check if should have no decay
        is_no_decay = any(nd in name.lower() for nd in ['bias', 'layernorm', 'layer_norm', 'ln'])
        
        if is_encoder:
            encoder_params.append(param)
        elif is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': config.weight_decay,
            'lr': lr
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'lr': lr
        },
        {
            'params': encoder_params,
            'weight_decay': config.encoder_weight_decay,
            'lr': lr * encoder_lr_mult
        }
    ]
    
    # Filter empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    optimizer = torch.optim.AdamW(param_groups)
    
    return optimizer


class EarlyStopping:
    """
    Early stopping with patience and model checkpointing.
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = 'max',
        restore_best: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric optimization direction
            restore_best: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(
        self,
        score: float,
        model: nn.Module
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            model: Model to potentially checkpoint
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return False
        
        # Check for improvement
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best:
                    self._restore_checkpoint(model)
                return True
        
        return False
    
    def _save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
    
    def _restore_checkpoint(self, model: nn.Module):
        """Restore best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
