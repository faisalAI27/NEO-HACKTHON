"""
Stratified sampling utilities for survival analysis with imbalanced censoring.

This module provides custom samplers that ensure balanced representation
of censored and uncensored patients in each batch and fold.
"""

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import List, Optional, Iterator, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
import warnings


class StratifiedSurvivalSampler(Sampler):
    """
    Sampler that ensures balanced censoring ratio in each batch.
    
    In survival analysis, heavily imbalanced censoring can lead to:
    - Biased risk score estimates
    - Poor calibration of survival curves
    - Degraded C-index performance
    
    This sampler stratifies by:
    1. Event status (censored vs uncensored)
    2. Optionally: time quartiles for more fine-grained stratification
    
    Example:
        >>> sampler = StratifiedSurvivalSampler(
        ...     events=events,
        ...     times=times,
        ...     batch_size=16,
        ...     stratify_by_time=True
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        events: np.ndarray,
        times: Optional[np.ndarray] = None,
        batch_size: int = 16,
        stratify_by_time: bool = False,
        n_time_bins: int = 4,
        oversample_events: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Initialize the stratified sampler.
        
        Args:
            events: Array of event indicators (1=event, 0=censored)
            times: Array of survival times (optional, for time stratification)
            batch_size: Number of samples per batch
            stratify_by_time: Whether to additionally stratify by time quartiles
            n_time_bins: Number of time bins for stratification
            oversample_events: Oversample events if minority class
            drop_last: Drop last incomplete batch
            seed: Random seed for reproducibility
        """
        self.events = np.array(events)
        self.times = np.array(times) if times is not None else None
        self.batch_size = batch_size
        self.stratify_by_time = stratify_by_time
        self.n_time_bins = n_time_bins
        self.oversample_events = oversample_events
        self.drop_last = drop_last
        self.seed = seed
        
        self.n_samples = len(self.events)
        self._create_strata()
        
    def _create_strata(self):
        """Create stratification groups based on events and optionally time."""
        # Create strata labels
        if self.stratify_by_time and self.times is not None:
            # Bin times into quartiles
            time_bins = np.percentile(
                self.times,
                np.linspace(0, 100, self.n_time_bins + 1)[1:-1]
            )
            time_strata = np.digitize(self.times, time_bins)
            
            # Combine event and time strata
            # Each stratum is (event, time_bin) pair
            self.strata_labels = self.events * (self.n_time_bins + 1) + time_strata
        else:
            # Just stratify by event
            self.strata_labels = self.events
        
        # Get indices for each stratum
        unique_strata = np.unique(self.strata_labels)
        self.strata_indices = {
            s: np.where(self.strata_labels == s)[0]
            for s in unique_strata
        }
        
        # Calculate sampling weights
        strata_sizes = {s: len(idx) for s, idx in self.strata_indices.items()}
        total = sum(strata_sizes.values())
        
        # Target uniform distribution across strata
        target_per_stratum = total / len(unique_strata)
        
        self.sampling_weights = {
            s: target_per_stratum / size if size > 0 else 0
            for s, size in strata_sizes.items()
        }
        
        # Compute event/censored ratio
        n_events = self.events.sum()
        n_censored = len(self.events) - n_events
        self.event_ratio = n_events / len(self.events)
        
        if self.oversample_events and self.event_ratio < 0.3:
            # Boost weight for event strata
            for s in self.strata_indices:
                # Strata with events have event indicator contribution
                is_event_stratum = any(self.events[i] == 1 for i in self.strata_indices[s][:5])
                if is_event_stratum:
                    self.sampling_weights[s] *= 1.5
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with balanced strata representation."""
        rng = np.random.RandomState(self.seed)
        
        # Shuffle indices within each stratum
        shuffled_strata = {
            s: rng.permutation(idx).tolist()
            for s, idx in self.strata_indices.items()
        }
        
        # Compute how many samples to draw from each stratum per batch
        strata_weights = np.array([
            self.sampling_weights[s] for s in sorted(shuffled_strata.keys())
        ])
        strata_weights = strata_weights / strata_weights.sum()
        
        samples_per_stratum = (strata_weights * self.batch_size).astype(int)
        # Ensure at least one sample from each stratum if possible
        samples_per_stratum = np.maximum(samples_per_stratum, 1)
        
        # Adjust to match batch size
        while samples_per_stratum.sum() > self.batch_size:
            max_idx = samples_per_stratum.argmax()
            samples_per_stratum[max_idx] -= 1
        while samples_per_stratum.sum() < self.batch_size:
            min_idx = samples_per_stratum.argmin()
            samples_per_stratum[min_idx] += 1
        
        # Generate batches
        batches = []
        strata_keys = sorted(shuffled_strata.keys())
        strata_pointers = {s: 0 for s in strata_keys}
        
        # Calculate number of complete iterations
        min_strata_len = min(len(idx) for idx in shuffled_strata.values())
        n_batches = len(self.events) // self.batch_size
        
        for batch_idx in range(n_batches):
            batch = []
            
            for stratum_idx, stratum in enumerate(strata_keys):
                n_samples = samples_per_stratum[stratum_idx]
                indices = shuffled_strata[stratum]
                ptr = strata_pointers[stratum]
                
                for _ in range(n_samples):
                    if ptr >= len(indices):
                        # Reshuffle and restart if exhausted
                        shuffled_strata[stratum] = rng.permutation(
                            self.strata_indices[stratum]
                        ).tolist()
                        indices = shuffled_strata[stratum]
                        ptr = 0
                    
                    batch.append(indices[ptr])
                    ptr += 1
                
                strata_pointers[stratum] = ptr
            
            # Shuffle batch
            rng.shuffle(batch)
            batches.append(batch)
        
        # Handle remaining samples
        if not self.drop_last:
            remaining = []
            for stratum in strata_keys:
                ptr = strata_pointers[stratum]
                remaining.extend(shuffled_strata[stratum][ptr:])
            
            if remaining:
                rng.shuffle(remaining)
                batches.append(remaining)
        
        return iter(batches)
    
    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class CensoringAwareBatchSampler(Sampler):
    """
    Batch sampler ensuring minimum events per batch for valid loss computation.
    
    Cox PH loss requires at least one event per batch to compute the risk set.
    This sampler guarantees a minimum number of events in each batch.
    
    Example:
        >>> sampler = CensoringAwareBatchSampler(
        ...     events=events,
        ...     batch_size=16,
        ...     min_events_per_batch=3
        ... )
    """
    
    def __init__(
        self,
        events: np.ndarray,
        batch_size: int = 16,
        min_events_per_batch: int = 2,
        seed: int = 42
    ):
        """
        Initialize the censoring-aware sampler.
        
        Args:
            events: Array of event indicators (1=event, 0=censored)
            batch_size: Number of samples per batch
            min_events_per_batch: Minimum events required per batch
            seed: Random seed
        """
        self.events = np.array(events)
        self.batch_size = batch_size
        self.min_events_per_batch = min_events_per_batch
        self.seed = seed
        
        self.event_indices = np.where(self.events == 1)[0]
        self.censored_indices = np.where(self.events == 0)[0]
        
        n_events = len(self.event_indices)
        if n_events < min_events_per_batch:
            warnings.warn(
                f"Dataset has only {n_events} events, "
                f"but min_events_per_batch is {min_events_per_batch}. "
                "Reducing min_events_per_batch to match."
            )
            self.min_events_per_batch = max(1, n_events)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with guaranteed minimum events."""
        rng = np.random.RandomState(self.seed)
        
        # Shuffle both pools
        event_pool = rng.permutation(self.event_indices).tolist()
        censored_pool = rng.permutation(self.censored_indices).tolist()
        
        n_censored_per_batch = self.batch_size - self.min_events_per_batch
        
        batches = []
        event_ptr = 0
        censored_ptr = 0
        
        while event_ptr < len(event_pool) or censored_ptr < len(censored_pool):
            batch = []
            
            # Add required events
            for _ in range(self.min_events_per_batch):
                if event_ptr >= len(event_pool):
                    # Recycle events with replacement
                    event_pool = rng.permutation(self.event_indices).tolist()
                    event_ptr = 0
                batch.append(event_pool[event_ptr])
                event_ptr += 1
            
            # Fill rest with censored
            for _ in range(n_censored_per_batch):
                if censored_ptr < len(censored_pool):
                    batch.append(censored_pool[censored_ptr])
                    censored_ptr += 1
                elif event_ptr < len(event_pool):
                    # No more censored, use events
                    batch.append(event_pool[event_ptr])
                    event_ptr += 1
            
            if len(batch) > 0:
                rng.shuffle(batch)
                batches.append(batch)
            
            # Stop when we've covered all censored and enough events
            if censored_ptr >= len(censored_pool) and event_ptr >= len(event_pool):
                break
        
        return iter(batches)
    
    def __len__(self) -> int:
        """Estimate number of batches."""
        return (len(self.events) + self.batch_size - 1) // self.batch_size


def create_stratified_cv_splits(
    patient_ids: List[str],
    events: np.ndarray,
    times: Optional[np.ndarray] = None,
    n_folds: int = 5,
    stratify_by_time: bool = False,
    n_time_bins: int = 3,
    seed: int = 42
) -> Dict[int, Dict[str, List[str]]]:
    """
    Create cross-validation splits stratified by event status and optionally time.
    
    This ensures each fold has similar:
    - Event/censoring ratio
    - (Optionally) Time distribution
    
    Args:
        patient_ids: List of patient identifiers
        events: Array of event indicators
        times: Array of survival times (optional)
        n_folds: Number of CV folds
        stratify_by_time: Also stratify by survival time
        n_time_bins: Number of time bins for stratification
        seed: Random seed
        
    Returns:
        Dictionary with fold indices mapping to train/val patient lists
    """
    patient_ids = np.array(patient_ids)
    events = np.array(events)
    
    # Create stratification labels
    if stratify_by_time and times is not None:
        times = np.array(times)
        # Create combined strata
        time_bins = np.percentile(times, np.linspace(0, 100, n_time_bins + 1)[1:-1])
        time_strata = np.digitize(times, time_bins)
        strata = events * (n_time_bins + 1) + time_strata
    else:
        strata = events
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, strata)):
        splits[fold] = {
            'train': patient_ids[train_idx].tolist(),
            'val': patient_ids[val_idx].tolist()
        }
        
        # Log statistics
        train_events = events[train_idx].sum()
        val_events = events[val_idx].sum()
        print(f"Fold {fold}: Train {len(train_idx)} ({train_events} events), "
              f"Val {len(val_idx)} ({val_events} events)")
    
    return splits


class TimeAwareStratifier:
    """
    Advanced stratifier that considers survival time distributions.
    
    Particularly useful for datasets with:
    - Long-tailed survival time distributions
    - Time-varying hazard patterns
    - Need for time-specific model evaluation
    """
    
    def __init__(
        self,
        quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
        handle_ties: str = 'random'
    ):
        """
        Initialize the stratifier.
        
        Args:
            quantiles: Quantile boundaries for time binning
            handle_ties: How to handle ties ('random', 'first', 'last')
        """
        self.quantiles = quantiles
        self.handle_ties = handle_ties
        self.bin_edges_ = None
    
    def fit(self, times: np.ndarray, events: np.ndarray):
        """
        Fit the stratifier to survival data.
        
        Args:
            times: Survival times
            events: Event indicators
        """
        # Use only event times to determine bins
        # This avoids bias from censoring distribution
        event_times = times[events == 1]
        
        if len(event_times) < len(self.quantiles) + 1:
            # Not enough events, use all times
            event_times = times
        
        self.bin_edges_ = np.percentile(event_times, [q * 100 for q in self.quantiles])
        return self
    
    def transform(self, times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """
        Create stratification labels.
        
        Args:
            times: Survival times
            events: Event indicators
            
        Returns:
            Array of stratum labels
        """
        if self.bin_edges_ is None:
            raise ValueError("Stratifier not fitted. Call fit() first.")
        
        time_bins = np.digitize(times, self.bin_edges_)
        
        # Combine with event status
        n_bins = len(self.quantiles) + 1
        strata = events * n_bins + time_bins
        
        return strata
    
    def fit_transform(self, times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(times, events)
        return self.transform(times, events)
