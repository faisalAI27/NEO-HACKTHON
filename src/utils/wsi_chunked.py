"""
Memory-efficient WSI processing with chunk-based feature extraction.

This module provides utilities for processing large whole-slide images
in memory-efficient chunks to prevent OOM errors during feature extraction.
"""

import os
import gc
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, List, Tuple, Generator, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ChunkConfig:
    """Configuration for chunked WSI processing."""
    chunk_size: int = 256  # Number of tiles per chunk
    max_tiles: int = 4000  # Maximum tiles per slide
    feature_dim: int = 1024  # Output feature dimension
    save_intermediate: bool = False  # Save intermediate chunks to disk
    temp_dir: Optional[Path] = None  # Directory for intermediate files
    memory_limit_gb: float = 4.0  # Approximate memory limit in GB
    prefetch_factor: int = 2  # DataLoader prefetch factor
    num_workers: int = 4  # DataLoader workers


class ChunkedWSIProcessor:
    """
    Memory-efficient WSI feature extraction using chunk-based processing.
    
    Instead of loading all tiles into memory at once, this processor:
    1. Extracts tile coordinates
    2. Processes tiles in chunks
    3. Aggregates features incrementally
    4. Optionally saves intermediate results
    
    Example:
        >>> processor = ChunkedWSIProcessor(
        ...     feature_extractor=uni_model,
        ...     config=ChunkConfig(chunk_size=256)
        ... )
        >>> features = processor.process_slide(slide_path, output_path)
    """
    
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        config: Optional[ChunkConfig] = None,
        device: str = 'cuda',
        transform: Optional[callable] = None
    ):
        """
        Initialize the chunked processor.
        
        Args:
            feature_extractor: Pre-trained model for feature extraction (e.g., UNI)
            config: ChunkConfig with processing parameters
            device: Device for inference ('cuda' or 'cpu')
            transform: Optional transform for tile preprocessing
        """
        self.feature_extractor = feature_extractor
        self.config = config or ChunkConfig()
        self.device = device
        self.transform = transform
        
        # Move model to device and set to eval mode
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()
        
        # Calculate effective chunk size based on memory limit
        self._adjust_chunk_size_for_memory()
        
        # Setup temp directory
        if self.config.save_intermediate:
            self.config.temp_dir = self.config.temp_dir or Path('/tmp/mosaic_chunks')
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _adjust_chunk_size_for_memory(self):
        """Adjust chunk size based on available GPU memory."""
        if self.device == 'cuda' and torch.cuda.is_available():
            # Get available GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem = gpu_mem - torch.cuda.memory_allocated()
            
            # Estimate memory per tile (approximate for ViT-L/16)
            # Input: 224x224x3 float32 + intermediate activations
            bytes_per_tile = 224 * 224 * 3 * 4 * 10  # ~5MB with activations
            
            # Calculate safe chunk size
            safe_mem = min(free_mem * 0.7, self.config.memory_limit_gb * 1e9)
            estimated_chunk_size = int(safe_mem / bytes_per_tile)
            
            # Use minimum of configured and estimated
            self.config.chunk_size = min(
                self.config.chunk_size,
                max(32, estimated_chunk_size)  # At least 32 tiles per chunk
            )
            
            print(f"Adjusted chunk size to {self.config.chunk_size} based on GPU memory")
    
    def _tile_generator(
        self,
        slide,
        coordinates: List[Tuple[int, int]],
        read_size: int
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields tiles one at a time.
        
        Args:
            slide: OpenSlide object
            coordinates: List of (x, y) coordinates
            read_size: Size of region to read at level 0
            
        Yields:
            Tuple of (index, tile_array)
        """
        from PIL import Image
        
        target_size = 224  # ViT input size
        
        for idx, (x, y) in enumerate(coordinates):
            try:
                # Read region from slide
                region = slide.read_region((x, y), 0, (read_size, read_size))
                region = region.convert('RGB')
                
                # Resize to target size
                region = region.resize((target_size, target_size), Image.BILINEAR)
                
                # Convert to numpy
                tile = np.array(region, dtype=np.uint8)
                
                yield idx, tile
                
            except Exception as e:
                print(f"Error reading tile at ({x}, {y}): {e}")
                continue
    
    def _process_chunk(
        self,
        tiles: List[np.ndarray],
        chunk_idx: int
    ) -> np.ndarray:
        """
        Process a single chunk of tiles through the feature extractor.
        
        Args:
            tiles: List of tile images as numpy arrays
            chunk_idx: Index of this chunk (for logging)
            
        Returns:
            Feature array of shape (n_tiles, feature_dim)
        """
        # Stack tiles into batch
        batch = np.stack(tiles, axis=0)  # (N, H, W, C)
        
        # Convert to tensor and normalize
        batch_tensor = torch.from_numpy(batch).float()
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
        batch_tensor = batch_tensor / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        batch_tensor = (batch_tensor - mean) / std
        
        # Apply custom transform if provided
        if self.transform is not None:
            batch_tensor = self.transform(batch_tensor)
        
        # Move to device
        batch_tensor = batch_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                features = self.feature_extractor(batch_tensor)
        
        # Move to CPU and convert to numpy
        features_np = features.cpu().numpy()
        
        # Clear GPU memory
        del batch_tensor, features
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return features_np
    
    def _save_chunk(
        self,
        features: np.ndarray,
        chunk_idx: int,
        slide_name: str
    ) -> Path:
        """Save intermediate chunk to disk."""
        chunk_path = self.config.temp_dir / f"{slide_name}_chunk_{chunk_idx:04d}.npy"
        np.save(chunk_path, features)
        return chunk_path
    
    def _load_and_merge_chunks(
        self,
        chunk_paths: List[Path]
    ) -> np.ndarray:
        """Load all chunk files and merge into single feature array."""
        all_features = []
        for path in chunk_paths:
            features = np.load(path)
            all_features.append(features)
            
            # Delete temp file
            path.unlink()
        
        return np.concatenate(all_features, axis=0)
    
    def process_slide(
        self,
        slide_path: Path,
        output_path: Path,
        coordinates: Optional[List[Tuple[int, int]]] = None,
        read_size: int = 512,
        force: bool = False
    ) -> np.ndarray:
        """
        Process entire slide with chunked feature extraction.
        
        Args:
            slide_path: Path to WSI file
            output_path: Path to save extracted features (HDF5)
            coordinates: Pre-computed tile coordinates (optional)
            read_size: Size of region to read at level 0
            force: Reprocess even if output exists
            
        Returns:
            Feature array of shape (n_tiles, feature_dim)
        """
        import openslide
        
        # Check if already processed
        if output_path.exists() and not force:
            print(f"Loading existing features from {output_path}")
            with h5py.File(output_path, 'r') as f:
                return f['features'][:]
        
        # Open slide
        slide = openslide.OpenSlide(str(slide_path))
        slide_name = slide_path.stem
        
        print(f"Processing {slide_name} with chunked extraction...")
        
        # Get coordinates if not provided
        if coordinates is None:
            from src.data.wsi_preprocessing import WSITiler
            tiler = WSITiler()
            mask, mask_downsample = tiler.get_tissue_mask(slide)
            coordinates, read_size = tiler.get_tile_coordinates(slide, mask, mask_downsample)
        
        n_tiles = len(coordinates)
        print(f"Processing {n_tiles} tiles in chunks of {self.config.chunk_size}")
        
        # Limit to max tiles
        if n_tiles > self.config.max_tiles:
            np.random.seed(42)
            indices = np.random.choice(n_tiles, self.config.max_tiles, replace=False)
            coordinates = [coordinates[i] for i in indices]
            n_tiles = len(coordinates)
        
        # Process in chunks
        all_features = []
        chunk_paths = []
        current_chunk = []
        chunk_idx = 0
        
        tile_gen = self._tile_generator(slide, coordinates, read_size)
        
        with tqdm(total=n_tiles, desc=f"Extracting features") as pbar:
            for idx, tile in tile_gen:
                current_chunk.append(tile)
                pbar.update(1)
                
                # Process when chunk is full
                if len(current_chunk) >= self.config.chunk_size:
                    features = self._process_chunk(current_chunk, chunk_idx)
                    
                    if self.config.save_intermediate:
                        chunk_path = self._save_chunk(features, chunk_idx, slide_name)
                        chunk_paths.append(chunk_path)
                    else:
                        all_features.append(features)
                    
                    current_chunk = []
                    chunk_idx += 1
                    
                    # Force garbage collection
                    gc.collect()
        
        # Process remaining tiles
        if current_chunk:
            features = self._process_chunk(current_chunk, chunk_idx)
            if self.config.save_intermediate:
                chunk_path = self._save_chunk(features, chunk_idx, slide_name)
                chunk_paths.append(chunk_path)
            else:
                all_features.append(features)
        
        # Merge all features
        if self.config.save_intermediate:
            final_features = self._load_and_merge_chunks(chunk_paths)
        else:
            final_features = np.concatenate(all_features, axis=0)
        
        # Save to HDF5
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'features',
                data=final_features,
                compression='gzip',
                compression_opts=4
            )
            f.attrs['n_tiles'] = final_features.shape[0]
            f.attrs['feature_dim'] = final_features.shape[1]
            f.attrs['slide_name'] = slide_name
        
        print(f"Saved features to {output_path}: shape {final_features.shape}")
        
        slide.close()
        return final_features
    
    def process_batch(
        self,
        slide_paths: List[Path],
        output_dir: Path,
        force: bool = False
    ) -> Dict[str, Path]:
        """
        Process multiple slides efficiently.
        
        Args:
            slide_paths: List of paths to WSI files
            output_dir: Directory to save extracted features
            force: Reprocess even if outputs exist
            
        Returns:
            Dictionary mapping slide names to output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for slide_path in tqdm(slide_paths, desc="Processing slides"):
            slide_name = slide_path.stem
            output_path = output_dir / f"{slide_name}.h5"
            
            try:
                self.process_slide(slide_path, output_path, force=force)
                results[slide_name] = output_path
            except Exception as e:
                print(f"Error processing {slide_name}: {e}")
                continue
        
        return results


class ChunkedFeatureLoader:
    """
    Memory-efficient loader for pre-extracted WSI features.
    
    Loads features from HDF5 files in chunks to prevent memory issues
    when working with many large feature files.
    """
    
    def __init__(
        self,
        feature_dir: Path,
        max_tiles: int = 4000,
        cache_size: int = 10
    ):
        """
        Initialize the feature loader.
        
        Args:
            feature_dir: Directory containing HDF5 feature files
            max_tiles: Maximum tiles to load per slide
            cache_size: Number of slides to keep in memory
        """
        self.feature_dir = Path(feature_dir)
        self.max_tiles = max_tiles
        self.cache_size = cache_size
        
        # LRU cache for loaded features
        from collections import OrderedDict
        self._cache = OrderedDict()
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
            gc.collect()
    
    def load_features(
        self,
        patient_id: str,
        pad_to: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load features for a patient, with optional padding.
        
        Args:
            patient_id: Patient/slide identifier
            pad_to: Pad features to this size (optional)
            
        Returns:
            Tuple of (features, actual_n_tiles)
        """
        # Check cache first
        cache_key = (patient_id, pad_to)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        
        # Load from disk
        feature_path = self.feature_dir / f"{patient_id}.h5"
        
        if not feature_path.exists():
            # Return zeros if file doesn't exist
            pad_size = pad_to or self.max_tiles
            return np.zeros((pad_size, 1024), dtype=np.float32), 0
        
        with h5py.File(feature_path, 'r') as f:
            features = f['features'][:]
        
        actual_n_tiles = features.shape[0]
        
        # Subsample if too many tiles
        if actual_n_tiles > self.max_tiles:
            indices = np.random.choice(actual_n_tiles, self.max_tiles, replace=False)
            features = features[indices]
            actual_n_tiles = self.max_tiles
        
        # Pad if requested
        if pad_to is not None and actual_n_tiles < pad_to:
            padding = np.zeros((pad_to - actual_n_tiles, features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        # Cache result
        result = (features, actual_n_tiles)
        self._cache[cache_key] = result
        self._evict_if_needed()
        
        return result
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._cache.clear()
        gc.collect()
