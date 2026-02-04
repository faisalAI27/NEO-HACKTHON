# 🚀 FLAGSHIP IMPLEMENTATION PLAN
# Multi-Modal Transformer for Survival Prediction with Cross-Modal Explainability

## Project Codename: **MOSAIC** (Multi-Omics Survival AI with Cross-modal Interpretability)

---

# PHASE 0: ENVIRONMENT & INFRASTRUCTURE SETUP

## 0.1 Hardware Requirements

```
RECOMMENDED SETUP:
├── GPU: NVIDIA A100 (80GB) or RTX 4090 (24GB)
│   └── Minimum: RTX 3090 (24GB) for WSI feature extraction
├── RAM: 128GB (WSI processing is memory-intensive)
├── Storage: 2TB SSD
│   ├── Raw SVS files: ~500GB (172 × 1-3GB each)
│   ├── Extracted features: ~50GB
│   └── Checkpoints/outputs: ~100GB
└── CPU: 16+ cores (for parallel tile extraction)
```

## 0.2 Software Environment

```bash
# Create conda environment
conda create -n mosaic python=3.10 -y
conda activate mosaic

# Core deep learning
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.1.0
pip install einops==0.7.0

# WSI processing
conda install -c conda-forge openslide-python
pip install openslide-python
pip install histolab==0.6.0
pip install h5py==3.10.0

# Pretrained pathology models
pip install timm==0.9.12
pip install huggingface_hub==0.20.0

# Survival analysis
pip install lifelines==0.27.8
pip install scikit-survival==0.22.0
pip install pycox==0.2.3

# Multi-omics & bioinformatics
pip install scanpy==1.9.6
pip install anndata==0.10.3

# Explainability
pip install captum==0.6.0
pip install shap==0.44.0

# Utilities
pip install pandas==2.1.0
pip install numpy==1.26.0
pip install scipy==1.11.0
pip install matplotlib==3.8.0
pip install seaborn==0.13.0
pip install tqdm==4.66.0
pip install wandb==0.16.0  # Experiment tracking
```

## 0.3 Project Directory Structure

```
MOSAIC/
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── data/
│   ├── raw/
│   │   ├── svs/                      # Symlink to Copy of data/
│   │   ├── clinical.txt
│   │   ├── transcriptomics.txt
│   │   ├── methylation.txt
│   │   ├── mutations.txt
│   │   └── follow_up.txt
│   ├── processed/
│   │   ├── wsi_features/             # Extracted tile features (.h5)
│   │   ├── rna_processed.pkl
│   │   ├── methylation_processed.pkl
│   │   ├── mutations_processed.pkl
│   │   └── clinical_processed.pkl
│   └── splits/
│       └── cv_folds.pkl              # Cross-validation splits
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wsi_dataset.py
│   │   ├── omics_dataset.py
│   │   └── multimodal_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders/
│   │   │   ├── wsi_encoder.py        # UNI/HIPT feature aggregation
│   │   │   ├── rna_encoder.py        # Gene expression encoder
│   │   │   ├── methylation_encoder.py
│   │   │   ├── mutation_encoder.py
│   │   │   └── clinical_encoder.py
│   │   ├── fusion/
│   │   │   ├── cross_attention.py
│   │   │   └── perceiver_fusion.py
│   │   └── mosaic.py                 # Main model
│   ├── losses/
│   │   ├── __init__.py
│   │   └── cox_loss.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── explainability/
│       ├── __init__.py
│       ├── attention_viz.py
│       └── shap_analysis.py
├── scripts/
│   ├── 01_extract_wsi_features.py
│   ├── 02_preprocess_omics.py
│   ├── 03_create_splits.py
│   ├── 04_train_unimodal.py
│   ├── 05_train_multimodal.py
│   ├── 06_evaluate.py
│   └── 07_explainability.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── predictions/
└── requirements.txt
```

---

# PHASE 1: DATA PREPROCESSING PIPELINE

## 1.1 Patient ID Harmonization

The first critical step is creating a **unified patient registry** that maps IDs across all modalities.

```python
# src/data/patient_registry.py

import pandas as pd
import re
from pathlib import Path

class PatientRegistry:
    """
    Unified patient ID mapping across all modalities.
    
    ID Formats in your data:
    - clinical.txt: TCGA-CR-7392
    - transcriptomics.txt: TCGA-CV-6934-01A-11R-1915-07
    - methylation.txt: TCGA-BA-4074-01
    - mutations.txt: TCGA-BA-4074-01A-11D-1998-08
    - SVS files: TCGA-CV-7235.svs
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.patient_map = {}
        
    def extract_patient_id(self, full_id: str) -> str:
        """Extract base patient ID (TCGA-XX-XXXX) from any format."""
        match = re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', full_id)
        if match:
            return match.group(1)
        return None
    
    def build_registry(self):
        """Build complete patient registry with modality availability."""
        
        # 1. Get all patient IDs from clinical (ground truth)
        clinical = pd.read_csv(self.data_dir / 'clinical.txt', sep='\t')
        clinical_ids = set(clinical['cases.submitter_id'].dropna())
        
        # 2. Get IDs from transcriptomics
        rna = pd.read_csv(self.data_dir / 'transcriptomics.txt', sep='\t', nrows=0)
        rna_ids = set(self.extract_patient_id(col) for col in rna.columns[1:])
        
        # 3. Get IDs from methylation
        meth = pd.read_csv(self.data_dir / 'methylation.txt', sep='\t', nrows=0)
        meth_ids = set(self.extract_patient_id(col) for col in meth.columns[1:])
        
        # 4. Get IDs from SVS files
        svs_dir = self.data_dir / 'Copy of data'
        svs_ids = set(f.stem for f in svs_dir.glob('*.svs'))
        
        # 5. Build registry
        all_ids = clinical_ids | rna_ids | meth_ids | svs_ids
        
        registry = []
        for pid in all_ids:
            registry.append({
                'patient_id': pid,
                'has_clinical': pid in clinical_ids,
                'has_rna': pid in rna_ids,
                'has_methylation': pid in meth_ids,
                'has_svs': pid in svs_ids,
                'has_all': all([
                    pid in clinical_ids,
                    pid in rna_ids,
                    pid in meth_ids,
                    pid in svs_ids
                ])
            })
        
        self.registry = pd.DataFrame(registry)
        return self.registry
    
    def get_complete_patients(self) -> list:
        """Return patients with ALL modalities available."""
        return self.registry[self.registry['has_all']]['patient_id'].tolist()
```

### Expected Output
```
PATIENT REGISTRY SUMMARY
========================
Total unique patients: 172
├── With clinical data: 82
├── With RNA-seq: 92
├── With methylation: 96
├── With SVS images: 172
└── With ALL modalities: ~70-75 (your analysis cohort)
```

---

## 1.2 Whole Slide Image (WSI) Feature Extraction

### 1.2.1 Tissue Detection & Tiling

```python
# src/data/wsi_preprocessing.py

import openslide
import numpy as np
from PIL import Image
from pathlib import Path
import h5py
from tqdm import tqdm
import cv2

class WSITiler:
    """
    Extract tissue tiles from whole slide images.
    
    Pipeline:
    1. Load SVS at low resolution (thumbnail)
    2. Detect tissue regions (Otsu thresholding)
    3. Generate tile coordinates at target magnification
    4. Extract and filter tiles
    """
    
    def __init__(
        self,
        tile_size: int = 256,
        target_mpp: float = 0.5,  # microns per pixel (20x ≈ 0.5 mpp)
        tissue_threshold: float = 0.5,  # minimum tissue fraction
        max_tiles: int = 4000,  # maximum tiles per slide
    ):
        self.tile_size = tile_size
        self.target_mpp = target_mpp
        self.tissue_threshold = tissue_threshold
        self.max_tiles = max_tiles
    
    def get_tissue_mask(self, slide: openslide.OpenSlide, level: int = -1) -> np.ndarray:
        """Generate binary tissue mask using Otsu thresholding."""
        
        # Read thumbnail
        thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
        thumb = np.array(thumb.convert('RGB'))
        
        # Convert to HSV and threshold on saturation
        hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Otsu thresholding
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask > 0
    
    def get_tile_coordinates(
        self, 
        slide: openslide.OpenSlide,
        tissue_mask: np.ndarray
    ) -> list[tuple[int, int]]:
        """Generate tile coordinates at target magnification."""
        
        # Get slide properties
        slide_mpp = float(slide.properties.get('openslide.mpp-x', 0.25))
        downsample = self.target_mpp / slide_mpp
        level = slide.get_best_level_for_downsample(downsample)
        
        # Calculate effective tile size at level 0
        level_downsample = slide.level_downsamples[level]
        tile_size_level0 = int(self.tile_size * level_downsample)
        
        # Calculate mask scale factor
        mask_scale = slide.level_dimensions[-1][0] / slide.level_dimensions[0][0]
        
        coordinates = []
        dims = slide.level_dimensions[0]
        
        for y in range(0, dims[1], tile_size_level0):
            for x in range(0, dims[0], tile_size_level0):
                # Check tissue mask
                mask_x = int(x * mask_scale)
                mask_y = int(y * mask_scale)
                mask_tile_size = int(tile_size_level0 * mask_scale)
                
                mask_region = tissue_mask[
                    mask_y:mask_y + mask_tile_size,
                    mask_x:mask_x + mask_tile_size
                ]
                
                tissue_fraction = mask_region.mean() if mask_region.size > 0 else 0
                
                if tissue_fraction >= self.tissue_threshold:
                    coordinates.append((x, y, level))
        
        # Limit tiles if too many
        if len(coordinates) > self.max_tiles:
            indices = np.random.choice(len(coordinates), self.max_tiles, replace=False)
            coordinates = [coordinates[i] for i in indices]
        
        return coordinates
    
    def extract_tiles(
        self,
        slide_path: Path,
        output_path: Path
    ) -> dict:
        """Extract all tiles from a slide and save to HDF5."""
        
        slide = openslide.OpenSlide(str(slide_path))
        
        # Get tissue mask
        tissue_mask = self.get_tissue_mask(slide)
        
        # Get tile coordinates
        coordinates = self.get_tile_coordinates(slide, tissue_mask)
        
        if len(coordinates) == 0:
            return {'num_tiles': 0, 'status': 'no_tissue'}
        
        # Extract tiles
        tiles = []
        for x, y, level in tqdm(coordinates, desc=f"Extracting {slide_path.stem}"):
            tile = slide.read_region((x, y), level, (self.tile_size, self.tile_size))
            tile = np.array(tile.convert('RGB'))
            tiles.append(tile)
        
        tiles = np.stack(tiles)
        
        # Save to HDF5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('tiles', data=tiles, compression='gzip')
            f.create_dataset('coordinates', data=np.array(coordinates))
            f.attrs['slide_path'] = str(slide_path)
            f.attrs['tile_size'] = self.tile_size
            f.attrs['target_mpp'] = self.target_mpp
        
        slide.close()
        
        return {
            'num_tiles': len(tiles),
            'status': 'success'
        }
```

### 1.2.2 Feature Extraction with UNI

```python
# src/data/wsi_feature_extraction.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
from huggingface_hub import hf_hub_download
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

class TileDataset(Dataset):
    """Dataset for loading tiles from HDF5."""
    
    def __init__(self, h5_path: Path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        with h5py.File(h5_path, 'r') as f:
            self.num_tiles = f['tiles'].shape[0]
    
    def __len__(self):
        return self.num_tiles
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            tile = f['tiles'][idx]
        
        if self.transform:
            tile = self.transform(tile)
        
        return tile


class UNIFeatureExtractor:
    """
    Extract features using UNI - Universal Foundation Model for Pathology.
    
    Reference: Chen et al., "Towards a general-purpose foundation model 
    for computational pathology", Nature Medicine 2024
    
    Model: ViT-L/16 trained on 100M+ pathology images
    Output: 1024-dimensional feature per tile
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = self._load_uni()
        self.transform = self._get_transform()
    
    def _load_uni(self) -> nn.Module:
        """Load pretrained UNI model."""
        
        # Download from HuggingFace
        model_path = hf_hub_download(
            repo_id="MahmoodLab/UNI",
            filename="pytorch_model.bin"
        )
        
        # Create ViT-L/16 architecture
        model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0,  # Remove classification head
            global_pool='token'  # Use CLS token
        )
        
        # Load pretrained weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self):
        """Get preprocessing transforms for UNI."""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def extract_features(
        self,
        tiles_h5_path: Path,
        output_path: Path,
        batch_size: int = 64
    ) -> dict:
        """Extract UNI features for all tiles in an HDF5 file."""
        
        dataset = TileDataset(tiles_h5_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        
        all_features = []
        
        for batch in tqdm(loader, desc="Extracting features"):
            batch = batch.to(self.device)
            features = self.model(batch)  # [B, 1024]
            all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        
        # Save features
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=all_features, compression='gzip')
            f.attrs['model'] = 'UNI'
            f.attrs['feature_dim'] = 1024
        
        return {
            'num_tiles': len(all_features),
            'feature_dim': 1024
        }
```

### Expected Output per Slide
```
SLIDE: TCGA-CV-7235.svs
========================
├── Original size: 98,304 × 67,584 pixels (0.25 μm/pixel)
├── Target magnification: 20× (0.5 μm/pixel)
├── Tile size: 256 × 256 pixels
├── Tiles extracted: 2,847
├── Feature shape: [2847, 1024]
└── Storage: features/TCGA-CV-7235.h5 (11.5 MB)
```

---

## 1.3 Transcriptomics Preprocessing

```python
# src/data/rna_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

class RNAPreprocessor:
    """
    Preprocess RNA-seq gene expression data.
    
    Pipeline:
    1. Load raw counts
    2. Filter low-expression genes
    3. Log2(count + 1) transformation
    4. Z-score normalization per gene
    5. Optional: Variance filtering / Pathway aggregation
    """
    
    def __init__(
        self,
        min_count: int = 10,  # Minimum count threshold
        min_samples: float = 0.2,  # Gene must be expressed in X% of samples
        variance_percentile: float = 0.5  # Keep top 50% most variable genes
    ):
        self.min_count = min_count
        self.min_samples = min_samples
        self.variance_percentile = variance_percentile
        self.gene_scaler = StandardScaler()
        self.selected_genes = None
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load transcriptomics data."""
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        
        # Extract patient IDs from column names
        import re
        new_cols = {}
        for col in df.columns:
            match = re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', col)
            if match:
                new_cols[col] = match.group(1)
        
        df = df.rename(columns=new_cols)
        
        # Remove duplicates (keep tumor sample -01 over normal -11)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        return df
    
    def filter_genes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter low-expression and low-variance genes."""
        
        # Filter 1: Minimum count threshold
        expressed = (df > self.min_count).sum(axis=1) >= (df.shape[1] * self.min_samples)
        df = df[expressed]
        print(f"After count filter: {df.shape[0]} genes")
        
        # Filter 2: Variance filter (after log transform)
        log_df = np.log2(df + 1)
        variances = log_df.var(axis=1)
        threshold = variances.quantile(1 - self.variance_percentile)
        high_var = variances >= threshold
        df = df[high_var]
        print(f"After variance filter: {df.shape[0]} genes")
        
        self.selected_genes = df.index.tolist()
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform and z-score normalization."""
        
        # Log2(count + 1)
        df = np.log2(df + 1)
        
        # Z-score per gene
        df = pd.DataFrame(
            self.gene_scaler.fit_transform(df.T).T,
            index=df.index,
            columns=df.columns
        )
        
        return df
    
    def process(self, filepath: Path, output_path: Path) -> dict:
        """Full preprocessing pipeline."""
        
        # Load
        df = self.load_data(filepath)
        print(f"Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
        
        # Filter
        df = self.filter_genes(df)
        
        # Transform
        df = self.transform(df)
        
        # Save
        output = {
            'expression_matrix': df,
            'gene_names': df.index.tolist(),
            'patient_ids': df.columns.tolist(),
            'scaler': self.gene_scaler
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        
        return {
            'num_genes': df.shape[0],
            'num_samples': df.shape[1],
            'output_path': output_path
        }
```

### Expected Output
```
RNA-SEQ PREPROCESSING
=====================
├── Input: 20,504 genes × 92 samples
├── After count filter: 15,847 genes
├── After variance filter: 7,923 genes
├── Transform: log2(count+1) → z-score
└── Output: processed/rna_processed.pkl
```

---

## 1.4 Methylation Preprocessing

```python
# src/data/methylation_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

class MethylationPreprocessor:
    """
    Preprocess DNA methylation data.
    
    Input: Beta values (0-1)
    Output: M-values (log2 ratio) - better statistical properties
    
    M = log2(beta / (1 - beta))
    """
    
    def __init__(
        self,
        variance_percentile: float = 0.5,  # Keep top 50% most variable
        clip_beta: tuple = (0.001, 0.999)  # Avoid log(0)
    ):
        self.variance_percentile = variance_percentile
        self.clip_beta = clip_beta
        self.selected_genes = None
    
    def beta_to_mvalue(self, beta: np.ndarray) -> np.ndarray:
        """Convert beta values to M-values."""
        beta = np.clip(beta, self.clip_beta[0], self.clip_beta[1])
        return np.log2(beta / (1 - beta))
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load methylation data."""
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        
        # Extract patient IDs
        import re
        new_cols = {}
        for col in df.columns:
            match = re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', col)
            if match:
                new_cols[col] = match.group(1)
        
        df = df.rename(columns=new_cols)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        return df
    
    def process(self, filepath: Path, output_path: Path) -> dict:
        """Full preprocessing pipeline."""
        
        # Load
        df = self.load_data(filepath)
        print(f"Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
        
        # Variance filter
        variances = df.var(axis=1)
        threshold = variances.quantile(1 - self.variance_percentile)
        df = df[variances >= threshold]
        print(f"After variance filter: {df.shape[0]} genes")
        
        self.selected_genes = df.index.tolist()
        
        # Convert to M-values
        df = df.apply(self.beta_to_mvalue)
        
        # Z-score normalization
        df = (df - df.mean()) / df.std()
        
        # Save
        output = {
            'methylation_matrix': df,
            'gene_names': df.index.tolist(),
            'patient_ids': df.columns.tolist()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        
        return {
            'num_genes': df.shape[0],
            'num_samples': df.shape[1]
        }
```

---

## 1.5 Mutation Preprocessing

```python
# src/data/mutation_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

class MutationPreprocessor:
    """
    Preprocess somatic mutation data from MAF format.
    
    Output features per patient:
    1. Binary mutation vector (gene × patient)
    2. Mutation burden (total count)
    3. Driver gene profile
    4. Mutation type distribution
    """
    
    # HNSC driver genes from TCGA publications
    DRIVER_GENES = [
        'TP53', 'CDKN2A', 'PIK3CA', 'NOTCH1', 'FAT1',
        'CASP8', 'HRAS', 'NFE2L2', 'AJUBA', 'TGFBR2',
        'HLA-A', 'FBXW7', 'PTEN', 'EGFR', 'NSD1'
    ]
    
    # Functional mutation types (non-silent)
    FUNCTIONAL_TYPES = [
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
        'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins'
    ]
    
    def __init__(self):
        self.gene_list = None
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load MAF file."""
        # MAF files can have many columns, select relevant ones
        usecols = [
            'Hugo_Symbol', 'Variant_Classification', 'Variant_Type',
            'Tumor_Sample_Barcode', 'HGVSp_Short', 't_depth', 't_alt_count'
        ]
        
        df = pd.read_csv(filepath, sep='\t', usecols=lambda c: c in usecols)
        return df
    
    def extract_patient_id(self, barcode: str) -> str:
        """Extract patient ID from tumor sample barcode."""
        import re
        match = re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', barcode)
        return match.group(1) if match else None
    
    def create_binary_matrix(self, maf: pd.DataFrame, patients: list) -> pd.DataFrame:
        """Create binary gene × patient mutation matrix."""
        
        # Filter to functional mutations
        maf = maf[maf['Variant_Classification'].isin(self.FUNCTIONAL_TYPES)]
        
        # Get patient IDs
        maf['patient_id'] = maf['Tumor_Sample_Barcode'].apply(self.extract_patient_id)
        
        # Get all mutated genes
        genes = maf['Hugo_Symbol'].unique()
        self.gene_list = list(genes)
        
        # Create matrix
        matrix = pd.DataFrame(0, index=genes, columns=patients)
        
        for _, row in maf.iterrows():
            gene = row['Hugo_Symbol']
            patient = row['patient_id']
            if patient in patients and gene in genes:
                matrix.loc[gene, patient] = 1
        
        return matrix
    
    def compute_features(self, maf: pd.DataFrame, patients: list) -> dict:
        """Compute all mutation features."""
        
        # Binary matrix
        binary_matrix = self.create_binary_matrix(maf, patients)
        
        # Mutation burden per patient
        mutation_burden = binary_matrix.sum(axis=0)
        
        # Driver gene profile
        driver_present = [g for g in self.DRIVER_GENES if g in binary_matrix.index]
        driver_matrix = binary_matrix.loc[driver_present]
        
        # Mutation type distribution
        maf['patient_id'] = maf['Tumor_Sample_Barcode'].apply(self.extract_patient_id)
        type_counts = maf.groupby(['patient_id', 'Variant_Classification']).size().unstack(fill_value=0)
        type_counts = type_counts.reindex(patients, fill_value=0)
        
        return {
            'binary_matrix': binary_matrix,
            'mutation_burden': mutation_burden,
            'driver_matrix': driver_matrix,
            'driver_genes': driver_present,
            'type_distribution': type_counts
        }
    
    def process(self, filepath: Path, patients: list, output_path: Path) -> dict:
        """Full preprocessing pipeline."""
        
        maf = self.load_data(filepath)
        print(f"Loaded: {len(maf)} mutations")
        
        features = self.compute_features(maf, patients)
        
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        
        return {
            'num_genes': features['binary_matrix'].shape[0],
            'num_patients': features['binary_matrix'].shape[1],
            'num_drivers': len(features['driver_genes'])
        }
```

---

## 1.6 Clinical Data & Survival Labels

```python
# src/data/clinical_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

class ClinicalPreprocessor:
    """
    Preprocess clinical data and extract survival labels.
    
    Key outputs:
    1. Survival time (days)
    2. Event indicator (1=death, 0=censored)
    3. Clinical features for model
    """
    
    # Clinical features to include in model
    FEATURES = [
        'demographic.age_at_index',
        'demographic.gender',
        'diagnoses.tumor_grade',
        'diagnoses.ajcc_pathologic_stage',
        'diagnoses.ajcc_pathologic_t',
        'diagnoses.ajcc_pathologic_n',
    ]
    
    def __init__(self):
        self.feature_encoders = {}
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load clinical data."""
        df = pd.read_csv(filepath, sep='\t')
        df = df.set_index('cases.submitter_id')
        return df
    
    def extract_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract survival time and event indicator."""
        
        survival = pd.DataFrame(index=df.index)
        
        # Event indicator: 1 if dead, 0 if alive (censored)
        survival['event'] = (df['demographic.vital_status'] == 'Dead').astype(int)
        
        # Survival time
        survival['time'] = df.apply(
            lambda row: row['demographic.days_to_death'] 
            if row['demographic.vital_status'] == 'Dead' 
            else row['diagnoses.days_to_last_follow_up'],
            axis=1
        )
        
        # Clean missing values
        survival = survival.dropna()
        survival = survival[survival['time'] > 0]
        
        return survival
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode clinical features for model input."""
        
        features = pd.DataFrame(index=df.index)
        
        for col in self.FEATURES:
            if col not in df.columns:
                continue
                
            values = df[col]
            
            if values.dtype == 'object':
                # Categorical: one-hot encode
                dummies = pd.get_dummies(values, prefix=col.split('.')[-1])
                features = pd.concat([features, dummies], axis=1)
            else:
                # Numeric: standardize
                features[col.split('.')[-1]] = (values - values.mean()) / values.std()
        
        return features
    
    def process(self, filepath: Path, output_path: Path) -> dict:
        """Full preprocessing pipeline."""
        
        df = self.load_data(filepath)
        print(f"Loaded: {len(df)} patients")
        
        # Extract survival
        survival = self.extract_survival(df)
        print(f"With survival data: {len(survival)} patients")
        
        # Encode features
        features = self.encode_features(df)
        
        # Align
        common_idx = survival.index.intersection(features.index)
        survival = survival.loc[common_idx]
        features = features.loc[common_idx]
        
        output = {
            'survival': survival,
            'features': features,
            'patient_ids': common_idx.tolist()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        
        return {
            'num_patients': len(common_idx),
            'num_events': survival['event'].sum(),
            'num_features': features.shape[1]
        }
```

### Expected Clinical Output
```
CLINICAL PREPROCESSING
======================
├── Total patients: 82
├── With survival data: 78
├── Events (deaths): 35 (45%)
├── Censored: 43 (55%)
├── Median survival: 1,247 days
├── Clinical features: 15 (age, gender, stage dummies, etc.)
└── Output: processed/clinical_processed.pkl
```

---

## 1.7 Cross-Validation Split Generation

```python
# src/data/create_splits.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pickle

def create_cv_splits(
    patient_ids: list,
    events: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    output_path: Path = None
) -> dict:
    """
    Create stratified cross-validation splits.
    
    Stratify by event status to ensure balanced censoring in each fold.
    """
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, events)):
        splits[fold] = {
            'train': [patient_ids[i] for i in train_idx],
            'val': [patient_ids[i] for i in val_idx]
        }
        
        train_events = events[train_idx].sum()
        val_events = events[val_idx].sum()
        
        print(f"Fold {fold}: Train={len(train_idx)} ({train_events} events), "
              f"Val={len(val_idx)} ({val_events} events)")
    
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(splits, f)
    
    return splits
```

---

# PHASE 2: MODALITY-SPECIFIC ENCODERS

## 2.1 WSI Encoder (Attention-based Multiple Instance Learning)

```python
# src/models/encoders/wsi_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GatedAttentionPooling(nn.Module):
    """
    Gated Attention-based MIL Pooling.
    
    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018
    
    Aggregates N tile features into single slide representation using learned attention.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # UNI feature dimension
        hidden_dim: int = 512,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Tile features [B, N, D] or [N, D]
            return_attention: Whether to return attention weights
        
        Returns:
            pooled: Aggregated features [B, D] or [D]
            attention: Attention weights [B, N] or [N] (optional)
        """
        
        # Handle both batched and unbatched input
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        # Gated attention
        A_V = self.attention_V(x)  # [B, N, H]
        A_U = self.attention_U(x)  # [B, N, H]
        A = self.attention_weights(A_V * A_U)  # [B, N, 1]
        A = A.squeeze(-1)  # [B, N]
        
        # Softmax over tiles
        A = F.softmax(A, dim=-1)  # [B, N]
        A = self.dropout(A)
        
        # Weighted aggregation
        pooled = torch.bmm(A.unsqueeze(1), x).squeeze(1)  # [B, D]
        
        if squeeze:
            pooled = pooled.squeeze(0)
            A = A.squeeze(0)
        
        if return_attention:
            return pooled, A
        return pooled


class WSIEncoder(nn.Module):
    """
    Full WSI encoder with feature projection and attention pooling.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        
        # Project UNI features
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention pooling
        self.attention_pool = GatedAttentionPooling(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # Final projection to common embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Tile features [B, N, 1024]
        
        Returns:
            embedding: Slide embedding [B, output_dim]
            attention: Tile attention weights [B, N] (optional)
        """
        
        # Project
        x = self.projector(x)  # [B, N, hidden_dim]
        
        # Pool
        if return_attention:
            pooled, attention = self.attention_pool(x, return_attention=True)
        else:
            pooled = self.attention_pool(x)
        
        # Output projection
        embedding = self.output_proj(pooled)  # [B, output_dim]
        
        if return_attention:
            return embedding, attention
        return embedding
```

---

## 2.2 Gene Expression Encoder

```python
# src/models/encoders/rna_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PathwayAttentionEncoder(nn.Module):
    """
    Gene expression encoder with pathway-aware attention.
    
    Architecture:
    1. Gene embedding layer
    2. Self-attention over genes (pathway context)
    3. Aggregation to pathway-level
    4. Projection to common embedding space
    """
    
    def __init__(
        self,
        num_genes: int = 7923,
        gene_embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Gene value embedding
        self.gene_encoder = nn.Sequential(
            nn.Linear(1, gene_embed_dim),
            nn.ReLU(),
            nn.Linear(gene_embed_dim, gene_embed_dim)
        )
        
        # Learnable gene position embeddings
        self.gene_embeddings = nn.Parameter(
            torch.randn(num_genes, gene_embed_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gene_embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling over genes
        self.pool_attention = nn.Sequential(
            nn.Linear(gene_embed_dim, gene_embed_dim // 2),
            nn.Tanh(),
            nn.Linear(gene_embed_dim // 2, 1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(gene_embed_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Gene expression values [B, num_genes]
        
        Returns:
            embedding: [B, output_dim]
            attention: [B, num_genes] (optional)
        """
        
        B = x.shape[0]
        
        # Encode gene values
        x = x.unsqueeze(-1)  # [B, G, 1]
        x = self.gene_encoder(x)  # [B, G, embed_dim]
        
        # Add gene position embeddings
        x = x + self.gene_embeddings.unsqueeze(0)  # [B, G, embed_dim]
        
        # Transformer encoding
        x = self.transformer(x)  # [B, G, embed_dim]
        
        # Attention pooling
        attn_logits = self.pool_attention(x).squeeze(-1)  # [B, G]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, G]
        
        # Weighted aggregation
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, embed_dim]
        
        # Output projection
        embedding = self.output_proj(pooled)  # [B, output_dim]
        
        if return_attention:
            return embedding, attn_weights
        return embedding


class MLPEncoder(nn.Module):
    """
    Simpler MLP-based gene expression encoder.
    Use when transformer is too heavy or for ablation studies.
    """
    
    def __init__(
        self,
        num_genes: int = 7923,
        hidden_dims: list = [2048, 512],
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        in_dim = num_genes
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.encoder(x)
```

---

## 2.3 Methylation Encoder

```python
# src/models/encoders/methylation_encoder.py

import torch
import torch.nn as nn

class MethylationEncoder(nn.Module):
    """
    DNA methylation encoder using MLP with bottleneck.
    
    Can optionally use VAE for probabilistic encoding.
    """
    
    def __init__(
        self,
        num_genes: int = 10057,
        hidden_dims: list = [2048, 512],
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        in_dim = num_genes
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.encoder(x)
```

---

## 2.4 Mutation Encoder

```python
# src/models/encoders/mutation_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MutationEncoder(nn.Module):
    """
    Somatic mutation encoder with attention over driver genes.
    
    Combines:
    1. Binary mutation vector encoding
    2. Driver gene attention
    3. Mutation burden feature
    """
    
    def __init__(
        self,
        num_genes: int = 1500,
        num_drivers: int = 15,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Binary matrix encoder
        self.binary_encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Driver gene attention
        self.driver_attention = nn.Sequential(
            nn.Linear(num_drivers, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Mutation burden encoder
        self.burden_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim // 4)
        )
        
        # Fusion and output
        combined_dim = hidden_dim + hidden_dim + hidden_dim // 4
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        binary: torch.Tensor,       # [B, num_genes]
        drivers: torch.Tensor,      # [B, num_drivers]
        burden: torch.Tensor        # [B, 1]
    ):
        """Encode mutation features."""
        
        # Encode each component
        binary_feat = self.binary_encoder(binary)    # [B, hidden_dim]
        driver_feat = self.driver_attention(drivers) # [B, hidden_dim]
        burden_feat = self.burden_encoder(burden)    # [B, hidden_dim//4]
        
        # Concatenate
        combined = torch.cat([binary_feat, driver_feat, burden_feat], dim=-1)
        
        # Output
        return self.output_proj(combined)  # [B, output_dim]
```

---

## 2.5 Clinical Encoder

```python
# src/models/encoders/clinical_encoder.py

import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    """
    Clinical features encoder.
    """
    
    def __init__(
        self,
        num_features: int = 15,
        hidden_dim: int = 64,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor):
        return self.encoder(x)
```

---

# PHASE 3: MULTI-MODAL FUSION ARCHITECTURE

## 3.1 Cross-Modal Attention Fusion

```python
# src/models/fusion/cross_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer.
    
    Allows each modality to attend to all other modalities,
    learning which cross-modal relationships are important.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,    # [B, embed_dim]
        context: torch.Tensor,  # [B, num_modalities, embed_dim]
        return_attention: bool = False
    ):
        """
        Args:
            query: Query modality embedding
            context: Stack of all modality embeddings
        
        Returns:
            output: Attended representation
            attention: Cross-modal attention weights (optional)
        """
        
        B = query.shape[0]
        
        # Add sequence dimension to query
        query = query.unsqueeze(1)  # [B, 1, D]
        
        # Project
        Q = self.q_proj(query)     # [B, 1, D]
        K = self.k_proj(context)   # [B, M, D]
        V = self.v_proj(context)   # [B, M, D]
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b m (h d) -> b h m d', h=self.num_heads)
        V = rearrange(V, 'b m (h d) -> b h m d', h=self.num_heads)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)  # [B, H, 1, D/H]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, 1, D]
        out = self.out_proj(out).squeeze(1)  # [B, D]
        
        if return_attention:
            attn_weights = attn.mean(dim=1).squeeze(1)  # [B, M]
            return out, attn_weights
        return out


class PerceiverFusion(nn.Module):
    """
    Perceiver-style fusion with learnable latent queries.
    
    Reference: Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
    
    Architecture:
    1. Learnable latent queries attend to all modality embeddings
    2. Self-attention among latents
    3. Pool latents for final representation
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_latents: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim) * 0.02)
        
        # Cross-attention layers (latents attend to modalities)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Self-attention layers (latents attend to each other)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.cross_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.self_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Output pooling
        self.pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, modality_embeddings: torch.Tensor, return_attention: bool = False):
        """
        Args:
            modality_embeddings: [B, num_modalities, embed_dim]
        
        Returns:
            output: Fused representation [B, embed_dim]
            attention: Cross-modal attention (optional)
        """
        
        B = modality_embeddings.shape[0]
        
        # Initialize latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, L, D]
        
        all_attentions = []
        
        # Alternating cross-attention and self-attention
        for i in range(len(self.cross_attn_layers)):
            # Cross-attention: latents query modalities
            attended, attn_weights = self.cross_attn_layers[i](
                latents, modality_embeddings, modality_embeddings,
                need_weights=True
            )
            latents = self.cross_norms[i](latents + attended)
            all_attentions.append(attn_weights)
            
            # Self-attention among latents
            latents = self.self_norms[i](latents + self.self_attn_layers[i](latents))
        
        # Pool latents
        pool_weights = F.softmax(self.pool(latents).squeeze(-1), dim=-1)  # [B, L]
        output = torch.bmm(pool_weights.unsqueeze(1), latents).squeeze(1)  # [B, D]
        
        if return_attention:
            # Average attention across layers and latents
            avg_attention = torch.stack(all_attentions).mean(dim=(0, 2))  # [B, M]
            return output, avg_attention
        return output
```

---

## 3.2 Complete MOSAIC Model

```python
# src/models/mosaic.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoders.wsi_encoder import WSIEncoder
from .encoders.rna_encoder import PathwayAttentionEncoder, MLPEncoder
from .encoders.methylation_encoder import MethylationEncoder
from .encoders.mutation_encoder import MutationEncoder
from .encoders.clinical_encoder import ClinicalEncoder
from .fusion.cross_attention import PerceiverFusion


class MOSAIC(nn.Module):
    """
    Multi-Omics Survival AI with Cross-modal Interpretability (MOSAIC)
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         MOSAIC                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │
    │  │  WSI  │ │  RNA  │ │ Meth  │ │  Mut  │ │ Clin  │           │
    │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘           │
    │      │         │         │         │         │                 │
    │      ▼         ▼         ▼         ▼         ▼                 │
    │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │
    │  │Encoder│ │Encoder│ │Encoder│ │Encoder│ │Encoder│           │
    │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘           │
    │      │         │         │         │         │                 │
    │      └─────────┴────┬────┴─────────┴─────────┘                 │
    │                     ▼                                          │
    │            ┌─────────────────┐                                 │
    │            │ Perceiver Fusion│                                 │
    │            │ (Cross-Modal    │                                 │
    │            │  Attention)     │                                 │
    │            └────────┬────────┘                                 │
    │                     ▼                                          │
    │            ┌─────────────────┐                                 │
    │            │  Survival Head  │                                 │
    │            │  (Risk Score)   │                                 │
    │            └─────────────────┘                                 │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        # Encoder dimensions
        wsi_input_dim: int = 1024,      # UNI feature dim
        num_genes_rna: int = 7923,
        num_genes_meth: int = 10057,
        num_genes_mut: int = 1500,
        num_drivers: int = 15,
        num_clinical: int = 15,
        # Architecture
        embed_dim: int = 256,
        num_latents: int = 16,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        # RNA encoder type
        rna_encoder_type: str = 'mlp'  # 'mlp' or 'transformer'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Modality encoders
        self.wsi_encoder = WSIEncoder(
            input_dim=wsi_input_dim,
            output_dim=embed_dim,
            dropout=dropout
        )
        
        if rna_encoder_type == 'transformer':
            self.rna_encoder = PathwayAttentionEncoder(
                num_genes=num_genes_rna,
                output_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.rna_encoder = MLPEncoder(
                num_genes=num_genes_rna,
                output_dim=embed_dim,
                dropout=dropout
            )
        
        self.meth_encoder = MethylationEncoder(
            num_genes=num_genes_meth,
            output_dim=embed_dim,
            dropout=dropout
        )
        
        self.mut_encoder = MutationEncoder(
            num_genes=num_genes_mut,
            num_drivers=num_drivers,
            output_dim=embed_dim,
            dropout=dropout
        )
        
        self.clinical_encoder = ClinicalEncoder(
            num_features=num_clinical,
            output_dim=embed_dim
        )
        
        # Modality type embeddings
        self.modality_embeddings = nn.Parameter(torch.randn(5, embed_dim) * 0.02)
        
        # Fusion
        self.fusion = PerceiverFusion(
            embed_dim=embed_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            dropout=dropout
        )
        
        # Survival prediction head
        self.survival_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)  # Risk score
        )
    
    def encode_modalities(
        self,
        wsi_features: torch.Tensor,      # [B, N_tiles, 1024]
        rna: torch.Tensor,                # [B, num_genes_rna]
        methylation: torch.Tensor,        # [B, num_genes_meth]
        mutation_binary: torch.Tensor,    # [B, num_genes_mut]
        mutation_drivers: torch.Tensor,   # [B, num_drivers]
        mutation_burden: torch.Tensor,    # [B, 1]
        clinical: torch.Tensor,           # [B, num_clinical]
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Encode each modality into common embedding space.
        
        Returns:
            embeddings: [B, 5, embed_dim]
            attention_dict: Dictionary of modality-specific attention weights
        """
        
        attention_dict = {}
        
        # WSI encoding
        if return_attention:
            wsi_emb, wsi_attn = self.wsi_encoder(wsi_features, return_attention=True)
            attention_dict['wsi'] = wsi_attn
        else:
            wsi_emb = self.wsi_encoder(wsi_features)
        
        # RNA encoding
        if return_attention and hasattr(self.rna_encoder, 'forward'):
            try:
                rna_emb, rna_attn = self.rna_encoder(rna, return_attention=True)
                attention_dict['rna'] = rna_attn
            except:
                rna_emb = self.rna_encoder(rna)
        else:
            rna_emb = self.rna_encoder(rna)
        
        # Methylation encoding
        meth_emb = self.meth_encoder(methylation)
        
        # Mutation encoding
        mut_emb = self.mut_encoder(mutation_binary, mutation_drivers, mutation_burden)
        
        # Clinical encoding
        clin_emb = self.clinical_encoder(clinical)
        
        # Stack modality embeddings
        embeddings = torch.stack([wsi_emb, rna_emb, meth_emb, mut_emb, clin_emb], dim=1)
        
        # Add modality type embeddings
        embeddings = embeddings + self.modality_embeddings.unsqueeze(0)
        
        if return_attention:
            return embeddings, attention_dict
        return embeddings, None
    
    def forward(
        self,
        wsi_features: torch.Tensor,
        rna: torch.Tensor,
        methylation: torch.Tensor,
        mutation_binary: torch.Tensor,
        mutation_drivers: torch.Tensor,
        mutation_burden: torch.Tensor,
        clinical: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MOSAIC.
        
        Returns:
            Dictionary containing:
            - risk_score: Predicted survival risk [B, 1]
            - modality_attention: Cross-modal attention weights [B, 5]
            - tile_attention: WSI tile attention [B, N_tiles] (if return_attention)
            - gene_attention: Gene attention [B, num_genes] (if return_attention)
        """
        
        # Encode modalities
        embeddings, attn_dict = self.encode_modalities(
            wsi_features, rna, methylation,
            mutation_binary, mutation_drivers, mutation_burden,
            clinical, return_attention=return_attention
        )
        
        # Fuse modalities
        if return_attention:
            fused, modality_attn = self.fusion(embeddings, return_attention=True)
        else:
            fused = self.fusion(embeddings)
            modality_attn = None
        
        # Predict risk
        risk_score = self.survival_head(fused)
        
        output = {'risk_score': risk_score}
        
        if return_attention:
            output['modality_attention'] = modality_attn
            if attn_dict:
                output.update(attn_dict)
        
        return output
```

---

# PHASE 4: TRAINING PIPELINE

## 4.1 Cox Proportional Hazards Loss

```python
# src/losses/cox_loss.py

import torch
import torch.nn as nn

class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards partial likelihood loss.
    
    For survival data with censoring:
    - Uncensored (event=1): We know exact time of event
    - Censored (event=0): We only know patient survived at least until that time
    
    Loss = -∑_{i: δᵢ=1} [hᵢ - log(∑_{j∈Rᵢ} exp(hⱼ))]
    
    where:
    - hᵢ = risk score for patient i
    - δᵢ = event indicator (1=event, 0=censored)
    - Rᵢ = risk set (patients at risk at time tᵢ)
    
    Reference: DeepSurv (Katzman et al., 2018)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        risk_scores: torch.Tensor,  # [B]
        times: torch.Tensor,         # [B]
        events: torch.Tensor         # [B]
    ) -> torch.Tensor:
        """
        Compute negative partial log-likelihood.
        
        Args:
            risk_scores: Predicted risk scores (higher = higher risk)
            times: Observed times (event or censoring)
            events: Event indicators (1=event, 0=censored)
        
        Returns:
            loss: Scalar loss value
        """
        
        # Ensure proper shapes
        risk_scores = risk_scores.view(-1)
        times = times.view(-1)
        events = events.view(-1)
        
        # Sort by time (descending) for efficient risk set computation
        order = torch.argsort(times, descending=True)
        risk_scores = risk_scores[order]
        events = events[order]
        
        # Compute log cumulative sum of exp(risk)
        # This gives log(∑_{j∈Rᵢ} exp(hⱼ)) for each i
        exp_risk = torch.exp(risk_scores)
        cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
        log_cumsum = torch.log(cumsum_exp_risk + 1e-8)
        
        # Partial likelihood
        # Only uncensored patients contribute to the loss
        likelihood = risk_scores - log_cumsum
        loss = -torch.sum(likelihood * events)
        
        # Normalize by number of events
        n_events = events.sum()
        if n_events > 0:
            loss = loss / n_events
        
        return loss


class CoxPHLossWithTies(nn.Module):
    """
    Cox loss with Efron's approximation for ties.
    
    Use this if you have many patients with the same event time.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """Efron's approximation for tied survival times."""
        
        risk_scores = risk_scores.view(-1)
        times = times.view(-1)
        events = events.view(-1)
        
        # Sort by time
        order = torch.argsort(times)
        risk_scores = risk_scores[order]
        times = times[order]
        events = events[order]
        
        n = len(times)
        loss = torch.tensor(0.0, device=risk_scores.device)
        
        i = 0
        while i < n:
            # Find all observations at this time
            t_current = times[i]
            j = i
            while j < n and times[j] == t_current:
                j += 1
            
            # Risk set at this time
            risk_set_mask = times >= t_current
            risk_set = risk_scores[risk_set_mask]
            
            # Events at this time
            event_mask = (times == t_current) & (events == 1)
            event_risks = risk_scores[event_mask]
            d = event_mask.sum()
            
            if d > 0:
                # Efron's approximation
                sum_risk_set = torch.exp(risk_set).sum()
                sum_event = torch.exp(event_risks).sum()
                
                for k in range(int(d)):
                    adjusted_risk_set = sum_risk_set - (k / d) * sum_event
                    loss -= event_risks[k] - torch.log(adjusted_risk_set + 1e-8)
            
            i = j
        
        n_events = events.sum()
        if n_events > 0:
            loss = loss / n_events
        
        return loss
```

---

## 4.2 Training Loop

```python
# src/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from lifelines.utils import concordance_index
import numpy as np
from typing import Dict, Optional

from ..models.mosaic import MOSAIC
from ..losses.cox_loss import CoxPHLoss


class MOSAICTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training MOSAIC.
    """
    
    def __init__(
        self,
        model_config: Dict,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 100
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = MOSAIC(**model_config)
        
        # Loss
        self.loss_fn = CoxPHLoss()
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # For validation metrics
        self.val_risk_scores = []
        self.val_times = []
        self.val_events = []
    
    def forward(self, batch: Dict) -> Dict:
        return self.model(
            wsi_features=batch['wsi_features'],
            rna=batch['rna'],
            methylation=batch['methylation'],
            mutation_binary=batch['mutation_binary'],
            mutation_drivers=batch['mutation_drivers'],
            mutation_burden=batch['mutation_burden'],
            clinical=batch['clinical']
        )
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Forward pass
        output = self(batch)
        risk_scores = output['risk_score'].squeeze()
        
        # Compute loss
        loss = self.loss_fn(
            risk_scores,
            batch['survival_time'],
            batch['event']
        )
        
        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        # Forward pass
        output = self(batch)
        risk_scores = output['risk_score'].squeeze()
        
        # Compute loss
        loss = self.loss_fn(
            risk_scores,
            batch['survival_time'],
            batch['event']
        )
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Store for C-index computation
        self.val_risk_scores.append(risk_scores.detach().cpu())
        self.val_times.append(batch['survival_time'].detach().cpu())
        self.val_events.append(batch['event'].detach().cpu())
    
    def on_validation_epoch_end(self) -> None:
        # Concatenate all validation predictions
        risk_scores = torch.cat(self.val_risk_scores).numpy()
        times = torch.cat(self.val_times).numpy()
        events = torch.cat(self.val_events).numpy()
        
        # Compute C-index
        c_index = concordance_index(times, -risk_scores, events)
        self.log('val_c_index', c_index, prog_bar=True)
        
        # Clear
        self.val_risk_scores.clear()
        self.val_times.clear()
        self.val_events.clear()
    
    def configure_optimizers(self):
        # Separate parameters for different learning rates
        encoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.learning_rate * 0.1},
            {'params': other_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler with warmup
        def warmup_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 0.5 ** ((epoch - self.warmup_epochs) / 20)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
```

---

## 4.3 Dataset Class

```python
# src/data/multimodal_dataset.py

import torch
from torch.utils.data import Dataset
import h5py
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class MultiModalSurvivalDataset(Dataset):
    """
    Multi-modal dataset for survival prediction.
    
    Loads pre-processed features for each patient:
    - WSI features (from HDF5)
    - RNA expression (from pickle)
    - Methylation (from pickle)
    - Mutations (from pickle)
    - Clinical + survival (from pickle)
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        wsi_feature_dir: Path,
        rna_data: Dict,
        meth_data: Dict,
        mut_data: Dict,
        clinical_data: Dict,
        max_tiles: int = 4000
    ):
        self.patient_ids = patient_ids
        self.wsi_feature_dir = Path(wsi_feature_dir)
        self.max_tiles = max_tiles
        
        # Store preprocessed data
        self.rna_matrix = rna_data['expression_matrix']
        self.meth_matrix = meth_data['methylation_matrix']
        self.mut_binary = mut_data['binary_matrix']
        self.mut_drivers = mut_data['driver_matrix']
        self.mut_burden = mut_data['mutation_burden']
        self.survival = clinical_data['survival']
        self.clinical_features = clinical_data['features']
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        
        # Load WSI features
        wsi_path = self.wsi_feature_dir / f"{pid}.h5"
        with h5py.File(wsi_path, 'r') as f:
            wsi_features = f['features'][:]
        
        # Pad/truncate to max_tiles
        n_tiles = wsi_features.shape[0]
        if n_tiles > self.max_tiles:
            indices = np.random.choice(n_tiles, self.max_tiles, replace=False)
            wsi_features = wsi_features[indices]
        elif n_tiles < self.max_tiles:
            padding = np.zeros((self.max_tiles - n_tiles, wsi_features.shape[1]))
            wsi_features = np.concatenate([wsi_features, padding], axis=0)
        
        # Get other modalities
        rna = self.rna_matrix[pid].values if pid in self.rna_matrix.columns else np.zeros(self.rna_matrix.shape[0])
        meth = self.meth_matrix[pid].values if pid in self.meth_matrix.columns else np.zeros(self.meth_matrix.shape[0])
        
        # Mutations
        mut_binary = self.mut_binary[pid].values if pid in self.mut_binary.columns else np.zeros(self.mut_binary.shape[0])
        mut_drivers = self.mut_drivers[pid].values if pid in self.mut_drivers.columns else np.zeros(self.mut_drivers.shape[0])
        mut_burden = self.mut_burden[pid] if pid in self.mut_burden.index else 0
        
        # Clinical
        clinical = self.clinical_features.loc[pid].values if pid in self.clinical_features.index else np.zeros(self.clinical_features.shape[1])
        survival_time = self.survival.loc[pid, 'time'] if pid in self.survival.index else 0
        event = self.survival.loc[pid, 'event'] if pid in self.survival.index else 0
        
        return {
            'patient_id': pid,
            'wsi_features': torch.tensor(wsi_features, dtype=torch.float32),
            'rna': torch.tensor(rna, dtype=torch.float32),
            'methylation': torch.tensor(meth, dtype=torch.float32),
            'mutation_binary': torch.tensor(mut_binary, dtype=torch.float32),
            'mutation_drivers': torch.tensor(mut_drivers, dtype=torch.float32),
            'mutation_burden': torch.tensor([mut_burden], dtype=torch.float32),
            'clinical': torch.tensor(clinical, dtype=torch.float32),
            'survival_time': torch.tensor(survival_time, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32)
        }
```

---

# PHASE 5: EXPLAINABILITY & INTERPRETATION

## 5.1 Attention Visualization

```python
# src/explainability/attention_viz.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
import openslide
from typing import Dict, Tuple
from PIL import Image

class AttentionVisualizer:
    """
    Visualize attention weights from MOSAIC model.
    
    Creates:
    1. WSI attention heatmaps
    2. Cross-modal attention bar plots
    3. Gene importance plots
    """
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def get_attention_weights(self, batch: Dict) -> Dict[str, np.ndarray]:
        """Extract attention weights from model."""
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        # Forward with attention
        output = self.model(
            wsi_features=batch['wsi_features'],
            rna=batch['rna'],
            methylation=batch['methylation'],
            mutation_binary=batch['mutation_binary'],
            mutation_drivers=batch['mutation_drivers'],
            mutation_burden=batch['mutation_burden'],
            clinical=batch['clinical'],
            return_attention=True
        )
        
        attention = {}
        
        if 'modality_attention' in output:
            attention['modality'] = output['modality_attention'].cpu().numpy()
        
        if 'wsi' in output:
            attention['wsi_tiles'] = output['wsi'].cpu().numpy()
        
        if 'rna' in output:
            attention['genes'] = output['rna'].cpu().numpy()
        
        return attention
    
    def visualize_wsi_attention(
        self,
        slide_path: Path,
        tile_coordinates: np.ndarray,
        attention_weights: np.ndarray,
        output_path: Path,
        thumbnail_size: Tuple[int, int] = (2048, 2048)
    ):
        """
        Create WSI attention heatmap overlay.
        """
        
        slide = openslide.OpenSlide(str(slide_path))
        
        # Get thumbnail
        thumb = slide.get_thumbnail(thumbnail_size)
        thumb = np.array(thumb.convert('RGB'))
        
        # Create attention map at thumbnail resolution
        scale_x = thumbnail_size[0] / slide.dimensions[0]
        scale_y = thumbnail_size[1] / slide.dimensions[1]
        
        attention_map = np.zeros(thumbnail_size[::-1])
        
        for (x, y, _), attn in zip(tile_coordinates, attention_weights):
            thumb_x = int(x * scale_x)
            thumb_y = int(y * scale_y)
            tile_size_thumb = int(256 * scale_x)
            
            attention_map[
                thumb_y:thumb_y + tile_size_thumb,
                thumb_x:thumb_x + tile_size_thumb
            ] = attn
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original slide
        axes[0].imshow(thumb)
        axes[0].set_title('Original H&E')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(thumb)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        slide.close()
    
    def visualize_modality_attention(
        self,
        attention_weights: np.ndarray,
        output_path: Path
    ):
        """
        Create cross-modal attention bar plot.
        """
        
        modalities = ['WSI', 'RNA', 'Methylation', 'Mutations', 'Clinical']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
        bars = ax.bar(modalities, attention_weights, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, attention_weights):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Attention Weight', fontsize=14)
        ax.set_title('Cross-Modal Attention Distribution', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(attention_weights) * 1.2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_gene_importance(
        self,
        gene_attention: np.ndarray,
        gene_names: list,
        output_path: Path,
        top_k: int = 30
    ):
        """
        Create gene importance plot.
        """
        
        # Get top genes
        top_indices = np.argsort(gene_attention)[-top_k:][::-1]
        top_genes = [gene_names[i] for i in top_indices]
        top_weights = gene_attention[top_indices]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(top_genes))
        bars = ax.barh(y_pos, top_weights, color='#3498DB', edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_title(f'Top {top_k} Genes by Attention Weight', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
```

---

## 5.2 SHAP Analysis

```python
# src/explainability/shap_analysis.py

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, List

class SHAPAnalyzer:
    """
    SHAP-based feature importance analysis for MOSAIC.
    """
    
    def __init__(self, model, background_data: Dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.background_data = background_data
    
    def compute_modality_importance(
        self,
        test_data: Dict,
        n_samples: int = 50
    ) -> Dict[str, float]:
        """
        Compute feature importance via modality ablation.
        
        More practical than SHAP for multi-modal models.
        """
        
        modalities = ['wsi_features', 'rna', 'methylation', 
                      'mutation_binary', 'clinical']
        
        # Baseline prediction (all modalities)
        with torch.no_grad():
            baseline_output = self.model(**{k: v.to(self.device) for k, v in test_data.items()})
            baseline_risk = baseline_output['risk_score'].cpu().numpy()
        
        importance = {}
        
        for modality in modalities:
            # Create ablated input (zero out modality)
            ablated_data = {k: v.clone() if torch.is_tensor(v) else v 
                          for k, v in test_data.items()}
            ablated_data[modality] = torch.zeros_like(test_data[modality])
            
            # Predict with ablation
            with torch.no_grad():
                ablated_output = self.model(**{k: v.to(self.device) for k, v in ablated_data.items()})
                ablated_risk = ablated_output['risk_score'].cpu().numpy()
            
            # Importance = change in prediction
            importance[modality] = np.abs(baseline_risk - ablated_risk).mean()
        
        # Normalize
        total = sum(importance.values())
        importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def plot_modality_importance(
        self,
        importance: Dict[str, float],
        output_path: str
    ):
        """Plot modality importance."""
        
        labels = {
            'wsi_features': 'Histopathology',
            'rna': 'Gene Expression',
            'methylation': 'DNA Methylation',
            'mutation_binary': 'Mutations',
            'clinical': 'Clinical'
        }
        
        names = [labels[k] for k in importance.keys()]
        values = list(importance.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
        bars = ax.barh(names, values, color=colors)
        
        ax.set_xlabel('Relative Importance', fontsize=12)
        ax.set_title('Modality Importance for Survival Prediction', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2%}', va='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
```

---

# PHASE 6: EVALUATION & VALIDATION

## 6.1 Comprehensive Metrics

```python
# src/evaluation/metrics.py

import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple

class SurvivalMetrics:
    """
    Comprehensive evaluation metrics for survival prediction.
    """
    
    @staticmethod
    def compute_c_index(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Compute concordance index (C-index).
        
        Interpretation:
        - 0.5 = random prediction
        - 1.0 = perfect concordance
        - >0.7 = good model
        - >0.8 = excellent model
        """
        return concordance_index(times, -risk_scores, events)
    
    @staticmethod
    def compute_time_dependent_auc(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        eval_times: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time-dependent AUC at specific time points.
        
        Common eval_times for HNSC: 365, 730, 1095, 1825 days (1, 2, 3, 5 years)
        """
        
        if eval_times is None:
            eval_times = np.array([365, 730, 1095, 1825])
        
        # Create survival object for sksurv
        survival_train = Surv.from_arrays(events.astype(bool), times)
        survival_test = Surv.from_arrays(events.astype(bool), times)
        
        # Filter eval_times to be within observed range
        max_time = times[events == 1].max() if events.sum() > 0 else times.max()
        eval_times = eval_times[eval_times < max_time]
        
        if len(eval_times) == 0:
            return np.array([]), np.array([])
        
        auc, mean_auc = cumulative_dynamic_auc(
            survival_train, survival_test, risk_scores, eval_times
        )
        
        return eval_times, auc
    
    @staticmethod
    def compute_integrated_brier_score(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Compute Integrated Brier Score (IBS).
        
        Lower is better. Random model ≈ 0.25.
        """
        
        from sksurv.metrics import brier_score
        
        # Need survival function estimates, not just risk scores
        # This requires additional model components (survival curve estimation)
        # For simplicity, return placeholder
        # In practice, use pycox or discrete-time models for proper IBS
        
        return None  # Implement with survival curve estimates
    
    @staticmethod
    def compute_all_metrics(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray
    ) -> Dict[str, float]:
        """Compute all survival metrics."""
        
        metrics = {}
        
        # C-index
        metrics['c_index'] = SurvivalMetrics.compute_c_index(risk_scores, times, events)
        
        # Time-dependent AUC
        eval_times, aucs = SurvivalMetrics.compute_time_dependent_auc(
            risk_scores, times, events
        )
        
        time_labels = {365: '1yr', 730: '2yr', 1095: '3yr', 1825: '5yr'}
        for t, auc in zip(eval_times, aucs):
            label = time_labels.get(t, f'{t}d')
            metrics[f'auc_{label}'] = auc
        
        if len(aucs) > 0:
            metrics['auc_mean'] = np.mean(aucs)
        
        return metrics
```

---

## 6.2 Risk Stratification & Kaplan-Meier Analysis

```python
# src/evaluation/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import List, Optional

class SurvivalVisualization:
    """
    Visualization tools for survival analysis results.
    """
    
    @staticmethod
    def plot_kaplan_meier_by_risk_group(
        risk_scores: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        n_groups: int = 2,
        output_path: str = None,
        group_labels: Optional[List[str]] = None
    ):
        """
        Plot Kaplan-Meier curves stratified by risk group.
        """
        
        # Create risk groups
        if n_groups == 2:
            threshold = np.median(risk_scores)
            groups = (risk_scores > threshold).astype(int)
            if group_labels is None:
                group_labels = ['Low Risk', 'High Risk']
        else:
            quantiles = np.percentile(risk_scores, np.linspace(0, 100, n_groups + 1)[1:-1])
            groups = np.digitize(risk_scores, quantiles)
            if group_labels is None:
                group_labels = [f'Risk Group {i+1}' for i in range(n_groups)]
        
        # Colors
        colors = ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'][:n_groups]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(n_groups):
            mask = groups == i
            kmf = KaplanMeierFitter()
            kmf.fit(times[mask], events[mask], label=group_labels[i])
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i], linewidth=2)
        
        # Log-rank test
        if n_groups == 2:
            result = logrank_test(
                times[groups == 0], times[groups == 1],
                events[groups == 0], events[groups == 1]
            )
            p_value = result.p_value
            ax.text(0.7, 0.9, f'Log-rank p = {p_value:.2e}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_calibration(
        predicted_survival: np.ndarray,  # At specific time point
        observed_events: np.ndarray,
        observed_times: np.ndarray,
        eval_time: float,
        n_bins: int = 10,
        output_path: str = None
    ):
        """
        Plot calibration curve (predicted vs observed survival).
        """
        
        # Bin predictions
        bins = np.percentile(predicted_survival, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(predicted_survival, bins[:-1]) - 1
        
        predicted_means = []
        observed_means = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_means.append(predicted_survival[mask].mean())
                
                # Kaplan-Meier estimate for this bin
                kmf = KaplanMeierFitter()
                kmf.fit(observed_times[mask], observed_events[mask])
                
                # Get survival probability at eval_time
                if eval_time <= observed_times[mask].max():
                    observed_surv = kmf.survival_function_at_times(eval_time).values[0]
                else:
                    observed_surv = kmf.survival_function_.iloc[-1].values[0]
                
                observed_means.append(observed_surv)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Actual calibration
        ax.scatter(predicted_means, observed_means, s=100, c='#3498DB', edgecolors='black', linewidth=1.5)
        ax.plot(predicted_means, observed_means, 'b-', linewidth=2, alpha=0.7, label='Model')
        
        ax.set_xlabel('Predicted Survival Probability', fontsize=12)
        ax.set_ylabel('Observed Survival Probability', fontsize=12)
        ax.set_title(f'Calibration Plot at {eval_time} days', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()
```

---

# PHASE 7: ABLATION STUDIES

## 7.1 Modality Ablation

```python
# scripts/ablation_studies.py

"""
Ablation studies to understand contribution of each modality.

Studies:
1. Leave-one-out: Remove one modality at a time
2. Single modality: Train with only one modality
3. Progressive addition: Add modalities one by one
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List
import json

class AblationStudies:
    """
    Systematic ablation experiments.
    """
    
    MODALITY_ORDER = ['clinical', 'wsi', 'rna', 'methylation', 'mutations']
    
    def __init__(self, trainer_class, model_config: Dict, data_config: Dict):
        self.trainer_class = trainer_class
        self.model_config = model_config
        self.data_config = data_config
        self.results = []
    
    def run_leave_one_out(self, train_data, val_data) -> pd.DataFrame:
        """
        Train model leaving out one modality at a time.
        """
        
        results = []
        
        # Full model (baseline)
        print("Training full model...")
        full_metrics = self._train_and_evaluate(train_data, val_data, exclude_modalities=[])
        results.append({'experiment': 'Full Model', **full_metrics})
        
        # Leave one out
        for modality in self.MODALITY_ORDER:
            print(f"Training without {modality}...")
            metrics = self._train_and_evaluate(train_data, val_data, exclude_modalities=[modality])
            results.append({'experiment': f'Without {modality}', **metrics})
        
        return pd.DataFrame(results)
    
    def run_single_modality(self, train_data, val_data) -> pd.DataFrame:
        """
        Train model with only one modality at a time.
        """
        
        results = []
        
        for modality in self.MODALITY_ORDER:
            print(f"Training with only {modality}...")
            include = [modality]
            if modality != 'clinical':
                include.append('clinical')  # Always include clinical for survival labels
            
            exclude = [m for m in self.MODALITY_ORDER if m not in include]
            metrics = self._train_and_evaluate(train_data, val_data, exclude_modalities=exclude)
            results.append({'experiment': f'Only {modality}', **metrics})
        
        return pd.DataFrame(results)
    
    def run_progressive_addition(self, train_data, val_data) -> pd.DataFrame:
        """
        Progressively add modalities in order of expected importance.
        """
        
        # Order: Clinical → WSI → RNA → Methylation → Mutations
        addition_order = ['clinical', 'wsi', 'rna', 'methylation', 'mutations']
        
        results = []
        included = []
        
        for modality in addition_order:
            included.append(modality)
            exclude = [m for m in self.MODALITY_ORDER if m not in included]
            
            print(f"Training with {included}...")
            metrics = self._train_and_evaluate(train_data, val_data, exclude_modalities=exclude)
            results.append({'experiment': ' + '.join(included), **metrics})
        
        return pd.DataFrame(results)
    
    def _train_and_evaluate(
        self,
        train_data,
        val_data,
        exclude_modalities: List[str]
    ) -> Dict[str, float]:
        """Train model and return metrics."""
        
        # Modify model config to exclude modalities
        config = self.model_config.copy()
        
        for modality in exclude_modalities:
            if modality == 'wsi':
                config['wsi_input_dim'] = 0
            elif modality == 'rna':
                config['num_genes_rna'] = 0
            elif modality == 'methylation':
                config['num_genes_meth'] = 0
            elif modality == 'mutations':
                config['num_genes_mut'] = 0
                config['num_drivers'] = 0
        
        # Train (implement actual training loop)
        # ...
        
        # Placeholder metrics
        return {
            'c_index': np.random.uniform(0.6, 0.8),
            'auc_1yr': np.random.uniform(0.6, 0.85),
            'auc_3yr': np.random.uniform(0.6, 0.85)
        }
    
    def plot_ablation_results(self, results: pd.DataFrame, output_path: str):
        """Visualize ablation results."""
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(results))
        width = 0.25
        
        ax.bar(x - width, results['c_index'], width, label='C-index', color='#3498DB')
        ax.bar(x, results['auc_1yr'], width, label='AUC (1yr)', color='#2ECC71')
        ax.bar(x + width, results['auc_3yr'], width, label='AUC (3yr)', color='#E74C3C')
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results['experiment'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0.5, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
```

---

# HYPERPARAMETER RECOMMENDATIONS

```yaml
# configs/model_config.yaml

# Encoder dimensions
wsi_input_dim: 1024          # UNI output dimension
num_genes_rna: 7923          # After variance filtering
num_genes_meth: 10057        # After variance filtering
num_genes_mut: 1500          # Genes with mutations
num_drivers: 15              # HNSC driver genes
num_clinical: 15             # Clinical features

# Architecture
embed_dim: 256               # Common embedding dimension
hidden_dim: 512              # Hidden layers
num_latents: 16              # Perceiver latent queries
num_heads: 8                 # Attention heads
num_fusion_layers: 2         # Perceiver depth
dropout: 0.1                 # Dropout rate

# Training
learning_rate: 0.0001        # Base learning rate
encoder_lr_mult: 0.1         # Lower LR for encoders
weight_decay: 0.00001        # L2 regularization
batch_size: 16               # Limited by WSI memory
max_epochs: 100              # With early stopping
warmup_epochs: 5             # LR warmup
patience: 15                 # Early stopping patience

# Data
max_tiles: 4000              # Max tiles per WSI
tile_size: 256               # Tile dimensions
target_mpp: 0.5              # Target resolution (20x)

# Cross-validation
n_folds: 5                   # Stratified K-fold
seed: 42                     # Reproducibility
```

---

# EXPECTED RESULTS

## Baseline Comparisons (Expected C-index)

| Model | C-index | Notes |
|-------|---------|-------|
| Clinical Only (Cox) | 0.60-0.65 | TNM stage, age |
| WSI Only (ABMIL) | 0.62-0.68 | Image features |
| RNA Only (MLP) | 0.65-0.70 | Gene expression |
| Clinical + RNA | 0.68-0.72 | Common baseline |
| **MOSAIC (Full)** | **0.75-0.85** | All modalities |

## Risk Stratification (Expected)

| Group | 3-Year Survival | Hazard Ratio |
|-------|-----------------|--------------|
| Low Risk | 70-80% | Reference |
| High Risk | 30-45% | 2.5-4.0 |

## Modality Importance (Expected)

| Modality | Relative Importance |
|----------|---------------------|
| Histopathology | 25-35% |
| Gene Expression | 20-30% |
| Clinical | 15-25% |
| Methylation | 10-20% |
| Mutations | 5-15% |

---

# COMMON ISSUES & SOLUTIONS

## Issue 1: Out of Memory (WSI Features)
```python
# Solution: Process tiles in chunks
chunk_size = 500
for i in range(0, n_tiles, chunk_size):
    chunk = tiles[i:i+chunk_size]
    features = model(chunk)
    save_features(features, f"chunk_{i}.h5")
```

## Issue 2: Imbalanced Censoring
```python
# Solution: Stratified sampling
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_idx, val_idx in skf.split(X, events):  # Stratify by event
    ...
```

## Issue 3: Missing Modalities for Some Patients
```python
# Solution: Modality masking + mean imputation
def handle_missing(batch, modality_masks):
    for i, mask in enumerate(modality_masks):
        if not mask[i]:
            batch[modality][i] = modality_means[modality]
    return batch
```

## Issue 4: Overfitting with Small Sample Size
```python
# Solution: Strong regularization + early stopping
model_config = {
    'dropout': 0.3,  # Increase dropout
    'weight_decay': 1e-4,  # Stronger L2
}
trainer = Trainer(
    early_stopping=EarlyStopping(patience=10, monitor='val_c_index')
)
```

---

# KEY REFERENCES

## Papers to Cite

1. **PORPOISE** - Chen et al., "Pan-cancer integrative histology-genomic analysis via multimodal deep learning", Cancer Cell 2022
2. **MCAT** - Chen et al., "Multimodal Co-Attention Transformer for Survival Prediction", ICCV 2021
3. **UNI** - Chen et al., "Towards a general-purpose foundation model for computational pathology", Nature Medicine 2024
4. **DeepSurv** - Katzman et al., "DeepSurv: personalized treatment recommender system", BMC Medical Research Methodology 2018
5. **Perceiver** - Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
6. **CLAM** - Lu et al., "Data-efficient and weakly supervised computational pathology", Nature Biomedical Engineering 2021

## Repositories

- CLAM: https://github.com/mahmoodlab/CLAM
- HIPT: https://github.com/mahmoodlab/HIPT
- UNI: https://huggingface.co/MahmoodLab/UNI
- pycox: https://github.com/havakv/pycox

---

# SUCCESS CRITERIA

## Minimum Viable Results
- [ ] C-index > 0.70 on held-out test set
- [ ] Significant risk stratification (log-rank p < 0.01)
- [ ] Attention visualizations showing clinically meaningful patterns

## Target Results
- [ ] C-index > 0.78
- [ ] AUC(3yr) > 0.80
- [ ] Clear modality importance ranking
- [ ] Interpretable attention highlighting known prognostic features

## Stretch Goals
- [ ] C-index > 0.82
- [ ] Novel prognostic marker discovery
- [ ] External validation on independent HNSC cohort
- [ ] Publication in Nature Communications or higher

---

*MOSAIC Implementation Plan v1.0*
*Generated: January 29, 2026*
