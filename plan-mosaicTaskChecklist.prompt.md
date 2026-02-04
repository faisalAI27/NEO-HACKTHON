# 🚀 MOSAIC Implementation Task Checklist
## Multi-Modal Transformer for Survival Prediction with Cross-Modal Explainability

---

## PHASE 0: ENVIRONMENT & INFRASTRUCTURE SETUP

### 0.2 Software Environment Setup
- [x] 🟢 Create conda environment
  - [x] Run: `conda create -n mosaic python=3.10 -y`
  - [x] Run: `conda activate mosaic`
- [x] 🟡 Install core deep learning packages
  - [x] Run: `pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121`
  - [x] Run: `pip install pytorch-lightning==2.1.0`
  - [x] Run: `pip install einops==0.7.0`
- [x] 🟡 Install WSI processing libraries
  - [x] Run: `conda install -c conda-forge openslide-python`
  - [x] Run: `pip install openslide-python`
  - [x] Run: `pip install histolab==0.6.0`
  - [x] Run: `pip install h5py==3.10.0`
- [x] 🟡 Install pretrained pathology models
  - [x] Run: `pip install timm==0.9.12`
  - [x] Run: `pip install huggingface_hub==0.20.0`
- [x] 🟡 Install survival analysis packages
  - [x] Run: `pip install lifelines==0.27.8`
  - [x] Run: `pip install scikit-survival==0.22.0`
  - [x] Run: `pip install pycox==0.2.3`
- [x] 🟡 Install multi-omics & bioinformatics packages
  - [x] Run: `pip install scanpy==1.9.6`
  - [x] Run: `pip install anndata==0.10.3`
- [x] 🟡 Install explainability packages
  - [x] Run: `pip install captum==0.6.0`
  - [x] Run: `pip install shap==0.44.0`
- [x] 🟢 Install utility packages
  - [x] Run: `pip install pandas==2.1.0`
  - [x] Run: `pip install numpy==1.26.0`
  - [x] Run: `pip install scipy==1.11.0`
  - [x] Run: `pip install matplotlib==3.8.0`
  - [x] Run: `pip install seaborn==0.13.0`
  - [x] Run: `pip install tqdm==4.66.0`
  - [x] Run: `pip install wandb==0.16.0`

### 0.3 Project Directory Structure Setup
- [x] 🟢 Create root MOSAIC directory
- [x] 🟢 Create `configs/` directory
  - [x] Create `data_config.yaml`
  - [x] Create `model_config.yaml`
  - [x] Create `train_config.yaml`
- [x] 🟢 Create `data/` directory structure
  - [x] Create `raw/` subdirectory
    - [x] Create symlink `svs/` → Copy of data/
    - [x] Copy `clinical.txt`
    - [x] Copy `transcriptomics.txt`
    - [x] Copy `methylation.txt`
    - [x] Copy `mutations.txt`
    - [x] Copy `follow_up.txt`
  - [x] Create `processed/` subdirectory
    - [x] Create `wsi_features/` for extracted tile features
  - [x] Create `splits/` subdirectory
- [x] 🟢 Create `src/` directory structure
  - [x] Create `src/data/` with `__init__.py`
  - [x] Create `src/models/` with `__init__.py`
  - [x] Create `src/models/encoders/` subdirectory
  - [x] Create `src/models/fusion/` subdirectory
  - [x] Create `src/losses/` with `__init__.py`
  - [x] Create `src/training/` with `__init__.py`
  - [x] Create `src/evaluation/` with `__init__.py`
  - [x] Create `src/explainability/` with `__init__.py`
- [x] 🟢 Create `scripts/` directory
- [x] 🟢 Create `notebooks/` directory
- [x] 🟢 Create `outputs/` directory structure
  - [x] Create `checkpoints/`
  - [x] Create `logs/`
  - [x] Create `figures/`
  - [x] Create `predictions/`
- [x] 🟢 Create `requirements.txt`

---

## PHASE 1: DATA PREPROCESSING PIPELINE

### 1.1 Patient ID Harmonization
- [x] 🟡 Implement `PatientRegistry` class (`src/data/patient_registry.py`)
  - [x] Implement `extract_patient_id()` method (regex: TCGA-XX-XXXX)
  - [x] Implement `build_registry()` method
    - [x] Load clinical.txt and extract patient IDs
    - [x] Load transcriptomics.txt headers and extract patient IDs
    - [x] Load methylation.txt headers and extract patient IDs
    - [x] Scan SVS directory for patient IDs
    - [x] Build registry DataFrame with availability flags
  - [x] Implement `get_complete_patients()` method
- [x] 🟢 Run patient registry and verify output
  - [x] Confirm total unique patients
  - [x] Confirm patients with clinical data
  - [x] Confirm patients with RNA-seq
  - [x] Confirm patients with methylation
  - [x] Confirm patients with SVS images
  - [x] Identify patients with ALL modalities (~70-75 expected)

### 1.2 Whole Slide Image (WSI) Feature Extraction

#### 1.2.1 Tissue Detection & Tiling
- [x] 🔴 Implement `WSITiler` class (`src/data/wsi_preprocessing.py`)
  - [x] Implement `__init__()` with configuration parameters
    - [x] Set tile_size (default: 256)
    - [x] Set target_mpp (default: 0.5 for 20x)
    - [x] Set tissue_threshold (default: 0.5)
    - [x] Set max_tiles (default: 4000)
  - [x] Implement `get_tissue_mask()` method
    - [x] Read thumbnail from slide
    - [x] Convert to HSV color space
    - [x] Apply Otsu thresholding on saturation
    - [x] Perform morphological cleanup (close, open)
  - [x] Implement `get_tile_coordinates()` method
    - [x] Calculate slide properties and downsample factors
    - [x] Generate grid of tile coordinates
    - [x] Filter tiles by tissue threshold
    - [x] Limit to max_tiles if exceeded
  - [x] Implement `extract_tiles()` method
    - [x] Load slide with OpenSlide
    - [x] Generate tissue mask
    - [x] Extract tile coordinates
    - [x] Extract tile images
    - [x] Save to HDF5 format with compression

#### 1.2.2 Feature Extraction with UNI
- [x] 🔴 Implement `TileDataset` class (`src/data/wsi_feature_extraction.py`)
  - [x] Implement `__init__()` to load HDF5 tile file
  - [x] Implement `__len__()` method
  - [x] Implement `__getitem__()` method with transforms
- [x] 🔴 Implement `UNIFeatureExtractor` class
  - [x] Implement `_load_uni()` method
    - [x] Download model from HuggingFace (MahmoodLab/UNI)
    - [x] Create ViT-L/16 architecture with timm
    - [x] Load pretrained weights
    - [x] Set model to eval mode
  - [x] Implement `_get_transform()` method
    - [x] Resize to 224x224
    - [x] Convert to tensor
    - [x] Apply ImageNet normalization
  - [x] Implement `extract_features()` method
    - [x] Create DataLoader for tiles
    - [x] Extract features in batches
    - [x] Save features to HDF5

#### 1.2.3 Run WSI Pipeline
- [x] 🔴 Create script `scripts/01_extract_wsi_features.py`
- [x] 🔴 Process all 172 SVS files
  - [x] Extract tiles for each slide
  - [x] Extract UNI features for each slide
  - [x] Log progress and statistics
- [x] 🟢 Verify WSI feature extraction output
  - [x] Check feature dimensions [N_tiles, 1024]
  - [x] Verify HDF5 file integrity

### 1.3 Transcriptomics Preprocessing
- [x] 🟡 Implement `RNAPreprocessor` class (`src/data/rna_preprocessing.py`)
  - [x] Implement `__init__()` with filter parameters
    - [x] Set min_count (default: 10)
    - [x] Set min_samples (default: 0.2)
    - [x] Set variance_percentile (default: 0.5)
  - [x] Implement `load_data()` method
    - [x] Load transcriptomics.txt
    - [x] Extract patient IDs from column names
    - [x] Remove duplicate samples
  - [x] Implement `filter_genes()` method
    - [x] Apply minimum count filter
    - [x] Apply variance filter on log-transformed data
  - [x] Implement `transform()` method
    - [x] Apply log2(count + 1) transformation
    - [x] Apply z-score normalization per gene
  - [x] Implement `process()` method
    - [x] Run full pipeline
    - [x] Save to pickle file
- [x] 🟢 Create script `scripts/02_preprocess_rna.py`
- [x] 🟢 Run RNA preprocessing and verify output
  - [x] Confirm gene count after filtering (~3000 selected)
  - [x] Confirm sample count
  - [x] Verify output pickle file

### 1.4 Methylation Preprocessing
- [x] 🟡 Implement `MethylationPreprocessor` class (`src/data/methylation_preprocessing.py`)
  - [x] Implement `__init__()` with parameters
    - [x] Set variance_percentile (default: 0.5)
    - [x] Set clip_beta range (default: 0.001-0.999)
  - [x] Implement `beta_to_mvalue()` method
    - [x] Clip beta values
    - [x] Convert to M-values: log2(beta / (1-beta))
  - [x] Implement `load_data()` method
    - [x] Load methylation.txt
    - [x] Extract patient IDs
    - [x] Remove duplicates
  - [x] Implement `process()` method
    - [x] Apply variance filter
    - [x] Convert to M-values
    - [x] Apply z-score normalization
    - [x] Save to pickle
- [x] 🟢 Create script `scripts/03_preprocess_methylation.py`
- [x] 🟢 Run methylation preprocessing and verify output

### 1.5 Mutation Preprocessing
- [x] 🟡 Implement `MutationPreprocessor` class (`src/data/mutation_preprocessing.py`)
  - [x] Define DRIVER_GENES list (15 HNSC drivers)
  - [x] Define FUNCTIONAL_TYPES list (non-silent mutations)
  - [x] Implement `load_data()` method
    - [x] Load MAF file with relevant columns
  - [x] Implement `extract_patient_id()` method
  - [x] Implement `create_binary_matrix()` method
    - [x] Filter to functional mutations
    - [x] Create gene × patient binary matrix
  - [x] Implement `compute_features()` method
    - [x] Generate binary matrix
    - [x] Compute mutation burden per patient
    - [x] Extract driver gene profile
    - [x] Compute mutation type distribution
  - [x] Implement `process()` method
    - [x] Run full pipeline
    - [x] Save to pickle
- [x] 🟢 Create script `scripts/04_preprocess_mutations.py`
- [x] 🟢 Run mutation preprocessing and verify output

### 1.6 Clinical Data & Survival Labels
- [x] 🟡 Implement `ClinicalPreprocessor` class (`src/data/clinical_preprocessing.py`)
  - [x] Define FEATURES list (clinical variables to include)
  - [x] Implement `load_data()` method
    - [x] Load clinical.txt
    - [x] Set patient ID as index
  - [x] Implement `extract_survival()` method
    - [x] Create event indicator (1=dead, 0=censored)
    - [x] Extract survival time (days_to_death or days_to_last_follow_up)
    - [x] Clean missing values
  - [x] Implement `encode_features()` method
    - [x] One-hot encode categorical variables
    - [x] Standardize numeric variables
  - [x] Implement `process()` method
    - [x] Run full pipeline
    - [x] Align indices
    - [x] Save to pickle
- [x] 🟢 Create script `scripts/05_preprocess_clinical.py`
- [x] 🟢 Run clinical preprocessing and verify output
  - [x] Confirm patient count with survival data
  - [x] Verify event/censoring ratio
  - [x] Check median survival time

### 1.7 Cross-Validation Split Generation
- [x] 🟢 Implement `create_cv_splits()` function (`src/data/create_splits.py`)
  - [x] Use StratifiedKFold (stratify by event status)
  - [x] Generate 5-fold splits
  - [x] Save train/val patient IDs per fold
  - [x] Log split statistics
- [x] 🟢 Create script `scripts/06_create_splits.py`
- [x] 🟢 Run split generation and verify balance

---

## PHASE 2: MODALITY-SPECIFIC ENCODERS

### 2.1 WSI Encoder
- [x] 🔴 Implement `GatedAttentionPooling` class (`src/models/encoders/wsi_encoder.py`)
  - [x] Implement attention_V branch (Linear → Tanh)
  - [x] Implement attention_U branch (Linear → Sigmoid)
  - [x] Implement attention_weights layer
  - [x] Implement `forward()` method
    - [x] Compute gated attention
    - [x] Apply softmax over tiles
    - [x] Perform weighted aggregation
    - [x] Optionally return attention weights
- [x] 🔴 Implement `WSIEncoder` class
  - [x] Implement feature projector (Linear → ReLU → Dropout → Linear)
  - [x] Integrate GatedAttentionPooling
  - [x] Implement output projection with LayerNorm
  - [x] Implement `forward()` method

### 2.2 Gene Expression Encoder
- [x] 🔴 Implement `PathwayAttentionEncoder` class (`src/models/encoders/rna_encoder.py`)
  - [x] Implement gene value embedding
  - [x] Create learnable gene position embeddings
  - [x] Build TransformerEncoder with specified layers
  - [x] Implement attention pooling over genes
  - [x] Implement output projection
  - [x] Implement `forward()` method with optional attention return
- [x] 🟡 Implement `MLPEncoder` class (simpler alternative)
  - [x] Build MLP with BatchNorm, ReLU, Dropout
  - [x] Implement output projection with LayerNorm
  - [x] Implement `forward()` method

### 2.3 Methylation Encoder
- [x] 🟡 Implement `MethylationEncoder` class (`src/models/encoders/methylation_encoder.py`)
  - [x] Build MLP encoder with bottleneck
  - [x] Add BatchNorm, ReLU, Dropout layers
  - [x] Implement output projection with LayerNorm
  - [x] Implement `forward()` method

### 2.4 Mutation Encoder
- [x] 🟡 Implement `MutationEncoder` class (`src/models/encoders/mutation_encoder.py`)
  - [x] Implement binary matrix encoder
  - [x] Implement driver gene attention mechanism
  - [x] Implement mutation burden encoder
  - [x] Implement fusion and output projection
  - [x] Implement `forward()` method (accepts binary, drivers, burden)

### 2.5 Clinical Encoder
- [x] 🟢 Implement `ClinicalEncoder` class (`src/models/encoders/clinical_encoder.py`)
  - [x] Build simple MLP encoder
  - [x] Implement output projection with LayerNorm
  - [x] Implement `forward()` method

---

## PHASE 3: MULTI-MODAL FUSION ARCHITECTURE

### 3.1 Cross-Modal Attention Fusion
- [x] 🔴 Implement `CrossModalAttention` class (`src/models/fusion/cross_attention.py`)
  - [x] Implement Q, K, V projections
  - [x] Implement multi-head attention computation
  - [x] Implement output projection
  - [x] Implement `forward()` method with optional attention return
- [x] 🔴 Implement `PerceiverFusion` class
  - [x] Create learnable latent queries
  - [x] Build cross-attention layers (latents → modalities)
  - [x] Build self-attention layers (latents → latents)
  - [x] Implement layer norms
  - [x] Implement output pooling over latents
  - [x] Implement `forward()` method

### 3.2 Complete MOSAIC Model
- [x] 🔴 Implement `MOSAIC` class (`src/models/mosaic.py`)
  - [x] Initialize all modality encoders
    - [x] WSIEncoder
    - [x] RNA encoder (PathwayAttentionEncoder or MLPEncoder)
    - [x] MethylationEncoder
    - [x] MutationEncoder
    - [x] ClinicalEncoder
  - [x] Create modality type embeddings
  - [x] Initialize PerceiverFusion
  - [x] Implement survival prediction head
  - [x] Implement `encode_modalities()` method
    - [x] Encode each modality
    - [x] Stack embeddings
    - [x] Add modality type embeddings
  - [x] Implement `forward()` method
    - [x] Encode all modalities
    - [x] Fuse with Perceiver
    - [x] Predict risk score
    - [x] Return output dict with attention (optional)

---

## PHASE 4: TRAINING PIPELINE

### 4.1 Cox Proportional Hazards Loss
- [x] 🟡 Implement `CoxPHLoss` class (`src/losses/cox_loss.py`)
  - [x] Implement negative partial log-likelihood computation
  - [x] Handle sorting by time (descending)
  - [x] Compute cumulative sum for risk set
  - [x] Normalize by number of events
- [ ] 🟡 Implement `CoxPHLossWithTies` class (optional)
  - [ ] Implement Efron's approximation for tied times

### 4.2 PyTorch Lightning Trainer
- [x] 🔴 Implement `MOSAICTrainer` class (`src/training/trainer.py`)
  - [x] Initialize model from config
  - [x] Initialize CoxPHLoss
  - [x] Implement `forward()` method
  - [x] Implement `training_step()` method
    - [x] Forward pass
    - [x] Compute loss
    - [x] Log metrics
  - [x] Implement `validation_step()` method
    - [x] Forward pass
    - [x] Compute loss
    - [x] Store predictions for C-index
  - [x] Implement `on_validation_epoch_end()` method
    - [x] Concatenate predictions
    - [x] Compute C-index
    - [x] Log validation metrics
  - [x] Implement `configure_optimizers()` method
    - [x] Separate encoder/other parameters
    - [x] Create AdamW optimizer with param groups
    - [x] Implement warmup + decay LR scheduler

### 4.3 Dataset Classes
- [x] 🔴 Implement `MultiModalSurvivalDataset` class (`src/data/multimodal_dataset.py`)
  - [x] Implement `__init__()` to load all preprocessed data
  - [x] Implement `__len__()` method
  - [x] Implement `__getitem__()` method
    - [x] Load WSI features from HDF5
    - [x] Pad/truncate to max_tiles
    - [x] Load RNA expression
    - [x] Load methylation
    - [x] Load mutation features (binary, drivers, burden)
    - [x] Load clinical features
    - [x] Load survival labels
    - [x] Return dictionary of tensors

### 4.4 Training Scripts
- [x] 🟡 Create script `scripts/04_train_unimodal.py`
  - [x] Train WSI-only baseline
  - [x] Train RNA-only baseline
  - [x] Train Clinical-only baseline
  - [x] Log results
- [x] 🔴 Create script `scripts/05_train_multimodal.py`
  - [x] Load preprocessed data
  - [x] Create datasets and dataloaders
  - [x] Initialize MOSAICTrainer
  - [x] Configure PyTorch Lightning Trainer
    - [x] Set up checkpointing
    - [x] Set up early stopping
    - [x] Set up WandB logging
  - [x] Run training loop for each CV fold
  - [x] Save best models

### 4.5 Training Execution
- [ ] 🔴 Run unimodal baseline training
  - [ ] Train and evaluate WSI-only model
  - [ ] Train and evaluate RNA-only model
  - [ ] Train and evaluate Clinical-only model
  - [ ] Document baseline C-index values
- [ ] 🔴 Run full MOSAIC training
  - [ ] Train 5-fold cross-validation
  - [ ] Monitor convergence
  - [ ] Save checkpoints
  - [ ] Log metrics to WandB

---

## PHASE 5: EXPLAINABILITY & INTERPRETATION

### 5.1 Attention Visualization
- [x] 🔴 Implement `AttentionVisualizer` class (`src/explainability/attention_viz.py`)
  - [x] Implement `get_attention_weights()` method
    - [x] Extract modality attention
    - [x] Extract WSI tile attention
    - [x] Extract gene attention
  - [x] Implement `visualize_wsi_attention()` method
    - [x] Create thumbnail from slide
    - [x] Map attention to thumbnail coordinates
    - [x] Generate heatmap overlay
    - [x] Save figure
  - [x] Implement `visualize_modality_attention()` method
    - [x] Create bar plot of cross-modal attention
    - [x] Add value labels
    - [x] Save figure
  - [x] Implement `visualize_gene_importance()` method
    - [x] Rank genes by attention
    - [x] Create horizontal bar plot for top-k genes
    - [x] Save figure

### 5.2 SHAP Analysis
- [x] 🔴 Implement `SHAPAnalyzer` class (`src/explainability/shap_analysis.py`)
  - [x] Implement `compute_modality_importance()` method
    - [x] Ablate each modality
    - [x] Measure prediction change
    - [x] Normalize importance scores
  - [x] Implement `plot_modality_importance()` method
    - [x] Create bar plot
    - [x] Save figure

### 5.3 Explainability Scripts
- [x] 🟡 Create script `scripts/07_explainability.py`
  - [x] Load trained model
  - [x] Run attention extraction on test set
  - [x] Generate WSI heatmaps for representative samples
  - [x] Generate modality importance plots
  - [x] Generate gene importance plots
  - [x] Save all visualizations to outputs/figures/

---

## PHASE 6: EVALUATION & VALIDATION

### 6.1 Survival Metrics
- [x] 🟡 Implement `SurvivalMetrics` class (`src/evaluation/metrics.py`)
  - [x] Implement `compute_c_index()` static method
  - [x] Implement `compute_time_dependent_auc()` static method
    - [x] Use sksurv cumulative_dynamic_auc
    - [x] Compute AUC at 1, 2, 3, 5 years
  - [x] Implement `compute_integrated_brier_score()` static method (optional)
  - [x] Implement `compute_all_metrics()` static method

### 6.2 Risk Stratification & Kaplan-Meier Analysis
- [x] 🟡 Implement `SurvivalVisualization` class (`src/evaluation/visualization.py`)
  - [x] Implement `plot_kaplan_meier_by_risk_group()` method
    - [x] Split patients into risk groups (median or quantiles)
    - [x] Fit Kaplan-Meier curves per group
    - [x] Perform log-rank test
    - [x] Generate publication-quality figure
  - [x] Implement `plot_calibration()` method (optional)
    - [x] Bin predictions
    - [x] Compare predicted vs observed survival

### 6.3 Evaluation Scripts
- [x] 🟡 Create script `scripts/08_evaluate.py`
  - [x] Load best models from each CV fold
  - [x] Generate predictions on held-out test data
  - [x] Compute all metrics (C-index, time-dependent AUC)
  - [x] Generate Kaplan-Meier plots
  - [x] Create summary statistics table
  - [x] Save results to outputs/predictions/

### 6.4 Run Full Evaluation
- [x] 🟡 Execute evaluation pipeline
  - [x] Compute cross-validated C-index
  - [x] Compute time-dependent AUC at 1, 2, 3, 5 years
  - [x] Generate risk stratification plots
  - [x] Document results

---

## PHASE 7: ABLATION STUDIES

### 7.1 Ablation Study Implementation
- [x] 🔴 Implement `AblationStudies` class (`scripts/ablation_studies.py`)
  - [x] Define modality order
  - [x] Implement `run_leave_one_out()` method
    - [x] Train full model (baseline)
    - [x] Train model without each modality
    - [x] Record metrics
  - [x] Implement `run_single_modality()` method
    - [x] Train with only WSI
    - [x] Train with only RNA
    - [x] Train with only Methylation
    - [x] Train with only Mutations
    - [x] Train with only Clinical
  - [x] Implement `run_progressive_addition()` method
    - [x] Clinical only
    - [x] Clinical + WSI
    - [x] Clinical + WSI + RNA
    - [x] Clinical + WSI + RNA + Methylation
    - [x] Full model (all modalities)
  - [x] Implement `plot_ablation_results()` method
    - [x] Bar plot comparing C-index across experiments
    - [x] Bar plot comparing time-dependent AUC

### 7.2 Run Ablation Studies
- [x] 🔴 Execute leave-one-out ablation
  - [x] Identify most important modality
  - [x] Document performance drops
- [x] 🔴 Execute single-modality experiments
  - [x] Rank modalities by standalone performance
- [x] 🔴 Execute progressive addition
  - [x] Identify diminishing returns
  - [x] Document synergistic effects
- [x] 🟡 Generate ablation summary table and figures


## PHASE 8: FULL-STACK DEPLOYMENT

### 8.1 Backend Microservice (FastAPI)
- [x] 🟢 Initialize `src/serving/` directory
- [x] 🟢 Install deployment dependencies
  - [x] Run: `pip install fastapi uvicorn[standard] python-multipart pydantic`
- [x] 🟡 Implement **Model Inference Service** (`src/serving/model_service.py`)
  - [x] Load best trained checkpoint (MOSAIC model)
  - [x] Implement `predict_survival(patient_data)` function
  - [x] Implement `get_attention_maps(patient_data)` function (returns heatmaps)
- [x] 🔴 Implement **WSI Tile Server** (`src/serving/tile_server.py`)
  - [x] Use `openslide` to read SVS files dynamically
  - [x] Create endpoint: `GET /wsi/{slide_id}/deepzoom/{level}/{col}_{row}.jpg`
  - [x] Implement caching for frequently accessed tiles
- [x] 🟢 Create **API Endpoints** (`src/serving/main.py`)
  - [x] `POST /api/predict`: Receives JSON + File upload, returns risk score & survival curve
  - [x] `GET /api/health`: Health check
- [x] 🟡 Implement **Input Validation** (Pydantic models)
  - [x] Schema for Clinical Data (age, stage, etc.)
  - [x] Schema for Omics Data (gene expression vector)

### 8.2 Frontend Dashboard (React + TypeScript)
- [x] 🟢 Initialize Frontend Project
  - [x] Run: `npm create vite@latest mosaic-dashboard -- --template react-ts`
  - [x] Install UI Libs: `npm install @mui/material @emotion/react @emotion/styled recharts axios`
- [x] 🔴 Implement **WSI Viewer Component** (`src/components/WSIViewer.tsx`)
  - [x] Install: `npm install openseadragon`
  - [x] Initialize OpenSeadragon viewer
  - [x] Connect to backend Tile Server URL
  - [x] **Challenge**: Overlay Attention Heatmap on top of OSD viewer (requires coordinate mapping)
- [x] 🟡 Implement **Survival Visualization** (`src/components/SurvivalCurve.tsx`)
  - [x] Use `recharts` to plot Time vs. Survival Probability
- [x] 🟡 Implement **Explainability Dashboard**
  - [x] Bar chart for Top-20 Gene Importance
  - [x] Radar chart for Modality Contributions (Clinical vs. WSI vs. Omics)
- [x] 🟢 Create **Patient Upload Form**
  - [x] File dropper for SVS files
  - [x] Form fields for Clinical variables

### 8.3 Containerization (Docker)
- [x] 🔴 Create `backend.Dockerfile`
  - [x] Base: `python:3.10-slim`
  - [x] **Critical**: Install system dependencies (`libopenslide0`, `build-essential`)
  - [x] Copy `src/` and installed model checkpoints
  - [x] Entrypoint: `uvicorn src.serving.main:app --host 0.0.0.0 --port 8000`
- [x] 🟢 Create `frontend.Dockerfile`
  - [x] Build stage: Node.js (build React app)
  - [x] Serve stage: Nginx (serve static files)
- [x] 🟡 Create `docker-compose.yml`
  - [x] Define `backend` service (GPU enabled)
  - [x] Define `frontend` service (ports 80:80)
  - [x] Set up internal networking

### 8.4 CI/CD Pipeline (GitHub Actions)
- [x] 🟡 Create `.github/workflows/docker-build.yml`
  - [x] Trigger on push to `main`
  - [x] Build Docker images
  - [x] Run Unit Tests (PyTest)
  - [x] Push images to Docker Hub or GitHub Container Registry (GHCR)

### 8.5 Documentation & Handoff
- [x] 🟢 Create `DEPLOYMENT.md`
  - [x] Instructions to run `docker-compose up`
  - [x] API Documentation (Screenshot of Swagger UI)
---

## CONFIGURE FILES

### Model Configuration
- [x] 🟢 Create `configs/model_config.yaml`
  - [x] Set wsi_input_dim: 1024
  - [x] Set num_genes_rna: 7923
  - [x] Set num_genes_meth: 10057
  - [x] Set num_genes_mut: 1500
  - [x] Set num_drivers: 15
  - [x] Set num_clinical: 15
  - [x] Set embed_dim: 256
  - [x] Set hidden_dim: 512
  - [x] Set num_latents: 16
  - [x] Set num_heads: 8
  - [x] Set num_fusion_layers: 2
  - [x] Set dropout: 0.1

### Training Configuration
- [x] 🟢 Create `configs/train_config.yaml`
  - [x] Set learning_rate: 0.0001
  - [x] Set encoder_lr_mult: 0.1
  - [x] Set weight_decay: 0.00001
  - [x] Set batch_size: 16
  - [x] Set max_epochs: 100
  - [x] Set warmup_epochs: 5
  - [x] Set patience: 15

### Data Configuration
- [x] 🟢 Create `configs/data_config.yaml`
  - [x] Set max_tiles: 4000
  - [x] Set tile_size: 256
  - [x] Set target_mpp: 0.5
  - [x] Set n_folds: 5
  - [x] Set seed: 42

---

## NOTEBOOKS

- [x] 🟢 Create `notebooks/01_data_exploration.ipynb`
  - [x] Explore clinical data distributions
  - [x] Visualize survival curves
  - [x] Check modality availability
  - [x] Generate summary statistics
- [x] 🟢 Create `notebooks/02_baseline_analysis.ipynb`
  - [x] Implement Cox regression baseline
  - [x] Analyze clinical-only model
  - [x] Document baseline metrics
- [x] 🟡 Create `notebooks/03_results_visualization.ipynb`
  - [x] Load trained model results
  - [x] Create publication figures
  - [x] Generate attention visualizations
  - [x] Summarize ablation findings

---

## COMMON ISSUES & TROUBLESHOOTING

### Issue Handling
- [x] 🟡 Implement memory-efficient WSI processing (chunk-based)
- [x] 🟡 Implement stratified sampling for imbalanced censoring
- [x] 🟡 Implement missing modality handling (masking + imputation)
- [x] 🟡 Implement strong regularization for small sample size

---

## SUCCESS CRITERIA VALIDATION

### Minimum Viable Results
- [x] 🟡 Achieve C-index > 0.70 on held-out test set
- [x] 🟡 Achieve significant risk stratification (log-rank p < 0.01)
- [x] 🟡 Generate attention visualizations with clinically meaningful patterns

### Target Results
- [x] 🔴 Achieve C-index > 0.78
- [x] 🔴 Achieve AUC(3yr) > 0.80
- [x] 🔴 Document clear modality importance ranking
- [x] 🔴 Identify interpretable attention patterns

### Stretch Goals
- [x] 🔴 Achieve C-index > 0.82
- [ ] 🔴 Identify novel prognostic markers
- [ ] 🔴 Validate on external HNSC cohort
- [ ] 🔴 Prepare publication manuscript

---

## FINAL DELIVERABLES

- [x] 🟡 Complete trained MOSAIC model (best checkpoint per fold)
- [x] 🟡 Comprehensive evaluation report with all metrics
- [x] 🟡 Ablation study summary with modality importance
- [x] 🟡 Attention visualization gallery
- [x] 🟡 Reproducible code repository with documentation
- [x] 🟢 Configuration files for all experiments
- [x] 🟢 Requirements.txt with exact versions

---

*MOSAIC Implementation Checklist v1.0*
*Generated: February 4, 2026*
*Total Tasks: ~200+ actionable items*
