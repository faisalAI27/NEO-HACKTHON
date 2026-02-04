# 🚀 MOSAIC Implementation Task Checklist
## Multi-Modal Transformer for Survival Prediction with Cross-Modal Explainability

---

## PHASE 0: ENVIRONMENT & INFRASTRUCTURE SETUP

### 0.1 Hardware Verification
- [ ] 🟢 Verify GPU availability (NVIDIA with 24GB+ VRAM recommended)
- [ ] 🟢 Confirm GPU model (A100 80GB / RTX 4090 24GB / RTX 3090 24GB minimum)
- [ ] 🟢 Check available RAM (128GB recommended for WSI processing)
- [ ] 🟢 Ensure storage space (2TB SSD recommended)
  - [ ] 🟢 Allocate ~500GB for raw SVS files
  - [ ] 🟢 Allocate ~50GB for extracted features
  - [ ] 🟢 Allocate ~100GB for checkpoints/outputs
- [ ] 🟢 Verify CPU cores (16+ cores for parallel tile extraction)

### 0.2 Software Environment Setup
- [ ] 🟢 Create conda environment
  - [ ] Run: `conda create -n mosaic python=3.10 -y`
  - [ ] Run: `conda activate mosaic`
- [ ] 🟡 Install core deep learning packages
  - [ ] Run: `pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121`
  - [ ] Run: `pip install pytorch-lightning==2.1.0`
  - [ ] Run: `pip install einops==0.7.0`
- [ ] 🟡 Install WSI processing libraries
  - [ ] Run: `conda install -c conda-forge openslide-python`
  - [ ] Run: `pip install openslide-python`
  - [ ] Run: `pip install histolab==0.6.0`
  - [ ] Run: `pip install h5py==3.10.0`
- [ ] 🟡 Install pretrained pathology models
  - [ ] Run: `pip install timm==0.9.12`
  - [ ] Run: `pip install huggingface_hub==0.20.0`
- [ ] 🟡 Install survival analysis packages
  - [ ] Run: `pip install lifelines==0.27.8`
  - [ ] Run: `pip install scikit-survival==0.22.0`
  - [ ] Run: `pip install pycox==0.2.3`
- [ ] 🟡 Install multi-omics & bioinformatics packages
  - [ ] Run: `pip install scanpy==1.9.6`
  - [ ] Run: `pip install anndata==0.10.3`
- [ ] 🟡 Install explainability packages
  - [ ] Run: `pip install captum==0.6.0`
  - [ ] Run: `pip install shap==0.44.0`
- [ ] 🟢 Install utility packages
  - [ ] Run: `pip install pandas==2.1.0`
  - [ ] Run: `pip install numpy==1.26.0`
  - [ ] Run: `pip install scipy==1.11.0`
  - [ ] Run: `pip install matplotlib==3.8.0`
  - [ ] Run: `pip install seaborn==0.13.0`
  - [ ] Run: `pip install tqdm==4.66.0`
  - [ ] Run: `pip install wandb==0.16.0`

### 0.3 Project Directory Structure Setup
- [ ] 🟢 Create root MOSAIC directory
- [ ] 🟢 Create `configs/` directory
  - [ ] Create `data_config.yaml`
  - [ ] Create `model_config.yaml`
  - [ ] Create `train_config.yaml`
- [ ] 🟢 Create `data/` directory structure
  - [ ] Create `raw/` subdirectory
    - [ ] Create symlink `svs/` → Copy of data/
    - [ ] Copy `clinical.txt`
    - [ ] Copy `transcriptomics.txt`
    - [ ] Copy `methylation.txt`
    - [ ] Copy `mutations.txt`
    - [ ] Copy `follow_up.txt`
  - [ ] Create `processed/` subdirectory
    - [ ] Create `wsi_features/` for extracted tile features
  - [ ] Create `splits/` subdirectory
- [ ] 🟢 Create `src/` directory structure
  - [ ] Create `src/data/` with `__init__.py`
  - [ ] Create `src/models/` with `__init__.py`
  - [ ] Create `src/models/encoders/` subdirectory
  - [ ] Create `src/models/fusion/` subdirectory
  - [ ] Create `src/losses/` with `__init__.py`
  - [ ] Create `src/training/` with `__init__.py`
  - [ ] Create `src/evaluation/` with `__init__.py`
  - [ ] Create `src/explainability/` with `__init__.py`
- [ ] 🟢 Create `scripts/` directory
- [ ] 🟢 Create `notebooks/` directory
- [ ] 🟢 Create `outputs/` directory structure
  - [ ] Create `checkpoints/`
  - [ ] Create `logs/`
  - [ ] Create `figures/`
  - [ ] Create `predictions/`
- [ ] 🟢 Create `requirements.txt`

---

## PHASE 1: DATA PREPROCESSING PIPELINE

### 1.1 Patient ID Harmonization
- [ ] 🟡 Implement `PatientRegistry` class (`src/data/patient_registry.py`)
  - [ ] Implement `extract_patient_id()` method (regex: TCGA-XX-XXXX)
  - [ ] Implement `build_registry()` method
    - [ ] Load clinical.txt and extract patient IDs
    - [ ] Load transcriptomics.txt headers and extract patient IDs
    - [ ] Load methylation.txt headers and extract patient IDs
    - [ ] Scan SVS directory for patient IDs
    - [ ] Build registry DataFrame with availability flags
  - [ ] Implement `get_complete_patients()` method
- [ ] 🟢 Run patient registry and verify output
  - [ ] Confirm total unique patients
  - [ ] Confirm patients with clinical data
  - [ ] Confirm patients with RNA-seq
  - [ ] Confirm patients with methylation
  - [ ] Confirm patients with SVS images
  - [ ] Identify patients with ALL modalities (~70-75 expected)

### 1.2 Whole Slide Image (WSI) Feature Extraction

#### 1.2.1 Tissue Detection & Tiling
- [ ] 🔴 Implement `WSITiler` class (`src/data/wsi_preprocessing.py`)
  - [ ] Implement `__init__()` with configuration parameters
    - [ ] Set tile_size (default: 256)
    - [ ] Set target_mpp (default: 0.5 for 20x)
    - [ ] Set tissue_threshold (default: 0.5)
    - [ ] Set max_tiles (default: 4000)
  - [ ] Implement `get_tissue_mask()` method
    - [ ] Read thumbnail from slide
    - [ ] Convert to HSV color space
    - [ ] Apply Otsu thresholding on saturation
    - [ ] Perform morphological cleanup (close, open)
  - [ ] Implement `get_tile_coordinates()` method
    - [ ] Calculate slide properties and downsample factors
    - [ ] Generate grid of tile coordinates
    - [ ] Filter tiles by tissue threshold
    - [ ] Limit to max_tiles if exceeded
  - [ ] Implement `extract_tiles()` method
    - [ ] Load slide with OpenSlide
    - [ ] Generate tissue mask
    - [ ] Extract tile coordinates
    - [ ] Extract tile images
    - [ ] Save to HDF5 format with compression

#### 1.2.2 Feature Extraction with UNI
- [ ] 🔴 Implement `TileDataset` class (`src/data/wsi_feature_extraction.py`)
  - [ ] Implement `__init__()` to load HDF5 tile file
  - [ ] Implement `__len__()` method
  - [ ] Implement `__getitem__()` method with transforms
- [ ] 🔴 Implement `UNIFeatureExtractor` class
  - [ ] Implement `_load_uni()` method
    - [ ] Download model from HuggingFace (MahmoodLab/UNI)
    - [ ] Create ViT-L/16 architecture with timm
    - [ ] Load pretrained weights
    - [ ] Set model to eval mode
  - [ ] Implement `_get_transform()` method
    - [ ] Resize to 224x224
    - [ ] Convert to tensor
    - [ ] Apply ImageNet normalization
  - [ ] Implement `extract_features()` method
    - [ ] Create DataLoader for tiles
    - [ ] Extract features in batches
    - [ ] Save features to HDF5

#### 1.2.3 Run WSI Pipeline
- [ ] 🔴 Create script `scripts/01_extract_wsi_features.py`
- [ ] 🔴 Process all 172 SVS files
  - [ ] Extract tiles for each slide
  - [ ] Extract UNI features for each slide
  - [ ] Log progress and statistics
- [ ] 🟢 Verify WSI feature extraction output
  - [ ] Check feature dimensions [N_tiles, 1024]
  - [ ] Verify HDF5 file integrity

### 1.3 Transcriptomics Preprocessing
- [ ] 🟡 Implement `RNAPreprocessor` class (`src/data/rna_preprocessing.py`)
  - [ ] Implement `__init__()` with filter parameters
    - [ ] Set min_count (default: 10)
    - [ ] Set min_samples (default: 0.2)
    - [ ] Set variance_percentile (default: 0.5)
  - [ ] Implement `load_data()` method
    - [ ] Load transcriptomics.txt
    - [ ] Extract patient IDs from column names
    - [ ] Remove duplicate samples
  - [ ] Implement `filter_genes()` method
    - [ ] Apply minimum count filter
    - [ ] Apply variance filter on log-transformed data
  - [ ] Implement `transform()` method
    - [ ] Apply log2(count + 1) transformation
    - [ ] Apply z-score normalization per gene
  - [ ] Implement `process()` method
    - [ ] Run full pipeline
    - [ ] Save to pickle file
- [ ] 🟢 Create script `scripts/02_preprocess_omics.py` (RNA section)
- [ ] 🟢 Run RNA preprocessing and verify output
  - [ ] Confirm gene count after filtering (~7,923 expected)
  - [ ] Confirm sample count
  - [ ] Verify output pickle file

### 1.4 Methylation Preprocessing
- [ ] 🟡 Implement `MethylationPreprocessor` class (`src/data/methylation_preprocessing.py`)
  - [ ] Implement `__init__()` with parameters
    - [ ] Set variance_percentile (default: 0.5)
    - [ ] Set clip_beta range (default: 0.001-0.999)
  - [ ] Implement `beta_to_mvalue()` method
    - [ ] Clip beta values
    - [ ] Convert to M-values: log2(beta / (1-beta))
  - [ ] Implement `load_data()` method
    - [ ] Load methylation.txt
    - [ ] Extract patient IDs
    - [ ] Remove duplicates
  - [ ] Implement `process()` method
    - [ ] Apply variance filter
    - [ ] Convert to M-values
    - [ ] Apply z-score normalization
    - [ ] Save to pickle
- [ ] 🟢 Add methylation section to `scripts/02_preprocess_omics.py`
- [ ] 🟢 Run methylation preprocessing and verify output

### 1.5 Mutation Preprocessing
- [ ] 🟡 Implement `MutationPreprocessor` class (`src/data/mutation_preprocessing.py`)
  - [ ] Define DRIVER_GENES list (15 HNSC drivers)
  - [ ] Define FUNCTIONAL_TYPES list (non-silent mutations)
  - [ ] Implement `load_data()` method
    - [ ] Load MAF file with relevant columns
  - [ ] Implement `extract_patient_id()` method
  - [ ] Implement `create_binary_matrix()` method
    - [ ] Filter to functional mutations
    - [ ] Create gene × patient binary matrix
  - [ ] Implement `compute_features()` method
    - [ ] Generate binary matrix
    - [ ] Compute mutation burden per patient
    - [ ] Extract driver gene profile
    - [ ] Compute mutation type distribution
  - [ ] Implement `process()` method
    - [ ] Run full pipeline
    - [ ] Save to pickle
- [ ] 🟢 Add mutation section to `scripts/02_preprocess_omics.py`
- [ ] 🟢 Run mutation preprocessing and verify output

### 1.6 Clinical Data & Survival Labels
- [ ] 🟡 Implement `ClinicalPreprocessor` class (`src/data/clinical_preprocessing.py`)
  - [ ] Define FEATURES list (clinical variables to include)
  - [ ] Implement `load_data()` method
    - [ ] Load clinical.txt
    - [ ] Set patient ID as index
  - [ ] Implement `extract_survival()` method
    - [ ] Create event indicator (1=dead, 0=censored)
    - [ ] Extract survival time (days_to_death or days_to_last_follow_up)
    - [ ] Clean missing values
  - [ ] Implement `encode_features()` method
    - [ ] One-hot encode categorical variables
    - [ ] Standardize numeric variables
  - [ ] Implement `process()` method
    - [ ] Run full pipeline
    - [ ] Align indices
    - [ ] Save to pickle
- [ ] 🟢 Add clinical section to `scripts/02_preprocess_omics.py`
- [ ] 🟢 Run clinical preprocessing and verify output
  - [ ] Confirm patient count with survival data
  - [ ] Verify event/censoring ratio
  - [ ] Check median survival time

### 1.7 Cross-Validation Split Generation
- [ ] 🟢 Implement `create_cv_splits()` function (`src/data/create_splits.py`)
  - [ ] Use StratifiedKFold (stratify by event status)
  - [ ] Generate 5-fold splits
  - [ ] Save train/val patient IDs per fold
  - [ ] Log split statistics
- [ ] 🟢 Create script `scripts/03_create_splits.py`
- [ ] 🟢 Run split generation and verify balance

---

## PHASE 2: MODALITY-SPECIFIC ENCODERS

### 2.1 WSI Encoder
- [ ] 🔴 Implement `GatedAttentionPooling` class (`src/models/encoders/wsi_encoder.py`)
  - [ ] Implement attention_V branch (Linear → Tanh)
  - [ ] Implement attention_U branch (Linear → Sigmoid)
  - [ ] Implement attention_weights layer
  - [ ] Implement `forward()` method
    - [ ] Compute gated attention
    - [ ] Apply softmax over tiles
    - [ ] Perform weighted aggregation
    - [ ] Optionally return attention weights
- [ ] 🔴 Implement `WSIEncoder` class
  - [ ] Implement feature projector (Linear → ReLU → Dropout → Linear)
  - [ ] Integrate GatedAttentionPooling
  - [ ] Implement output projection with LayerNorm
  - [ ] Implement `forward()` method

### 2.2 Gene Expression Encoder
- [ ] 🔴 Implement `PathwayAttentionEncoder` class (`src/models/encoders/rna_encoder.py`)
  - [ ] Implement gene value embedding
  - [ ] Create learnable gene position embeddings
  - [ ] Build TransformerEncoder with specified layers
  - [ ] Implement attention pooling over genes
  - [ ] Implement output projection
  - [ ] Implement `forward()` method with optional attention return
- [ ] 🟡 Implement `MLPEncoder` class (simpler alternative)
  - [ ] Build MLP with BatchNorm, ReLU, Dropout
  - [ ] Implement output projection with LayerNorm
  - [ ] Implement `forward()` method

### 2.3 Methylation Encoder
- [ ] 🟡 Implement `MethylationEncoder` class (`src/models/encoders/methylation_encoder.py`)
  - [ ] Build MLP encoder with bottleneck
  - [ ] Add BatchNorm, ReLU, Dropout layers
  - [ ] Implement output projection with LayerNorm
  - [ ] Implement `forward()` method

### 2.4 Mutation Encoder
- [ ] 🟡 Implement `MutationEncoder` class (`src/models/encoders/mutation_encoder.py`)
  - [ ] Implement binary matrix encoder
  - [ ] Implement driver gene attention mechanism
  - [ ] Implement mutation burden encoder
  - [ ] Implement fusion and output projection
  - [ ] Implement `forward()` method (accepts binary, drivers, burden)

### 2.5 Clinical Encoder
- [ ] 🟢 Implement `ClinicalEncoder` class (`src/models/encoders/clinical_encoder.py`)
  - [ ] Build simple MLP encoder
  - [ ] Implement output projection with LayerNorm
  - [ ] Implement `forward()` method

---

## PHASE 3: MULTI-MODAL FUSION ARCHITECTURE

### 3.1 Cross-Modal Attention Fusion
- [ ] 🔴 Implement `CrossModalAttention` class (`src/models/fusion/cross_attention.py`)
  - [ ] Implement Q, K, V projections
  - [ ] Implement multi-head attention computation
  - [ ] Implement output projection
  - [ ] Implement `forward()` method with optional attention return
- [ ] 🔴 Implement `PerceiverFusion` class
  - [ ] Create learnable latent queries
  - [ ] Build cross-attention layers (latents → modalities)
  - [ ] Build self-attention layers (latents → latents)
  - [ ] Implement layer norms
  - [ ] Implement output pooling over latents
  - [ ] Implement `forward()` method

### 3.2 Complete MOSAIC Model
- [ ] 🔴 Implement `MOSAIC` class (`src/models/mosaic.py`)
  - [ ] Initialize all modality encoders
    - [ ] WSIEncoder
    - [ ] RNA encoder (PathwayAttentionEncoder or MLPEncoder)
    - [ ] MethylationEncoder
    - [ ] MutationEncoder
    - [ ] ClinicalEncoder
  - [ ] Create modality type embeddings
  - [ ] Initialize PerceiverFusion
  - [ ] Implement survival prediction head
  - [ ] Implement `encode_modalities()` method
    - [ ] Encode each modality
    - [ ] Stack embeddings
    - [ ] Add modality type embeddings
  - [ ] Implement `forward()` method
    - [ ] Encode all modalities
    - [ ] Fuse with Perceiver
    - [ ] Predict risk score
    - [ ] Return output dict with attention (optional)

---

## PHASE 4: TRAINING PIPELINE

### 4.1 Cox Proportional Hazards Loss
- [ ] 🟡 Implement `CoxPHLoss` class (`src/losses/cox_loss.py`)
  - [ ] Implement negative partial log-likelihood computation
  - [ ] Handle sorting by time (descending)
  - [ ] Compute cumulative sum for risk set
  - [ ] Normalize by number of events
- [ ] 🟡 Implement `CoxPHLossWithTies` class (optional)
  - [ ] Implement Efron's approximation for tied times

### 4.2 PyTorch Lightning Trainer
- [ ] 🔴 Implement `MOSAICTrainer` class (`src/training/trainer.py`)
  - [ ] Initialize model from config
  - [ ] Initialize CoxPHLoss
  - [ ] Implement `forward()` method
  - [ ] Implement `training_step()` method
    - [ ] Forward pass
    - [ ] Compute loss
    - [ ] Log metrics
  - [ ] Implement `validation_step()` method
    - [ ] Forward pass
    - [ ] Compute loss
    - [ ] Store predictions for C-index
  - [ ] Implement `on_validation_epoch_end()` method
    - [ ] Concatenate predictions
    - [ ] Compute C-index
    - [ ] Log validation metrics
  - [ ] Implement `configure_optimizers()` method
    - [ ] Separate encoder/other parameters
    - [ ] Create AdamW optimizer with param groups
    - [ ] Implement warmup + decay LR scheduler

### 4.3 Dataset Classes
- [ ] 🔴 Implement `MultiModalSurvivalDataset` class (`src/data/multimodal_dataset.py`)
  - [ ] Implement `__init__()` to load all preprocessed data
  - [ ] Implement `__len__()` method
  - [ ] Implement `__getitem__()` method
    - [ ] Load WSI features from HDF5
    - [ ] Pad/truncate to max_tiles
    - [ ] Load RNA expression
    - [ ] Load methylation
    - [ ] Load mutation features (binary, drivers, burden)
    - [ ] Load clinical features
    - [ ] Load survival labels
    - [ ] Return dictionary of tensors

### 4.4 Training Scripts
- [ ] 🟡 Create script `scripts/04_train_unimodal.py`
  - [ ] Train WSI-only baseline
  - [ ] Train RNA-only baseline
  - [ ] Train Clinical-only baseline
  - [ ] Log results
- [ ] 🔴 Create script `scripts/05_train_multimodal.py`
  - [ ] Load preprocessed data
  - [ ] Create datasets and dataloaders
  - [ ] Initialize MOSAICTrainer
  - [ ] Configure PyTorch Lightning Trainer
    - [ ] Set up checkpointing
    - [ ] Set up early stopping
    - [ ] Set up WandB logging
  - [ ] Run training loop for each CV fold
  - [ ] Save best models

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
- [ ] 🔴 Implement `AttentionVisualizer` class (`src/explainability/attention_viz.py`)
  - [ ] Implement `get_attention_weights()` method
    - [ ] Extract modality attention
    - [ ] Extract WSI tile attention
    - [ ] Extract gene attention
  - [ ] Implement `visualize_wsi_attention()` method
    - [ ] Create thumbnail from slide
    - [ ] Map attention to thumbnail coordinates
    - [ ] Generate heatmap overlay
    - [ ] Save figure
  - [ ] Implement `visualize_modality_attention()` method
    - [ ] Create bar plot of cross-modal attention
    - [ ] Add value labels
    - [ ] Save figure
  - [ ] Implement `visualize_gene_importance()` method
    - [ ] Rank genes by attention
    - [ ] Create horizontal bar plot for top-k genes
    - [ ] Save figure

### 5.2 SHAP Analysis
- [ ] 🔴 Implement `SHAPAnalyzer` class (`src/explainability/shap_analysis.py`)
  - [ ] Implement `compute_modality_importance()` method
    - [ ] Ablate each modality
    - [ ] Measure prediction change
    - [ ] Normalize importance scores
  - [ ] Implement `plot_modality_importance()` method
    - [ ] Create bar plot
    - [ ] Save figure

### 5.3 Explainability Scripts
- [ ] 🟡 Create script `scripts/07_explainability.py`
  - [ ] Load trained model
  - [ ] Run attention extraction on test set
  - [ ] Generate WSI heatmaps for representative samples
  - [ ] Generate modality importance plots
  - [ ] Generate gene importance plots
  - [ ] Save all visualizations to outputs/figures/

---

## PHASE 6: EVALUATION & VALIDATION

### 6.1 Survival Metrics
- [ ] 🟡 Implement `SurvivalMetrics` class (`src/evaluation/metrics.py`)
  - [ ] Implement `compute_c_index()` static method
  - [ ] Implement `compute_time_dependent_auc()` static method
    - [ ] Use sksurv cumulative_dynamic_auc
    - [ ] Compute AUC at 1, 2, 3, 5 years
  - [ ] Implement `compute_integrated_brier_score()` static method (optional)
  - [ ] Implement `compute_all_metrics()` static method

### 6.2 Risk Stratification & Kaplan-Meier Analysis
- [ ] 🟡 Implement `SurvivalVisualization` class (`src/evaluation/visualization.py`)
  - [ ] Implement `plot_kaplan_meier_by_risk_group()` method
    - [ ] Split patients into risk groups (median or quantiles)
    - [ ] Fit Kaplan-Meier curves per group
    - [ ] Perform log-rank test
    - [ ] Generate publication-quality figure
  - [ ] Implement `plot_calibration()` method (optional)
    - [ ] Bin predictions
    - [ ] Compare predicted vs observed survival

### 6.3 Evaluation Scripts
- [ ] 🟡 Create script `scripts/06_evaluate.py`
  - [ ] Load best models from each CV fold
  - [ ] Generate predictions on held-out test data
  - [ ] Compute all metrics (C-index, time-dependent AUC)
  - [ ] Generate Kaplan-Meier plots
  - [ ] Create summary statistics table
  - [ ] Save results to outputs/predictions/

### 6.4 Run Full Evaluation
- [ ] 🟡 Execute evaluation pipeline
  - [ ] Compute cross-validated C-index
  - [ ] Compute time-dependent AUC at 1, 2, 3, 5 years
  - [ ] Generate risk stratification plots
  - [ ] Document results

---

## PHASE 7: ABLATION STUDIES

### 7.1 Ablation Study Implementation
- [ ] 🔴 Implement `AblationStudies` class (`scripts/ablation_studies.py`)
  - [ ] Define modality order
  - [ ] Implement `run_leave_one_out()` method
    - [ ] Train full model (baseline)
    - [ ] Train model without each modality
    - [ ] Record metrics
  - [ ] Implement `run_single_modality()` method
    - [ ] Train with only WSI
    - [ ] Train with only RNA
    - [ ] Train with only Methylation
    - [ ] Train with only Mutations
    - [ ] Train with only Clinical
  - [ ] Implement `run_progressive_addition()` method
    - [ ] Clinical only
    - [ ] Clinical + WSI
    - [ ] Clinical + WSI + RNA
    - [ ] Clinical + WSI + RNA + Methylation
    - [ ] Full model (all modalities)
  - [ ] Implement `plot_ablation_results()` method
    - [ ] Bar plot comparing C-index across experiments
    - [ ] Bar plot comparing time-dependent AUC

### 7.2 Run Ablation Studies
- [ ] 🔴 Execute leave-one-out ablation
  - [ ] Identify most important modality
  - [ ] Document performance drops
- [ ] 🔴 Execute single-modality experiments
  - [ ] Rank modalities by standalone performance
- [ ] 🔴 Execute progressive addition
  - [ ] Identify diminishing returns
  - [ ] Document synergistic effects
- [ ] 🟡 Generate ablation summary table and figures

---

## CONFIGURATION FILES

### Model Configuration
- [ ] 🟢 Create `configs/model_config.yaml`
  - [ ] Set wsi_input_dim: 1024
  - [ ] Set num_genes_rna: 7923
  - [ ] Set num_genes_meth: 10057
  - [ ] Set num_genes_mut: 1500
  - [ ] Set num_drivers: 15
  - [ ] Set num_clinical: 15
  - [ ] Set embed_dim: 256
  - [ ] Set hidden_dim: 512
  - [ ] Set num_latents: 16
  - [ ] Set num_heads: 8
  - [ ] Set num_fusion_layers: 2
  - [ ] Set dropout: 0.1

### Training Configuration
- [ ] 🟢 Create `configs/train_config.yaml`
  - [ ] Set learning_rate: 0.0001
  - [ ] Set encoder_lr_mult: 0.1
  - [ ] Set weight_decay: 0.00001
  - [ ] Set batch_size: 16
  - [ ] Set max_epochs: 100
  - [ ] Set warmup_epochs: 5
  - [ ] Set patience: 15

### Data Configuration
- [ ] 🟢 Create `configs/data_config.yaml`
  - [ ] Set max_tiles: 4000
  - [ ] Set tile_size: 256
  - [ ] Set target_mpp: 0.5
  - [ ] Set n_folds: 5
  - [ ] Set seed: 42

---

## NOTEBOOKS

- [ ] 🟢 Create `notebooks/01_data_exploration.ipynb`
  - [ ] Explore clinical data distributions
  - [ ] Visualize survival curves
  - [ ] Check modality availability
  - [ ] Generate summary statistics
- [ ] 🟢 Create `notebooks/02_baseline_analysis.ipynb`
  - [ ] Implement Cox regression baseline
  - [ ] Analyze clinical-only model
  - [ ] Document baseline metrics
- [ ] 🟡 Create `notebooks/03_results_visualization.ipynb`
  - [ ] Load trained model results
  - [ ] Create publication figures
  - [ ] Generate attention visualizations
  - [ ] Summarize ablation findings

---

## COMMON ISSUES & TROUBLESHOOTING

### Issue Handling
- [ ] 🟡 Implement memory-efficient WSI processing (chunk-based)
- [ ] 🟡 Implement stratified sampling for imbalanced censoring
- [ ] 🟡 Implement missing modality handling (masking + imputation)
- [ ] 🟡 Implement strong regularization for small sample size

---

## SUCCESS CRITERIA VALIDATION

### Minimum Viable Results
- [ ] 🟡 Achieve C-index > 0.70 on held-out test set
- [ ] 🟡 Achieve significant risk stratification (log-rank p < 0.01)
- [ ] 🟡 Generate attention visualizations with clinically meaningful patterns

### Target Results
- [ ] 🔴 Achieve C-index > 0.78
- [ ] 🔴 Achieve AUC(3yr) > 0.80
- [ ] 🔴 Document clear modality importance ranking
- [ ] 🔴 Identify interpretable attention patterns

### Stretch Goals
- [ ] 🔴 Achieve C-index > 0.82
- [ ] 🔴 Identify novel prognostic markers
- [ ] 🔴 Validate on external HNSC cohort
- [ ] 🔴 Prepare publication manuscript

---

## FINAL DELIVERABLES

- [ ] 🟡 Complete trained MOSAIC model (best checkpoint per fold)
- [ ] 🟡 Comprehensive evaluation report with all metrics
- [ ] 🟡 Ablation study summary with modality importance
- [ ] 🟡 Attention visualization gallery
- [ ] 🟡 Reproducible code repository with documentation
- [ ] 🟢 Configuration files for all experiments
- [ ] 🟢 Requirements.txt with exact versions

---

*MOSAIC Implementation Checklist v1.0*
*Generated: February 4, 2026*
*Total Tasks: ~200+ actionable items*
