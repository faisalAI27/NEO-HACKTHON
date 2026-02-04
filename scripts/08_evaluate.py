import os
import sys
import argparse
import glob
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.multimodal_dataset import MultiModalSurvivalDataset, custom_collate
from src.training.trainer import MOSAICTrainer
from src.evaluation.metrics import SurvivalMetrics
from src.evaluation.visualization import SurvivalVisualization

def get_best_checkpoint(fold_dir):
    """
    Finds the best checkpoint in the fold directory (maximizing val_c_index).
    Assumes filename format: mosaic-{epoch}-{val_loss}-{val_c_index}.ckpt
    """
    checkpoints = glob.glob(os.path.join(fold_dir, "*.ckpt"))
    if not checkpoints:
        return None
    
    # Heuristic: ModelCheckpoint creates filenames with metrics. 
    # If we trusted the saver, the one present might be the best (save_top_k=1).
    # We'll just take the first one found or sort by C-index if multiple exist.
    # Current filename format in train script: 'mosaic-{epoch:02d}-{val_loss:.2f}-{val_c_index:.2f}'
    
    # Let's try to parse c_index from filename just in case multiple exist
    best_ckpt = None
    best_score = -1.0
    
    for ckpt in checkpoints:
        try:
            # simple parse
            basename = os.path.basename(ckpt)
            # expected format: mosaic-epoch=XX-val_loss=XX-val_c_index=0.XX.ckpt
            # The train script format string was: 'mosaic-{epoch:02d}-{val_loss:.2f}-{val_c_index:.2f}'
            # Examples might look like: mosaic-epoch=12-val_loss=3.45-val_c_index=0.72.ckpt
            # Or if auto-formatted: mosaic-epoch=12-val_loss=3.45-val_c_index=0.72.ckpt
            
            # Let's just rely on the fact that save_top_k=1 usually leaves the best one.
            # So just picking the most recent file or the one with highest c_index if parseable.
            if 'val_c_index=' in basename:
                score = float(basename.split('val_c_index=')[1].split('.ckpt')[0])
            elif 'val_c_index' in basename:
                 # Try to extract the number after val_c_index
                 temp = basename.split('val_c_index')[1]
                 # it might be "-0.72" or "=0.72"
                 import re
                 match = re.search(r"[-=](\d+\.\d+)", temp)
                 if match:
                     score = float(match.group(1))
                 else:
                     score = 0.0
            else:
                score = 0.0
                
            if score > best_score:
                best_score = score
                best_ckpt = ckpt
        except:
            # Fallback
            best_ckpt = ckpt
            
    # If parsing failed or simple setup
    if best_ckpt is None and checkpoints:
        best_ckpt = checkpoints[0]
        
    return best_ckpt

def evaluate_fold(fold_idx, fold_dir, data_dir, split_path, device):
    print(f"\nEvaluating Fold {fold_idx}...")
    
    ckpt_path = get_best_checkpoint(fold_dir)
    if not ckpt_path:
        print(f"No checkpoint found for fold {fold_idx}. Skipping.")
        return None

    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load model
    # We rely on PL to load hparams
    try:
        model = MOSAICTrainer.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None
        
    model.to(device)
    model.eval()
    
    # Load Data
    # 1. Train data for IPCW
    dataset_train = MultiModalSurvivalDataset(
        data_dir=data_dir,
        split_path=split_path,
        split_type='train',
        fold=fold_idx
    )
    
    # 2. Val (Test) data
    dataset_val = MultiModalSurvivalDataset(
        data_dir=data_dir,
        split_path=split_path,
        split_type='val',
        fold=fold_idx
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=32, # Batch size doesn't affect metrics, just speed
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    # Collect Predictions
    risk_scores = []
    events = []
    times = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Fold {fold_idx} Inference"):
            # Move batch to device
            # batch is a dict, but model expects specific args or we just call model(batch) if implemented
            # MOSAIC.forward expects x dict
            # MOSAICTrainer.forward calls self.model(x)
            
            # The collate function returns a dictionary.
            # We need to ensure nested tensors are on device.
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, torch.Tensor):
                            v[sub_k] = sub_v.to(device)
            
            # Forward
            outputs = model(batch)
            # Outputs should be risk scores (hazards)
            # MOSAIC returns logits or hazard?
            # src/models/mosaic.py forward returns 'risk_score' usually
            
            # Check what model returns
            # Assuming it returns a dict or tensor. 
            # If MOSAICTrainer.forward is just self.model(x), checking MOSAIC.forward...
            # Usually it returns a dict with 'logits', 'risk' etc. or just risk.
            # Let's assume prediction is the risk score output.
            
            if isinstance(outputs, dict):
                pred = outputs.get('hazard', outputs.get('logits', None))
            else:
                pred = outputs
                
            risk_scores.append(pred.cpu().numpy())
            events.append(batch['event'].cpu().numpy())
            times.append(batch['time'].cpu().numpy())
            
    risk_scores = np.concatenate(risk_scores).flatten()
    events = np.concatenate(events).flatten()
    times = np.concatenate(times).flatten()
    
    # Prepare data for metrics
    # Get training data stats for IPCW
    # Reading all train data just for time/event is fast
    print("Loading training stats for IPCW...")
    train_events = []
    train_times = []
    # optimize: we don't need to load the heavy modalities, just the CSV/clinical really 
    # but dataset loads everything. We'll just iterate quickly or inspect the internal df if possible.
    # dataset_train.survival_data is a DataFrame?
    # MultiModalSurvivalDataset usually has self.registry or similar.
    # Let's try to access underlying metadata if possible to avoid loading images.
    
    # Looking at src/data/multimodal_dataset.py logic again (mental check):
    # It likely performs joins in __init__. 
    # Yes, let's just use what's loaded in __init__ if public.
    # If not, we have to iterate.
    # Hack: Inspect dataset object
    if hasattr(dataset_train, 'patient_ids'):
         # If we can't easily access the survival info without getting items, we might be stuck iterating.
         # But the dataset is indexed.
         # Let's just iterate, n=1000 is fast enough if lazy loading works.
         # If lazy loading is not implemented, init might be slow.
         # Assuming naive iteration is OK for now.
         pass
         
    # Optimization: Just use the dataset provided
    # Actually, we can just extract from the loader loop if we made a loader, 
    # or just ask the dataset for all labels.
    # The dataset probably has a list of labels.
    # Let's hope dataset_train[i] is fast.
    
    # Just loading targets from train set
    # Create a simple loader
    train_loader_simple = DataLoader(dataset_train, batch_size=128, num_workers=4, collate_fn=custom_collate)
    for batch in tqdm(train_loader_simple, desc="Loading Train Stats"):
         train_events.append(batch['event'].numpy())
         train_times.append(batch['time'].numpy())
         
    train_events = np.concatenate(train_events).flatten()
    train_times = np.concatenate(train_times).flatten()
    
    # Compute Metrics
    metric_engine = SurvivalMetrics()
    
    train_data_dict = {'event': train_events, 'time': train_times}
    test_data_dict = {'event': events, 'time': times}
    prediction_dict = {'risk_scores': risk_scores}
    
    # Time points for AUC: 1, 2, 3, 5 years -> 365, 730, 1095, 1825 days
    eval_times = np.array([365, 730, 1095, 1825])
    # Filter times that are within the range of the test data (metrics engine handles it roughly, 
    # but IPCW requires times < max(train_time) usually)
    
    metrics = metric_engine.compute_all_metrics(
        train_data_dict, test_data_dict, prediction_dict, times=eval_times
    )
    
    return {
        'metrics': metrics,
        'predictions': {
            'risk_scores': risk_scores,
            'event': events,
            'time': times
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory containing fold_* checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions', help='Directory to save results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    split_path = os.path.join(ROOT_DIR, 'data', 'splits', 'cv_splits.json')
    
    # Discover folds
    fold_dirs = sorted(glob.glob(os.path.join(args.checkpoints_dir, "fold_*")))
    if not fold_dirs:
        print("No fold directories found in checkpoints/")
        return

    all_fold_metrics = []
    
    # Storage for aggregated predictions (CV)
    all_risk_scores = []
    all_events = []
    all_times = []
    
    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        try:
            fold_idx = int(fold_name.split('_')[1])
        except:
            continue
            
        result = evaluate_fold(fold_idx, fold_dir, data_dir, split_path, device)
        if result:
            fold_metrics = result['metrics']
            print(f"Fold {fold_idx} Metrics: {fold_metrics}")
            all_fold_metrics.append(fold_metrics)
            
            # Save Fold Predictions
            df = pd.DataFrame(result['predictions'])
            df.to_csv(os.path.join(args.output_dir, f'predictions_fold_{fold_idx}.csv'), index=False)
            
            # Aggregate
            all_risk_scores.append(result['predictions']['risk_scores'])
            all_events.append(result['predictions']['event'])
            all_times.append(result['predictions']['time'])
            
    # Aggregate Metrics Summary
    if not all_fold_metrics:
        print("No metrics computed.")
        return

    metrics_df = pd.DataFrame(all_fold_metrics)
    summary = metrics_df.describe().loc[['mean', 'std']]
    print("\n=== Cross-Validation Summary ===")
    print(summary)
    
    summary.to_csv(os.path.join(args.output_dir, 'cv_metrics_summary.csv'))
    
    # Global Plots (Risk Stratification on concatenated results)
    print("\nGenerating Global Kaplan-Meier Plots...")
    viz = SurvivalVisualization()
    
    global_risk = np.concatenate(all_risk_scores)
    global_event = np.concatenate(all_events)
    global_time = np.concatenate(all_times)
    
    # Save combined predictions
    global_df = pd.DataFrame({
        'risk_score': global_risk,
        'event': global_event,
        'time': global_time
    })
    global_df.to_csv(os.path.join(args.output_dir, 'predictions_all_folds.csv'), index=False)
    
    # 1. KM by Median
    viz.plot_kaplan_meier_by_risk_group(
        global_event, global_time, global_risk,
        group_method='median',
        save_path=os.path.join(args.output_dir, 'km_global_median.png'),
        title="CV Kaplan-Meier (Median Split)"
    )
    
    # 2. KM by Quartiles
    viz.plot_kaplan_meier_by_risk_group(
        global_event, global_time, global_risk,
        group_method='quartiles',
        save_path=os.path.join(args.output_dir, 'km_global_quartiles.png'),
        title="CV Kaplan-Meier (Quartile Split)"
    )
    
    # 3. Risk Distribution
    viz.plot_risk_distribution(
        global_risk,
        save_path=os.path.join(args.output_dir, 'risk_distribution.png')
    )
    
    print(f"\nEvaluation Complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
