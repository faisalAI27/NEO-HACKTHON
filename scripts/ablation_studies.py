import os
import sys
import argparse
import copy
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.multimodal_dataset import MultiModalSurvivalDataset, custom_collate
from src.training.trainer import MOSAICTrainer
from src.evaluation.metrics import SurvivalMetrics

class AblationStudies:
    def __init__(self, output_dir: str, device: str = 'cuda'):
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
        # Base Data Dims (Full set)
        self.base_data_dims = {
            'rna': 3000,
            'wsi': 1024,
            'clinical': 3,
            'mutations': 1852,
            'drivers': 0, # Not an encoder key usually, passed to mutation encoder
            'methylation': 3000
        }
        
        # Base Model Config
        self.base_config = {
            'encoders': {
                'rna_out_dim': 256,
                'wsi_out_dim': 256,
                'clin_out_dim': 256,
                'mut_out_dim': 256,
                'meth_out_dim': 256,
                'rna': {'hidden_dim': 512, 'num_layers': 2},
                'wsi': {'hidden_dim': 256, 'dropout': 0.25}, 
                'methylation': {'hidden_dim': 512, 'dropout': 0.3},
            },
            'fusion': {
                'latent_dim': 256,
                'num_latents': 32,
                'depth': 2
            }
        }
        
        # Paths
        self.data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
        self.split_path = os.path.join(ROOT_DIR, 'data', 'splits', 'cv_splits.json')
        
        # Results storage
        self.results = []

    def get_config_for_modalities(self, modalities: list):
        """
        Creates a config dict with only the specified modalities in data_dims.
        """
        config = copy.deepcopy(self.base_config)
        # Filter data_dims
        filtered_dims = {}
        
        # Always include auxiliary keys if needed (like 'drivers' for mutation)
        # Assuming 'drivers' is a parameter for mutations, checking how it's used.
        # In base_data_dims, 'drivers' is present.
        
        for mod in modalities:
            if mod in self.base_data_dims:
                filtered_dims[mod] = self.base_data_dims[mod]
            else:
                print(f"Warning: Modality {mod} not found in base dims.")
        
        # Special case: if mutations is present, include drivers count if needed
        if 'mutations' in modalities and 'drivers' in self.base_data_dims:
             filtered_dims['drivers'] = self.base_data_dims['drivers']
             
        config['data_dims'] = filtered_dims
        return config

    def train_and_evaluate(self, experiment_name: str, config: dict, fold: int = 0, max_epochs: int = 30):
        print(f"\n--- Running Experiment: {experiment_name} (Fold {fold}) ---")
        pl.seed_everything(42)
        
        # 1. Data Loaders
        train_dataset = MultiModalSurvivalDataset(
            data_dir=self.data_dir,
            split_path=self.split_path,
            split_type='train',
            fold=fold
        )
        val_dataset = MultiModalSurvivalDataset(
            data_dir=self.data_dir,
            split_path=self.split_path,
            split_type='val',
            fold=fold
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, collate_fn=custom_collate, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate)
        
        # 2. Model
        model = MOSAICTrainer(
            model_config=config,
            lr=1e-4,
            max_epochs=max_epochs
        )
        
        # 3. Trainer
        ckpt_dir = os.path.join(self.output_dir, 'checkpoints', experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='model-{epoch:02d}-{val_c_index:.2f}',
            save_top_k=1,
            monitor='val_c_index',
            mode='max'
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=[checkpoint_callback, early_stop],
            default_root_dir=ckpt_dir,
            enable_progress_bar=True,
            logger=False # simplify logging
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Evaluate Best Model
        best_path = checkpoint_callback.best_model_path
        if not best_path:
            # Fallback if no checkpoint saved (e.g. very short training)
            best_path = None # Should use last model or fail
            print("Warning: No best checkpoint found, using current.")
            
        print(f"Loading best model from {best_path}")
        if best_path:
            model = MOSAICTrainer.load_from_checkpoint(best_path)
            model.eval()
            model.to(self.device)
        
        # Inference
        risk_scores = []
        events = []
        times = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                    elif isinstance(v, dict):
                        for sk, sv in v.items():
                           if isinstance(sv, torch.Tensor):
                               v[sk] = sv.to(self.device)
                               
                out = model(batch)
                if isinstance(out, dict):
                    pred = out.get('hazard', out.get('logits'))
                else:
                    pred = out
                
                risk_scores.append(pred.cpu().numpy())
                events.append(batch['event'].cpu().numpy())
                times.append(batch['time'].cpu().numpy())
                
        risk_scores = np.concatenate(risk_scores).flatten()
        events = np.concatenate(events).flatten()
        times = np.concatenate(times).flatten()
        
        # Calc Metrics
        metric_engine = SurvivalMetrics()
        c_index = metric_engine.compute_c_index(events, times, risk_scores)
        
        # Record
        result = {
            'experiment': experiment_name,
            'c_index': c_index,
            'modalities': "+".join(sorted(config['data_dims'].keys()))
        }
        self.results.append(result)
        print(f"Result for {experiment_name}: C-Index = {c_index:.4f}")
        
        return c_index

    def run_leave_one_out(self, all_modalities=['wsi', 'rna', 'methylation', 'mutations', 'clinical'], max_epochs=10):
        print("\n=== Starting Leave-One-Out Ablation ===")
        
        # 1. Full Model (Baseline)
        config_full = self.get_config_for_modalities(all_modalities)
        self.train_and_evaluate("Full_Model", config_full, max_epochs=max_epochs)
        
        # 2. Leave one out
        for mod in all_modalities:
            subset = [m for m in all_modalities if m != mod]
            config_loo = self.get_config_for_modalities(subset)
            self.train_and_evaluate(f"No_{mod.capitalize()}", config_loo, max_epochs=max_epochs)

    def run_single_modality(self, all_modalities=['wsi', 'rna', 'methylation', 'mutations', 'clinical'], max_epochs=10):
        print("\n=== Starting Single Modality Ablation ===")
        
        for mod in all_modalities:
            config_single = self.get_config_for_modalities([mod])
            self.train_and_evaluate(f"Only_{mod.capitalize()}", config_single, max_epochs=max_epochs)

    def run_progressive_addition(self, modality_order=['clinical', 'wsi', 'rna', 'methylation', 'mutations'], max_epochs=10):
        print("\n=== Starting Progressive Addition Ablation ===")
        
        current_mods = []
        for mod in modality_order:
            current_mods.append(mod)
            name = f"Add_{mod.capitalize()}"
            config_prog = self.get_config_for_modalities(current_mods)
            self.train_and_evaluate(name, config_prog, max_epochs=max_epochs)

    def plot_results(self):
        if not self.results:
            print("No results to plot.")
            return
            
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_dir, 'ablation_results.csv'), index=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='experiment', y='c_index', hue='experiment', palette='viridis', legend=False)
        plt.title('Ablation Study Results: C-Index')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.4, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ablation_c_index.png'))
        print(f"Results saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['loo', 'single', 'progressive', 'all'], help='Ablation mode')
    parser.add_argument('--output_dir', type=str, default='outputs/ablations')
    parser.add_argument('--max_epochs', type=int, default=10, help='Reduced epochs for ablation speed')
    args = parser.parse_args()
    
    study = AblationStudies(output_dir=args.output_dir)
    
    all_mods = ['clinical', 'wsi', 'rna', 'methylation', 'mutations']
    
    # Using small max_epochs can be overridden inside methods if I updated them to accept it, 
    # but I hardcoded 30 in define. Let's patch it.
    # Note: patched manually in `train_and_evaluate` definition above to take arg, 
    # but `run_*` methods call it without arg.
    # I will rely on default or update the methods.
    # Actually simplest is to modify `train_and_evaluate` to accept max_epochs and pass it.
    # But I can't edit the class now easily without rewriting.
    # For now, I'll let it use default 30, or I can monkeypatch if needed. 
    # Wait, I defined `train_and_evaluate` to take `max_epochs` with default 30.
    # The `run_*` methods call it WITHOUT max_epochs arg, so they use 30.
    # If I want to control it via args, I should have passed it.
    # I'll just save the file as is. 30 epochs is reasonable for ablation on small data.
    
    if args.mode == 'loo' or args.mode == 'all':
        study.run_leave_one_out(all_mods, max_epochs=args.max_epochs)
        
    if args.mode == 'single' or args.mode == 'all':
        study.run_single_modality(all_mods, max_epochs=args.max_epochs)
        
    if args.mode == 'progressive' or args.mode == 'all':
        study.run_progressive_addition(['clinical', 'wsi', 'rna', 'methylation', 'mutations'], max_epochs=args.max_epochs)
        
    study.plot_results()

if __name__ == "__main__":
    main()
