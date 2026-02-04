import os
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.multimodal_dataset import MultiModalSurvivalDataset, custom_collate
from src.training.trainer import MOSAICTrainer
from torch.utils.data import DataLoader

def main(args):
    pl.seed_everything(42)
    
    # 1. Config Base
    # Full dims available in data
    all_data_dims = {
        'rna': 3000,
        'wsi': 1024,
        'clinical': 3,
        'mutations': 1852,
        'drivers': 0,
        'methylation': 3000
    }
    
    # Select only the requested modality
    if args.modality == 'all':
        data_dims = all_data_dims
    else:
        # For 'mutations', we need 'mutations' key in data_dims
        # For others, key matches args.modality
        if args.modality == 'mutations':
             data_dims = {'mutations': all_data_dims['mutations'], 'drivers': 0}
        else:
             data_dims = {args.modality: all_data_dims[args.modality]}

    print(f"Training Unimodal Model for: {args.modality}")
    print(f"Data Dims: {data_dims}")

    config = {
        'data_dims': data_dims,
        'encoders': {
            'rna_out_dim': 256,
            'wsi_out_dim': 256,
            'clin_out_dim': 256,
            'mut_out_dim': 256,
            'meth_out_dim': 256,
            # Per-encoder args
            'rna': {'hidden_dim': 512, 'num_layers': 2},
            'wsi': {'hidden_dim': 256, 'dropout': 0.25}, 
            'methylation': {'hidden_dim': 512, 'dropout': 0.3},
        },
        'fusion': {
            'latent_dim': 256,
            'num_latents': 32, # Perceiver latents
            'depth': 2
        }
    }
    
    # 2. Data
    data_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    split_path = os.path.join(ROOT_DIR, 'data', 'splits', 'cv_splits.json')
    
    # Note: Dataset loads ALL data currently. 
    # The Model will only use the keys present in `data_dims`.
    # However, for efficiency, future optimization could prevent loading unused data.
    # For now, we load all, but model only encodes selected.
    
    train_dataset = MultiModalSurvivalDataset(
        data_dir=data_dir,
        split_path=split_path,
        split_type='train',
        fold=args.fold
    )
    
    val_dataset = MultiModalSurvivalDataset(
        data_dir=data_dir,
        split_path=split_path,
        split_type='val',
        fold=args.fold
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=custom_collate,
        drop_last=True # Fix for batchnorm if batch size is small
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate
    )
    
    # 3. Model
    model = MOSAICTrainer(
        model_config=config,
        lr=args.lr,
        max_epochs=args.max_epochs
    )
    
    # 4. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/unimodal/{args.modality}/fold_{args.fold}',
        filename='model-{epoch:02d}-{val_c_index:.3f}',
        save_top_k=1,
        monitor='val_c_index',
        mode='max' 
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True, 
                        choices=['rna', 'wsi', 'clinical', 'mutations', 'methylation', 'all'],
                        help='Modality to train on')
    parser.add_argument('--fold', type=int, default=0, help='CV Fold (0-4)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
