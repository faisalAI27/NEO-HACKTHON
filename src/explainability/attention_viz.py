import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# Add project root
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultiModalSurvivalDataset, custom_collate
from src.training.trainer import MOSAICTrainer


class AttentionVisualizer:
    def __init__(
        self, checkpoint_path, fold=0, device=None, output_dir="results/attention"
    ):
        self.checkpoint_path = checkpoint_path
        self.fold = fold
        self.output_dir = output_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load Model
        print(f"Loading checkpoint: {checkpoint_path}")
        self.trainer = MOSAICTrainer.load_from_checkpoint(checkpoint_path)
        self.trainer.to(self.device)
        self.trainer.eval()
        self.model = self.trainer.model

        # Load Data
        print("Setting up data loader...")
        self.data_dir = os.path.join(ROOT_DIR, "data", "processed")
        self.split_path = os.path.join(ROOT_DIR, "data", "splits", "cv_splits.json")

        self.dataset = MultiModalSurvivalDataset(
            data_dir=self.data_dir,
            split_path=self.split_path,
            split_type="val",
            fold=fold,
            max_wsi_tiles=20000,  # Ensure we get all tiles for visualization
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=1,  # Analyze one patient at a time for detailed viz
            shuffle=False,
            num_workers=2,
            collate_fn=custom_collate,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Storage
        self.patient_attention = (
            {}
        )  # case_id -> {modality: attn, wsi_attn: ..., rna_attn: ...}

    def compute_all_attention(self):
        """
        Runs inference on the dataset and collects attention maps.
        """
        print("Computing attention maps...")
        self.patient_attention = {}

        with torch.no_grad():
            for batch_dict in tqdm(self.loader):
                # Handle batch
                batch = batch_dict  # custom_collate return dict
                case_ids = batch["case_id"]  # List

                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, torch.Tensor):
                                v[sub_k] = sub_v.to(self.device)

                # Forward with attention
                # mosaic.py now returns (logits, all_attn_dict)
                logits, all_attn = self.model(batch, return_attention=True)

                # Unwrap batch
                # all_attn is a dict where values are tensors (Batch, ...)
                # Since batch_size=1 (mostly), or N. We need to split rows.

                bs = len(case_ids)

                # Identify keys
                keys = list(all_attn.keys())

                for i in range(bs):
                    cid = case_ids[i]
                    p_data = {}

                    for k in keys:
                        val = all_attn[k]
                        if val is None:
                            continue
                        # Slice
                        p_data[k] = val[i].cpu().numpy()

                    self.patient_attention[cid] = p_data

    def visualize_modality_importance(self):
        """
        Global bar plot of modality importance (averaged over all patients).
        """
        print("Generating Modality Importance Plot...")

        # Aggregate Cross-Modal Attention
        # Key: 'cross_modal' -> Shape (Latents, Modalities) or (Heads, Latents, Modalities)
        # We need to check shape.

        # Assuming we can inspect one
        if not self.patient_attention:
            print("No attention data found. Run compute_all_attention() first.")
            return

        first_sample = next(iter(self.patient_attention.values()))
        cm_attn = first_sample.get("cross_modal", None)

        if cm_attn is None:
            print("No Cross-Modal attention found.")
            return

        # Determine Modality Names from Model
        model_mods = list(self.model.encoders.keys())
        # We need to filter only those present in the batch, but per patient it varies?
        # Dataset returns consistent keys.

        # Accumulate
        mod_scores = {m: [] for m in model_mods}

        for cid, data in self.patient_attention.items():
            # cm: (Heads, Latents, Modalities)
            cm = data["cross_modal"]
            # Average over Heads and Latents
            # If shape is (H, L, M) -> mean(0,1) -> (M,)
            # If shape is (L, M) -> mean(0) -> (M,)

            if cm.ndim == 3:
                score = np.mean(cm, axis=(0, 1))
            elif cm.ndim == 2:
                score = np.mean(cm, axis=0)
            else:
                score = cm

            # Use safe mapping
            # Cross attention last dim size corresponds to number of modalities encoded.
            # If batch had all modalities, size == len(model_mods).
            # If some missing, size < len.
            # But which ones?
            # In MOSAIC.forward, 'encoded_modalities' depends on 'if mod in batch'.
            # Dataset usually provides all keys.

            if len(score) != len(model_mods):
                # This suggests dynamic modality dropping.
                # Assuming standard order of keys in model.encoders
                # We need to know which keys were used.
                # Impossible to know from just the tensor unless we saved it.
                # For now assume consistent.
                pass

            for i, val in enumerate(score):
                if i < len(model_mods):
                    mod_scores[model_mods[i]].append(val)

        # Plot
        means = {m: np.mean(vals) for m, vals in mod_scores.items() if len(vals) > 0}

        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(list(means.items()), columns=["Modality", "Attention"])
        sns.barplot(data=df, x="Modality", y="Attention")
        plt.title("Global Average Modality Attention")
        plt.savefig(os.path.join(self.output_dir, "modality_importance_global.png"))
        plt.close()
        print("Saved modality_importance_global.png")

    def visualize_wsi_heatmap(self, case_id, top_k=5):
        """
        Visualizes WSI attention for a specific patient.
        """
        if case_id not in self.patient_attention:
            print(f"Case {case_id} not found in results.")
            return

        res = self.patient_attention[case_id]
        if "wsi" not in res:
            print("No WSI attention found.")
            return

        # WSI Attention: (n_tiles, ) or (n_tiles, 1)
        # Note: GatedAttentionPooling returns (A_weights) which is (1, n_tiles).
        # Our update returns (Batch=1, n_tiles).
        attn = res["wsi"].flatten()

        # Load Coords
        h5_path = os.path.join(self.data_dir, "wsi_features", f"{case_id}.h5")
        if not os.path.exists(h5_path):
            print(f"WSI feature file not found: {h5_path}")
            return

        try:
            with h5py.File(h5_path, "r") as f:
                coords = f["coords"][:]  # (N, 2)

                # Check consistency
                if len(coords) != len(attn):
                    # Truncate attn if padded
                    # Padding usually adds zeros at the end.
                    if len(attn) > len(coords):
                        attn = attn[: len(coords)]
                    else:
                        print(f"Attn {len(attn)} shorter than coords {len(coords)}?")
                        return

                # Plot Heatmap
                plt.figure(figsize=(10, 8))
                # Scatter coords, color by attn
                # Invert Y usually for slides
                sc = plt.scatter(
                    coords[:, 0], -coords[:, 1], c=attn, cmap="viridis", alpha=0.6, s=20
                )
                plt.colorbar(sc, label="Attention Score")
                plt.title(f"WSI Attention Heatmap: {case_id}")
                plt.axis("equal")
                plt.savefig(os.path.join(self.output_dir, f"wsi_heatmap_{case_id}.png"))
                plt.close()
                print(f"Saved wsi_heatmap_{case_id}.png")

                # Top K Tiles
                if "imgs" in f:
                    imgs = f["imgs"]
                    top_indices = np.argsort(attn)[-top_k:][::-1]

                    fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 4))
                    if top_k == 1:
                        axes = [axes]
                    elif isinstance(axes, np.ndarray):
                        axes = axes.flatten()

                    for i, idx in enumerate(top_indices):
                        tile = imgs[idx]
                        score = attn[idx]
                        axes[i].imshow(tile)
                        axes[i].set_title(f"Rank {i+1}\nScore: {score:.4f}")
                        axes[i].axis("off")

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(self.output_dir, f"wsi_top_tiles_{case_id}.png")
                    )
                    plt.close()
                    print(f"Saved wsi_top_tiles_{case_id}.png")

        except Exception as e:
            print(f"Error visualizing WSI: {e}")

    def visualize_rna_importance(self, top_k=20):
        """
        Aggregates gene attention to find top genes.
        """
        print("Analyzing Gene Importance...")
        gene_attns = []

        for cid, data in self.patient_attention.items():
            if "rna" in data:
                # Shape: (num_genes, )
                gene_attns.append(data["rna"])

        if not gene_attns:
            print("No RNA attention data found.")
            return

        # Stack: (Num_Patients, Num_Genes)
        try:
            # Check shapes equal
            shapes = [g.shape for g in gene_attns]
            if len(set(shapes)) > 1:
                print("Warning: Variable gene shapes. Skipping global aggregation.")
                return

            all_genes = np.stack(gene_attns)
            mean_gene_attn = np.mean(all_genes, axis=0)  # (Num_Genes, )

            # Plot Top K
            # Indices
            top_indices = np.argsort(mean_gene_attn)[-top_k:]
            top_scores = mean_gene_attn[top_indices]

            # Names?
            # We don't have gene names easily loaded yet.
            # Using Generic Names
            labels = [f"Gene_{i}" for i in top_indices]

            plt.figure(figsize=(8, 10))
            plt.barh(range(top_k), top_scores)
            plt.yticks(range(top_k), labels)
            plt.xlabel("Importance Score")
            plt.title(f"Top {top_k} Important Genes")
            plt.savefig(os.path.join(self.output_dir, "gene_importance.png"))
            plt.close()
            print("Saved gene_importance.png")

        except Exception as e:
            print(f"Error analyzing RNA: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--output_dir", type=str, default="results/attention")

    args = parser.parse_args()

    viz = AttentionVisualizer(
        args.checkpoint, fold=args.fold, output_dir=args.output_dir
    )
    viz.compute_all_attention()
    viz.visualize_modality_importance()
    viz.visualize_rna_importance()

    # Pick 2-3 random cases for WSI
    import random

    cases = list(viz.patient_attention.keys())
    if cases:
        selected = random.sample(cases, min(3, len(cases)))
        for c in selected:
            viz.visualize_wsi_heatmap(c)
