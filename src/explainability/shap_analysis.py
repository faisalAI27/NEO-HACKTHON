import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultiModalSurvivalDataset, custom_collate
from src.training.trainer import MOSAICTrainer


class ModelWrapper:
    def __init__(self, model, reference_batch, target_modality):
        self.model = model
        self.reference_batch = reference_batch  # A single sample or batch of samples to use as fixed context
        self.target_modality = target_modality
        self.device = next(model.parameters()).device

    def predict(self, x_input):
        # x_input: numpy array (N, features)
        N = x_input.shape[0]

        # We need to construct a batch where target_modality comes from x_input
        # and other modalities come from reference_batch.
        # IF reference_batch has 1 sample, we repeat it N times.

        batch = {}
        with torch.no_grad():
            for k, v in self.reference_batch.items():
                if k == self.target_modality:
                    continue

                if isinstance(v, torch.Tensor):
                    # Take first sample of reference if it's a batch
                    ref_sample = v[0:1]  # (1, ...)
                    batch[k] = ref_sample.repeat(N, *([1] * (v.ndim - 1))).to(
                        self.device
                    )
                elif isinstance(v, dict):
                    batch[k] = {}
                    for sub_k, sub_v in v.items():
                        ref_sample = sub_v[0:1]
                        batch[k][sub_k] = ref_sample.repeat(
                            N, *([1] * (sub_v.ndim - 1))
                        ).to(self.device)

            # Set target
            # x_input is numpy
            batch[self.target_modality] = torch.tensor(x_input, dtype=torch.float32).to(
                self.device
            )

            # Forward
            logits = self.model(batch)
            return logits.cpu().numpy().flatten()


def run_shap(
    checkpoint_path, modality="clinical", fold=0, num_background=10, num_test=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")

    trainer = MOSAICTrainer.load_from_checkpoint(checkpoint_path)
    trainer.to(device)
    trainer.eval()
    model = trainer.model

    # Load Data (Train for background, Val for testing)
    data_dir = os.path.join(ROOT_DIR, "data", "processed")
    split_path = os.path.join(ROOT_DIR, "data", "splits", "cv_splits.json")

    # We use Train data to estimate background distribution
    train_dataset = MultiModalSurvivalDataset(data_dir, split_path, "train", fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=num_background,
        shuffle=True,
        collate_fn=custom_collate,
    )

    val_dataset = MultiModalSurvivalDataset(data_dir, split_path, "val", fold)
    val_loader = DataLoader(
        val_dataset, batch_size=num_test, shuffle=False, collate_fn=custom_collate
    )

    # Get Background Data
    bg_batch_raw = next(iter(train_loader))  # Dictionary
    # We need to extract the numpy part for the target modality to pass to KernelExplainer
    # And keep the rest as context?
    # KernelExplainer(model, data). data should be summary (kmeans) or small sample.

    # For 'data', we pass the target modality numpy array of background samples.
    bg_data_modality = bg_batch_raw[modality].numpy()  # (N_bg, Features)

    # But wait, ModelWrapper needs access to OTHER modalities.
    # If we pass multiple background samples to SHAP, SHAP integrates over them.
    # The Wrapper needs to know which "Other Modalities" to pair with the perturbed input.
    # This is complex in Multi-modal.
    # Simplified strategy: Use ONE reference sample (e.g. mean or median-like patient) for strict 'other' context.
    # OR better: The "Wrapper" is initialized with a FIXED context (e.g. the specific patient we are explaining),
    # and we only perturb the target modality. This explains "feature contribution GIVEN other modalities".

    print(f"Analyzing modality: {modality}")

    # Let's pick 1 sample from Val to explain
    val_batch = next(iter(val_loader))  # Batch of 'num_test' samples

    explanations = []

    # We explain each sample individually
    for i in range(num_test):
        print(f"Explaining sample {i}...")

        # Prepare context for this sample (all modalities fixed to this sample's values)
        # We need a batch of size 1
        sample_context = {}
        for k, v in val_batch.items():
            if isinstance(v, torch.Tensor):
                sample_context[k] = v[i : i + 1].to(device)
            elif isinstance(v, dict):
                sample_context[k] = {}
                for sk, sv in v.items():
                    sample_context[k][sk] = sv[i : i + 1].to(device)

        # wrapper
        wrapper = ModelWrapper(model, sample_context, modality)

        # Target input to explain
        x_explain = val_batch[modality][i : i + 1].numpy()  # (1, Feats)

        # Background: We need a background distribution for THIS modality.
        # We use the train set samples.
        explainer = shap.KernelExplainer(wrapper.predict, bg_data_modality)

        shap_values = explainer.shap_values(
            x_explain, nsamples=100
        )  # Limited nsamples for speed
        explanations.append(
            shap_values[0]
        )  # shap_values is list if output is list? wrapper returns (N,). So shap_values is (1, Feats).

    # Aggregate
    explanations = np.array(explanations)  # (Num_Test, Feats)

    # Summary Plot
    # Check feature names
    feature_names = None
    if modality == "clinical":
        feature_names = ["Gender", "Age", "Stage"]

    plt.figure()
    shap.summary_plot(
        explanations,
        features=val_batch[modality][:num_test].numpy(),
        feature_names=feature_names,
        show=False,
    )

    out_dir = "results/shap"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/shap_summary_{modality}.png")
    print(f"Saved SHAP plot to {out_dir}/shap_summary_{modality}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--modality", type=str, default="clinical")
    args = parser.parse_args()

    run_shap(args.checkpoint, args.modality)
