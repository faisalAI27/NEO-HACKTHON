"""
MOSAIC Model Inference Service

Provides functionality to load trained checkpoints and perform inference
for survival prediction and attention map extraction.
"""

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model components
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.mosaic import MOSAIC
from src.training.trainer import MOSAICTrainer


class MOSAICModelService:
    """
    Service class for loading and running inference with trained MOSAIC models.

    Usage:
        service = MOSAICModelService()
        service.load_checkpoint("checkpoints/fold_0/best.ckpt")
        result = service.predict_survival(patient_data)
    """

    def __init__(self, device: Optional[str] = None, demo_mode: bool = True):
        """
        Initialize the model service.

        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            demo_mode: If True, allow simulated predictions when no model is loaded
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[MOSAICTrainer] = None
        self.checkpoint_path: Optional[str] = None
        self.is_loaded = False
        self.demo_mode = demo_mode

        logger.info(f"MOSAICModelService initialized on device: {self.device}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load a trained model checkpoint.

        Args:
            checkpoint_path: Path to the .ckpt file

        Returns:
            bool: True if loaded successfully
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                # Try to find best checkpoint in directory
                if checkpoint_path.is_dir():
                    ckpts = list(checkpoint_path.glob("*.ckpt"))
                    if not ckpts:
                        logger.error(f"No checkpoints found in {checkpoint_path}")
                        return False
                    checkpoint_path = sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
                else:
                    logger.error(f"Checkpoint not found: {checkpoint_path}")
                    return False

            logger.info(f"Loading checkpoint from: {checkpoint_path}")

            self.model = MOSAICTrainer.load_from_checkpoint(
                str(checkpoint_path), map_location=self.device
            )
            self.model.to(self.device)
            self.model.eval()

            self.checkpoint_path = str(checkpoint_path)
            self.is_loaded = True

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _generate_demo_prediction(
        self, patient_data: Dict[str, Any], time_points: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generate a simulated prediction for demo purposes when no model is loaded.

        Args:
            patient_data: Dictionary containing patient modality data
            time_points: Optional list of time points for survival curve

        Returns:
            Dictionary with simulated prediction results
        """
        import random

        logger.info("Generating demo prediction (no model loaded)")

        if time_points is None:
            time_points = [365, 730, 1095, 1825, 3650]  # 1, 2, 3, 5, 10 years

        # Generate a random risk score based on clinical data if available
        risk_value = random.uniform(-1.0, 1.0)

        # Adjust based on clinical features if present
        clinical = patient_data.get("clinical")
        if clinical is not None:
            if isinstance(clinical, np.ndarray) and len(clinical) > 0:
                # Use age as a rough risk modifier
                age_factor = (clinical[0] - 50) / 50 if clinical[0] > 0 else 0
                risk_value += age_factor * 0.3

        risk_value = float(np.clip(risk_value, -2, 2))

        # Generate survival probabilities
        baseline_hazard = 0.001
        survival_probs = []
        for t in time_points:
            surv_prob = np.exp(-np.exp(risk_value) * baseline_hazard * t)
            survival_probs.append(float(np.clip(surv_prob, 0.01, 0.99)))

        # Risk stratification
        if risk_value > 0.5:
            risk_group = "high"
        elif risk_value > -0.5:
            risk_group = "medium"
        else:
            risk_group = "low"

        return {
            "risk_score": risk_value,
            "survival_probability": dict(zip(time_points, survival_probs)),
            "risk_group": risk_group,
            "model_checkpoint": "DEMO_MODE (no model loaded)",
        }

    def _generate_demo_attention(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated attention maps for demo mode."""
        import random

        modalities = list(patient_data.keys())

        # Generate random modality importance
        importance = {mod: random.uniform(0.1, 1.0) for mod in modalities}
        total = sum(importance.values())
        importance = {k: v / total for k, v in importance.items()}

        return {
            "cross_modal": None,
            "intra_modal": None,
            "modality_importance": importance,
            "demo_mode": True,
        }

    def load_best_checkpoint(
        self, checkpoints_dir: str = "checkpoints", fold: int = 0
    ) -> bool:
        """
        Automatically find and load the best checkpoint from training.

        Args:
            checkpoints_dir: Base directory containing fold checkpoints
            fold: Fold number to load (default: 0)

        Returns:
            bool: True if loaded successfully
        """
        fold_dir = Path(checkpoints_dir) / f"fold_{fold}"

        if not fold_dir.exists():
            logger.error(f"Fold directory not found: {fold_dir}")
            return False

        # Find best checkpoint (usually named with val_c_index)
        ckpts = list(fold_dir.glob("*.ckpt"))
        if not ckpts:
            logger.error(f"No checkpoints in {fold_dir}")
            return False

        # Sort by modification time or parse metric from filename
        # Prefer files with higher c_index in name
        def extract_metric(p):
            name = p.stem
            if "c_index" in name or "val_c" in name:
                try:
                    # Extract number after c_index or val_c
                    parts = name.split("=")
                    for i, part in enumerate(parts):
                        if "c_index" in part or "val_c" in part:
                            metric_str = parts[i + 1].split("-")[0]
                            return float(metric_str)
                except:
                    pass
            return 0.0

        best_ckpt = max(ckpts, key=extract_metric)
        return self.load_checkpoint(str(best_ckpt))

    def _prepare_input(self, patient_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert patient data dict to model input tensors.

        Args:
            patient_data: Dictionary with modality data
                - clinical: dict or array of clinical features
                - rna: array of gene expression values
                - methylation: array of methylation values
                - mutations: array of mutation indicators
                - wsi: array of WSI features (pre-extracted)

        Returns:
            Dict of tensors ready for model input
        """
        inputs = {}

        for key, value in patient_data.items():
            if key in ["time", "event", "case_id"]:
                continue

            if value is None:
                continue

            # Convert to tensor
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
            elif isinstance(value, (list, tuple)):
                tensor = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                tensor = value.float()
            else:
                continue

            # Add batch dimension if needed
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            inputs[key] = tensor.to(self.device)

        return inputs

    def predict_survival(
        self, patient_data: Dict[str, Any], time_points: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Predict survival for a patient.

        Args:
            patient_data: Dictionary containing patient modality data
            time_points: Optional list of time points for survival curve

        Returns:
            Dictionary containing:
                - risk_score: Predicted hazard/risk score
                - survival_probability: Estimated survival probabilities at time_points
                - risk_group: Risk stratification ('high', 'medium', 'low')
        """
        if not self.is_loaded:
            if self.demo_mode:
                return self._generate_demo_prediction(patient_data, time_points)
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        # Prepare inputs
        inputs = self._prepare_input(patient_data)

        if not inputs:
            raise ValueError("No valid modality data provided")

        # Run inference
        with torch.no_grad():
            risk_score = self.model.model(inputs)

        risk_value = risk_score.squeeze().cpu().item()

        # Convert to survival probability (simplified exponential model)
        # S(t) = exp(-risk * baseline_hazard * t)
        # Using normalized risk score for interpretation
        if time_points is None:
            time_points = [365, 730, 1095, 1825, 3650]  # 1, 2, 3, 5, 10 years

        # Baseline hazard approximation (could be learned from training data)
        baseline_hazard = 0.001  # Approximate daily hazard
        survival_probs = []

        for t in time_points:
            # S(t) ≈ exp(-exp(risk_score) * baseline_hazard * t)
            surv_prob = np.exp(-np.exp(risk_value) * baseline_hazard * t)
            survival_probs.append(float(np.clip(surv_prob, 0, 1)))

        # Risk stratification based on score
        if risk_value > 0.5:
            risk_group = "high"
        elif risk_value > -0.5:
            risk_group = "medium"
        else:
            risk_group = "low"

        return {
            "risk_score": float(risk_value),
            "survival_probability": dict(zip(time_points, survival_probs)),
            "risk_group": risk_group,
            "model_checkpoint": self.checkpoint_path,
        }

    def get_attention_maps(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract attention weights from the model for interpretability.

        Args:
            patient_data: Dictionary containing patient modality data

        Returns:
            Dictionary containing:
                - cross_modal: Cross-modal attention weights from Perceiver
                - intra_modal: Per-modality attention weights (e.g., gene attention)
                - modality_importance: Aggregated importance per modality
        """
        if not self.is_loaded:
            if self.demo_mode:
                demo_pred = self._generate_demo_prediction(patient_data)
                demo_attn = self._generate_demo_attention(patient_data)
                return {
                    "risk_score": demo_pred["risk_score"],
                    "risk_group": demo_pred["risk_group"],
                    "survival_probability": demo_pred["survival_probability"],
                    "attention": demo_attn,
                }
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        # Prepare inputs
        inputs = self._prepare_input(patient_data)

        if not inputs:
            raise ValueError("No valid modality data provided")

        # Run inference with attention
        with torch.no_grad():
            risk_score, attention = self.model.model(inputs, return_attention=True)

        result = {
            "risk_score": float(risk_score.squeeze().cpu().item()),
            "attention": {},
        }

        # Process cross-modal attention
        if "cross_modal" in attention and attention["cross_modal"] is not None:
            cross_attn = attention["cross_modal"]
            if isinstance(cross_attn, torch.Tensor):
                result["attention"]["cross_modal"] = cross_attn.cpu().numpy().tolist()
            elif isinstance(cross_attn, dict):
                result["attention"]["cross_modal"] = {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in cross_attn.items()
                }

        # Process intra-modal attention
        modality_importance = {}
        for mod_key, mod_attn in attention.items():
            if mod_key == "cross_modal":
                continue
            if mod_attn is not None:
                if isinstance(mod_attn, torch.Tensor):
                    attn_np = mod_attn.cpu().numpy()
                    result["attention"][mod_key] = attn_np.tolist()
                    # Compute importance as mean attention
                    modality_importance[mod_key] = float(np.mean(attn_np))

        # Compute overall modality importance from cross-modal attention
        if "cross_modal" in result["attention"]:
            cross = result["attention"]["cross_modal"]
            if isinstance(cross, list) and len(cross) > 0:
                # Assuming attention shape relates to modality tokens
                # This is a simplified interpretation
                n_modalities = len(inputs)
                total_attn = np.array(cross).mean()
                for i, mod in enumerate(inputs.keys()):
                    if mod not in modality_importance:
                        modality_importance[mod] = float(total_attn / n_modalities)

        result["modality_importance"] = modality_importance

        return result

    def batch_predict(self, patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple patients.

        Args:
            patients: List of patient data dictionaries

        Returns:
            List of prediction results
        """
        results = []
        for patient in patients:
            try:
                result = self.predict_survival(patient)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results

    @property
    def available_modalities(self) -> List[str]:
        """Return list of modalities the model was trained with."""
        if not self.is_loaded:
            return []
        return list(self.model.model.encoders.keys())

    @property
    def model_config(self) -> Dict[str, Any]:
        """Return the model configuration."""
        if not self.is_loaded:
            return {}
        return dict(self.model.config)


# Convenience function for quick inference
def quick_predict(
    patient_data: Dict[str, Any], checkpoint_path: str = "checkpoints/fold_0"
) -> Dict[str, Any]:
    """
    Quick prediction function for single patient.

    Args:
        patient_data: Patient modality data
        checkpoint_path: Path to checkpoint

    Returns:
        Prediction result dictionary
    """
    service = MOSAICModelService()
    service.load_checkpoint(checkpoint_path)
    return service.predict_survival(patient_data)
