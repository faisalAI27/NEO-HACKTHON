import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

from src.losses.cox_loss import CoxPHLoss
from src.models.mosaic import MOSAIC


class MOSAICTrainer(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
    ):
        """
        PyTorch Lightning Module for MOSAIC model.

        Args:
            model_config (dict): Configuration dictionary for MOSAIC model.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer.
            max_epochs (int): Max epochs for scheduler logic.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = model_config
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize Model
        self.model = MOSAIC(config=model_config)

        # Initialize Loss
        self.criterion = CoxPHLoss()

        # Storage for validation metrics
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Batch is expected to be a dict with keys: 'rna', 'wsi', 'clinical', ... and 'time', 'event'
        # But 'rna', 'wsi' etc might be in a sub-dict or flat.
        # Assuming the Dataset returns a flat dict where keys match model input keys,
        # plus 'time' and 'event'.

        # Separate inputs from targets
        inputs = {
            k: v for k, v in batch.items() if k not in ["time", "event", "case_id"]
        }
        time = batch["time"]
        event = batch["event"]

        # Forward pass
        risk_scores = self.model(inputs)

        # Compute Loss
        loss = self.criterion(risk_scores, time, event)

        # Log
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {
            k: v for k, v in batch.items() if k not in ["time", "event", "case_id"]
        }
        time = batch["time"]
        event = batch["event"]

        risk_scores = self.model(inputs)
        val_loss = self.criterion(risk_scores, time, event)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Store for C-index calculation
        self.validation_step_outputs.append(
            {
                "risk_scores": risk_scores.detach().cpu(),
                "time": time.detach().cpu(),
                "event": event.detach().cpu(),
            }
        )

        return val_loss

    def on_validation_epoch_end(self):
        # Aggregate logic
        if not self.validation_step_outputs:
            return

        all_risk = (
            torch.cat([x["risk_scores"] for x in self.validation_step_outputs])
            .numpy()
            .flatten()
        )
        all_time = (
            torch.cat([x["time"] for x in self.validation_step_outputs])
            .numpy()
            .flatten()
        )
        all_event = (
            torch.cat([x["event"] for x in self.validation_step_outputs])
            .numpy()
            .flatten()
        )

        # Clear memory
        self.validation_step_outputs.clear()

        # C-Index (Censoring handled by lifelines: event_observed=all_event)
        # Note: concordance_index(event_times, predicted_scores, event_observed)
        # BUT: For CoxPH, higher risk score = shorter survival.
        # C-index expects high score = high survival time (concordance).
        # So we should pass -risk_scores as the prediction variable for C-index if we want "standard" C-index
        # where 1.0 is perfect.
        # WAIT. DeepSurv / PyCox / lifelines documentation:
        # lifelines.utils.concordance_index(event_times, predicted_scores, event_observed=None)
        # "If predicted_scores is a risk score (higher => earlier failure), then C-index < 0.5.
        # Usually we want C-index > 0.5. So use -predicted_scores."

        try:
            c_index = concordance_index(all_time, -all_risk, event_observed=all_event)
        except Exception as e:
            print(f"Error computing C-index: {e}")
            c_index = 0.5

        self.log("val_c_index", c_index, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # Separate learning rates if needed (e.g. lower for WSI encoder) could be done here.
        # For now, simple AdamW.
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
