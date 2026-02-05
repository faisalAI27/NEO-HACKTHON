import torch
import torch.nn as nn


class CoxPHLoss(nn.Module):
    """
    Negative Log Likelihood Loss for Cox Proportional Hazards Model.

    Formula:
    L = - sum_{i: E_i=1} ( h_i - log( sum_{j \in R_i} exp(h_j) ) )

    where:
    - h_i is the predicted log-hazard (risk score) for patient i.
    - R_i is the risk set at time T_i (all patients j where T_j >= T_i).
    - E_i is the event indicator (1 if event occurred, 0 if censored).
    """

    def __init__(self):
        super().__init__()

    def forward(self, risk_scores, survival_times, events):
        """
        Args:
            risk_scores (Tensor): Predicted risk scores (log hazard), shape (Batch, 1) or (Batch,)
            survival_times (Tensor): Time to event or censoring, shape (Batch, 1) or (Batch,)
            events (Tensor): Event indicator (1=dead, 0=censored), shape (Batch, 1) or (Batch,)

        Returns:
            start_loss (Tensor): Scalar loss value
        """
        # Ensure inputs are flat
        risk_scores = risk_scores.view(-1)
        survival_times = survival_times.view(-1)
        events = events.view(-1)

        # Sort by survival time (descending) explicitly to handle risk sets easily
        # Note: In a batch, we only approximate the risk set.
        # Large batch sizes are preferred for CoxPH.
        sorted_indices = torch.argsort(survival_times, descending=True)

        risk_scores_sorted = risk_scores[sorted_indices]
        events_sorted = events[sorted_indices]

        # Calculate log-sum-exp of risk scores for the risk set
        # Since we sorted by time descending, the risk set for patient i
        # includes all patients 0...i (who have time >= time_i)
        # So we need cumulative sum from 0 to i.

        # Max trick for log sum exp to avoid overflow could be added if needed,
        # but pure exp is standard for basic Cox implementations if values are small.
        risk_exp = torch.exp(risk_scores_sorted)

        cumsum_risk = torch.cumsum(risk_exp, dim=0)

        # Log of the sum
        log_cumsum_risk = torch.log(cumsum_risk + 1e-8)  # eps for stability

        # Calculate Term: h_i - log_cumsum
        # Only for events (E=1)
        # Note: We do not include censored patients in the outer sum,
        # but they ARE included in the risk set (inner sum/cumsum).

        log_likelihood = events_sorted * (risk_scores_sorted - log_cumsum_risk)

        # Negative Log Likelihood
        # Normalize by number of observed events to keep scale consistent
        num_events = torch.sum(events_sorted)

        if num_events == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)

        loss = -torch.sum(log_likelihood) / num_events

        return loss
