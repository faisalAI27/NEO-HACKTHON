from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


class SurvivalVisualization:
    """
    Class for generating survival analysis visualizations, including
    Kaplan-Meier curves and calibration plots.
    """

    def __init__(self, style: str = "whitegrid"):
        """
        Initialize the visualizer.

        Args:
            style (str): Seaborn style to use.
        """
        sns.set_style(style)

    def plot_kaplan_meier_by_risk_group(
        self,
        event_indicator: np.ndarray,
        time_to_event: np.ndarray,
        risk_scores: np.ndarray,
        save_path: Optional[str] = None,
        group_method: str = "median",
        title: str = "Kaplan-Meier Survival Curves by Risk Group",
    ):
        """
        Splits patients into risk groups and plots Kaplan-Meier curves.

        Args:
            event_indicator (np.ndarray): Binary array indicating if event occurred.
            time_to_event (np.ndarray): Array of time to event.
            risk_scores (np.ndarray): Predicted risk scores (higher = higher risk).
            save_path (Optional[str]): Path to save the figure.
            group_method (str): Method to split groups ('median' or 'quartiles').
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        kmf = KaplanMeierFitter()

        # Ensure boolean event indicator
        event_bool = event_indicator.astype(bool)

        if group_method == "median":
            threshold = np.median(risk_scores)
            is_high_risk = risk_scores > threshold
            groups = {"Low Risk": ~is_high_risk, "High Risk": is_high_risk}
        elif group_method == "quartiles":
            q1, q3 = np.percentile(risk_scores, [25, 75])
            groups = {
                "Low Risk (< Q1)": risk_scores < q1,
                "Medium Risk": (risk_scores >= q1) & (risk_scores <= q3),
                "High Risk (> Q3)": risk_scores > q3,
            }
        else:
            raise ValueError(f"Unknown group_method: {group_method}")

        # Iterate over groups and plot
        for label, mask in groups.items():
            if np.sum(mask) > 0:
                kmf.fit(time_to_event[mask], event_bool[mask], label=label)
                kmf.plot_survival_function(ci_show=True)

        # Log-rank test (only if exactly 2 groups for simplicity in title, though can calculate pairwise)
        if len(groups) == 2:
            masks = list(groups.values())
            results = logrank_test(
                time_to_event[masks[0]],
                time_to_event[masks[1]],
                event_observed_A=event_bool[masks[0]],
                event_observed_B=event_bool[masks[1]],
            )
            p_value = results.p_value
            plt.title(f"{title}\nLog-rank p-value: {p_value:.4e}")
        else:
            plt.title(title)

        plt.xlabel("Time (days)")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_calibration_at_time(
        self,
        event_indicator: np.ndarray,
        time_to_event: np.ndarray,
        survival_probs_at_time: np.ndarray,
        time_point: float,
        n_bins: int = 5,
        save_path: Optional[str] = None,
    ):
        """
        Plots calibration curve at a specific time point.

        Args:
            event_indicator (np.ndarray): Binary array indicating if event occurred.
            time_to_event (np.ndarray): Array of time to event.
            survival_probs_at_time (np.ndarray): Predicted survival probabilities at 'time_point'.
            time_point (float): The time point for evaluation.
            n_bins (int): Number of bins for calibration.
            save_path (Optional[str]): Path to save the figure.
        """
        # Create bins based on predicted survival probabilities
        # We want to compare Predicted Survival Probability vs Observed Survival Probability (KM)

        # 1. Bin data
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(survival_probs_at_time, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        pred_probs = []
        obs_probs = []
        counts = []

        kmf = KaplanMeierFitter()

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                # Mean predicted probability in this bin
                mean_pred = np.mean(survival_probs_at_time[mask])
                pred_probs.append(mean_pred)

                # Observe survival at time_point using KM
                # We need to fit KM on this subset
                kmf.fit(time_to_event[mask], event_indicator[mask])

                # Get survival at 'time_point'
                # survival_function_ returns dataframe with index as time
                try:
                    obs_surv = kmf.predict(time_point)
                except:
                    # Fallback if time_point is beyond max observed time in this bin
                    obs_surv = (
                        kmf.survival_function_.iloc[-1, 0]
                        if not kmf.survival_function_.empty
                        else 0.0
                    )

                obs_probs.append(obs_surv)
                counts.append(np.sum(mask))
            else:
                pass

        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.plot(pred_probs, obs_probs, "o-", label="Model")

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(f"Predicted Survival Probability at t={time_point}")
        plt.ylabel(f"Observed Survival Probability at t={time_point}")
        plt.title(f"Calibration Curve (t={time_point} days)")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_risk_distribution(
        self, risk_scores: np.ndarray, save_path: Optional[str] = None
    ):
        """
        Plots the distribution of risk scores.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(risk_scores, kde=True)
        plt.title("Distribution of Predicted Risk Scores")
        plt.xlabel("Risk Score")
        plt.ylabel("Count")

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
