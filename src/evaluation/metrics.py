from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


class SurvivalMetrics:
    """
    Class for computing survival analysis metrics including C-Index,
    Time-dependent AUC, and Integrated Brier Score.
    """

    @staticmethod
    def compute_c_index(
        event_indicator: np.ndarray,
        time_to_event: np.ndarray,
        risk_scores: np.ndarray,
        tied_tol: float = 1e-8,
    ) -> float:
        """
        Computes the Concordance Index (C-index) using scikit-survival.

        Args:
            event_indicator (np.ndarray): Boolean or binary array indicating if event occurred.
            time_to_event (np.ndarray): Array of time to event (or censoring).
            risk_scores (np.ndarray): Predicted risk scores (higher score = higher risk).
            tied_tol (float): Tolerance for tied risk scores.

        Returns:
            float: C-index value.
        """
        # sksurv expects boolean event indicator
        event_bool = event_indicator.astype(bool)
        result = concordance_index_censored(
            event_bool, time_to_event, risk_scores, tied_tol=tied_tol
        )
        return result[0]

    @staticmethod
    def compute_lifelines_c_index(
        event_indicator: np.ndarray, time_to_event: np.ndarray, risk_scores: np.ndarray
    ) -> float:
        """
        Computes C-index using lifelines.

        Args:
            event_indicator (np.ndarray): Binary array indicating if event occurred.
            time_to_event (np.ndarray): Array of time to event.
            risk_scores (np.ndarray): Predicted risk scores.

        Returns:
            float: C-index value.
        """
        # lifelines.utils.concordance_index(event_times, predicted_scores, event_observed)
        # "predicted_scores" is typically expected to be a survival time prediction or similar.
        # If we pass risk scores (hazard), we should negate them because higher risk = lower survival time.
        return concordance_index(time_to_event, -risk_scores, event_indicator)

    @staticmethod
    def compute_time_dependent_auc(
        train_event_indicator: np.ndarray,
        train_time_to_event: np.ndarray,
        test_event_indicator: np.ndarray,
        test_time_to_event: np.ndarray,
        risk_scores: np.ndarray,
        times: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Computes Time-dependent AUC at specified time points using IPCW.

        Args:
            train_event_indicator, train_time_to_event: Training data stats for IPCW estimation.
            test_event_indicator, test_time_to_event: Test data ground truth.
            risk_scores: risk predictions for test data.
            times: time points to evaluate AUC at.

        Returns:
            Tuple[np.ndarray, float]: (AUC at each time point, Mean AUC)
        """
        # Create structured arrays for sksurv
        dtype = [("Status", "?"), ("Survival_in_days", "<f8")]

        y_train = np.array(
            list(zip(train_event_indicator.astype(bool), train_time_to_event)),
            dtype=dtype,
        )
        y_test = np.array(
            list(zip(test_event_indicator.astype(bool), test_time_to_event)),
            dtype=dtype,
        )

        # Ensure times are within valid range (between min and max of test times)
        # sksurv requires times to be within the range of observed event times
        # We'll just pass them through and let the caller or library handle range errors,
        # or we could filter them. For now, trusting the caller.

        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
        return auc, mean_auc

    @staticmethod
    def compute_integrated_brier_score(
        train_event_indicator: np.ndarray,
        train_time_to_event: np.ndarray,
        test_event_indicator: np.ndarray,
        test_time_to_event: np.ndarray,
        survival_probs: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """
        Computes Integrated Brier Score.

        Args:
            train_*: Training data for IPCW.
            test_*: Test data ground truth.
            survival_probs (np.ndarray): Matrix of shape (n_samples, n_times) containing
                                         predicted survival probabilities at 'times'.
            times (np.ndarray): The time points corresponding to columns of survival_probs.

        Returns:
            float: Integrated Brier Score.
        """
        dtype = [("Status", "?"), ("Survival_in_days", "<f8")]
        y_train = np.array(
            list(zip(train_event_indicator.astype(bool), train_time_to_event)),
            dtype=dtype,
        )
        y_test = np.array(
            list(zip(test_event_indicator.astype(bool), test_time_to_event)),
            dtype=dtype,
        )

        score = integrated_brier_score(y_train, y_test, survival_probs, times)
        return score

    def compute_all_metrics(
        self,
        train_data: Dict[str, np.ndarray],
        test_data: Dict[str, np.ndarray],
        predictions: Dict[str, np.ndarray],
        times: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Wrapper to compute all available metrics.

        Args:
            train_data: Dict with 'event' and 'time' keys for training set.
            test_data: Dict with 'event' and 'time' keys for test set.
            predictions: Dict with 'risk_scores' (required) and 'survival_probabilities' (optional).
            times: Array of time points for time-dependent metrics.

        Returns:
            Dict[str, float]: Dictionary of metric names and values.
        """
        results = {}

        e_train, t_train = train_data["event"], train_data["time"]
        e_test, t_test = test_data["event"], test_data["time"]
        risk_scores = predictions["risk_scores"]

        # 1. C-Index
        results["c_index"] = self.compute_c_index(e_test, t_test, risk_scores)

        # 2. Time-Dependent Metrics (if times provided)
        if times is not None:
            try:
                auc, mean_auc = self.compute_time_dependent_auc(
                    e_train, t_train, e_test, t_test, risk_scores, times
                )
                results["mean_auc"] = mean_auc
                for i, t in enumerate(times):
                    results[f"auc_at_{t:.1f}"] = auc[i]
            except Exception as e:
                print(f"Warning: Could not compute Time-dependent AUC: {e}")

            # 3. Brier Score (if survival probabilities provided)
            if "survival_probabilities" in predictions:
                try:
                    surv_probs = predictions["survival_probabilities"]
                    ibs = self.compute_integrated_brier_score(
                        e_train, t_train, e_test, t_test, surv_probs, times
                    )
                    results["integrated_brier_score"] = ibs
                except Exception as e:
                    print(f"Warning: Could not compute Integrated Brier Score: {e}")

        return results
