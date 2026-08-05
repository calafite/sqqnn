import numpy as np
from typing import Dict

class ModelMetrics:
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.square(y_true - y_pred)))

    @staticmethod
    def hinge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        loss: np.ndarray = np.maximum(0.0, 1.0 - y_pred * y_true)
        return float(np.mean(loss))

    @staticmethod
    def compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_bin: np.ndarray = np.where(y_pred >= 0.0, 1.0, -1.0)

        tp_count: float = float(np.sum((y_bin == 1.0) & (y_true == 1.0)))
        tn_count: float = float(np.sum((y_bin == -1.0) & (y_true == -1.0)))
        fp_count: float = float(np.sum((y_bin == 1.0) & (y_true == -1.0)))
        fn_count: float = float(np.sum((y_bin == -1.0) & (y_true == 1.0)))

        all_count: float = tp_count + tn_count + fp_count + fn_count
        pos_pred: float = tp_count + fp_count
        pos_true: float = tp_count + fn_count
        neg_true: float = tn_count + fp_count

        acc_score: float = (tp_count + tn_count) / all_count if all_count > 0 else 0.0
        prec_score: float = tp_count / pos_pred if pos_pred > 0 else 0.0
        sens_score: float = tp_count / pos_true if pos_true > 0 else 0.0
        spec_score: float = tn_count / neg_true if neg_true > 0 else 0.0

        f1_score: float = 0.0
        if prec_score + sens_score > 0:
            f1_score = 2.0 * (prec_score * sens_score) / (prec_score + sens_score)

        return {
            "accuracy": acc_score,
            "precision": prec_score,
            "sensitivity": sens_score,
            "specificity": spec_score,
            "f1_score": f1_score
        }
