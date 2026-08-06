from typing import Any

import numpy as np

from core.network import RegNet
from utils.shared import SharedRNG


class LlsSolver:
    def __init__(self, epsilon: float = 1e-16) -> None:
        self.epsilon: float = epsilon

    def fit(self, x_mat: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
        y_clip: np.ndarray = np.copy(y_raw)

        # clipping values to avoid singularity in arctanh
        y_clip[y_raw == -1] = -1.0 + self.epsilon
        y_clip[y_raw == 1] = 1.0 - self.epsilon

        y_trans: np.ndarray = np.arctanh(y_clip)

        # weight computation through pseudo inverse projection
        x_pinv: np.ndarray = np.linalg.pinv(x_mat)
        return x_pinv @ y_trans


class GdSolver:
    def __init__(self, step_size: float = 0.01, max_epochs: int = 100) -> None:
        self.step_size: float = step_size
        self.max_epochs: int = max_epochs

    def fit(self, net: RegNet, x_mat: np.ndarray, y_raw: np.ndarray) -> RegNet:
        n_features: int = x_mat.shape[1]
        rng = SharedRNG.rng

        w_alpha: np.ndarray = (
            net.w_alpha
            if net.w_alpha is not None
            else rng.uniform(-0.1, 0.1, size=n_features)
        )
        w_beta: np.ndarray = (
            net.w_beta
            if net.w_beta is not None
            else rng.uniform(-0.1, 0.1, size=n_features)
        )
        w_gamma: np.ndarray = (
            net.w_gamma
            if net.w_gamma is not None
            else rng.uniform(-0.1, 0.1, size=n_features)
        )
        theta: float = (
            net.theta
            if net.theta != 0.0
            else float(rng.uniform(-0.1, 0.1))
        )
        omega: float = (
            net.omega
            if net.omega != 0.0
            else float(rng.uniform(-0.1, 0.1))
        )

        current_net: RegNet = net.with_parameters(
            w_alpha, w_beta, w_gamma, theta, omega
        )

        for _ in range(self.max_epochs):
            gradients: dict[str, Any] = current_net.compute_gradients(x_mat, y_raw)

            w_alpha = w_alpha - self.step_size * gradients["w_alpha"]
            w_beta = w_beta - self.step_size * gradients["w_beta"]
            w_gamma = w_gamma - self.step_size * gradients["w_gamma"]
            theta = theta - self.step_size * gradients["theta"]
            omega = omega - self.step_size * gradients["omega"]

            current_net = current_net.with_parameters(
                w_alpha, w_beta, w_gamma, theta, omega
            )

        return current_net
