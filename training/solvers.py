from typing import Any, Sequence

import numpy as np

from core.network import RegNet
from training.optimisers import AdamOptimiser
from utils.shared import SharedRNG


class LlsSolver:
    def __init__(self, epsilon: float = 1e-16, l2_penalty: float = 0.0) -> None:
        self.epsilon: float = epsilon
        self.l2_penalty: float = l2_penalty

    def fit(self, x_mat: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
        y_clip: np.ndarray = np.copy(y_raw)

        y_clip[y_raw == -1] = -1.0 + self.epsilon
        y_clip[y_raw == 1] = 1.0 - self.epsilon

        y_trans: np.ndarray = np.arctanh(y_clip)

        if self.l2_penalty == 0:
            x_pinv: np.ndarray = np.linalg.pinv(x_mat)
            return x_pinv @ y_trans

        n_features: int = x_mat.shape[1]
        reg_matrix: np.ndarray = self.l2_penalty * np.eye(n_features)
        gram_matrix: np.ndarray = x_mat.T @ x_mat + reg_matrix
        target_vector: np.ndarray = x_mat.T @ y_trans

        return np.linalg.solve(gram_matrix, target_vector)


class AdamSolver:
    def __init__(
        self, optimiser: AdamOptimiser | None = None, max_epochs: int = 100
    ) -> None:
        self.optimiser: AdamOptimiser = optimiser or AdamOptimiser()
        self.max_epochs: int = max_epochs

    def fit(self, network: RegNet, x_mat: np.ndarray, y_raw: np.ndarray) -> tuple[RegNet, Sequence[float]]:
        n_features: int = x_mat.shape[1]

        w_alpha: np.ndarray = self.resolve_weights(network.w_alpha, n_features)
        w_beta: np.ndarray = self.resolve_weights(network.w_beta, n_features)
        w_gamma: np.ndarray = self.resolve_weights(network.w_gamma, n_features)
        theta: float = self.resolve_scalars(network.theta)
        omega: float = self.resolve_scalars(network.omega)

        current_network: RegNet = network.with_parameters(
            w_alpha, w_beta, w_gamma, theta, omega
        )

        history: list[float] = []

        for _ in range(self.max_epochs):
            gradients, loss = current_network.compute_gradients(x_mat, y_raw)
            history.append(loss)

            parameters: dict[str, Any] = {
                "w_alpha": current_network.w_alpha,
                "w_beta": current_network.w_beta,
                "w_gamma": current_network.w_gamma,
                "theta": current_network.theta,
                "omega": current_network.omega,
            }

            updated_parameters: dict[str, Any] = self.optimiser.update(
                parameters, gradients
            )

            current_network = current_network.with_parameters(
                updated_parameters["w_alpha"],
                updated_parameters["w_beta"],
                updated_parameters["w_gamma"],
                float(updated_parameters["theta"]),
                float(updated_parameters["omega"]),
            )

        return current_network, history

    @staticmethod
    def resolve_weights(weights: np.ndarray | None, n_features: int) -> np.ndarray:
        rng: Any = SharedRNG.rng
        return weights if weights is not None else rng.uniform(-np.pi, np.pi, size=n_features)

    @staticmethod
    def resolve_scalars(value: float) -> float:
        rng: Any = SharedRNG.rng
        return value if value != 0.0 else float(rng.uniform(-np.pi, np.pi))
