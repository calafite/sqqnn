from collections.abc import Sequence

import numpy as np

from core.network import RegNet
from training.optimisers import AdamOptimiser, Optimiser, Parameter
from utils.shared import SharedRNG


class LlsSolver:
    def __init__(self, epsilon: float = 1e-16, l2_penalty: float = 0.0) -> None:
        self.epsilon: float = epsilon
        self.l2_penalty: float = l2_penalty

    def fit(self, x_mat: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
        y_clip: np.ndarray = np.clip(y_raw, -1.0 + self.epsilon, 1.0 - self.epsilon)
        y_trans: np.ndarray = np.arctanh(y_clip)

        if self.l2_penalty == 0:
            x_pinv: np.ndarray = np.linalg.pinv(x_mat)
            return np.asarray(x_pinv @ y_trans)

        n_features: int = x_mat.shape[1]
        reg_matrix: np.ndarray = self.l2_penalty * np.eye(n_features)
        gram_matrix: np.ndarray = x_mat.T @ x_mat + reg_matrix
        target_vector: np.ndarray = x_mat.T @ y_trans

        return np.linalg.solve(gram_matrix, target_vector)


class GradientSolver:
    def __init__(
        self, optimiser: Optimiser | None = None, max_epochs: int = 100
    ) -> None:
        self.optimiser: Optimiser = optimiser or AdamOptimiser()
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

            parameters: dict[str, Parameter] = {
                "w_alpha": w_alpha,
                "w_beta": w_beta,
                "w_gamma": w_gamma,
                "theta": theta,
                "omega": omega,
            }

            updated_parameters: dict[str, Parameter] = self.optimiser.update(
                parameters, gradients
            )

            next_alpha = updated_parameters["w_alpha"]
            next_beta = updated_parameters["w_beta"]
            next_gamma = updated_parameters["w_gamma"]
            assert isinstance(next_alpha, np.ndarray)
            assert isinstance(next_beta, np.ndarray)
            assert isinstance(next_gamma, np.ndarray)

            w_alpha, w_beta, w_gamma = next_alpha, next_beta, next_gamma
            theta = float(updated_parameters["theta"])
            omega = float(updated_parameters["omega"])

            current_network = network.with_parameters(
                w_alpha, w_beta, w_gamma, theta, omega
            )

        return current_network, history

    @staticmethod
    def resolve_weights(weights: np.ndarray | None, n_features: int) -> np.ndarray:
        rng = SharedRNG.rng
        return (
            weights
            if weights is not None
            else np.asarray(rng.uniform(-np.pi, np.pi, size=n_features))
        )

    @staticmethod
    def resolve_scalars(value: float) -> float:
        rng = SharedRNG.rng
        return value if value != 0.0 else float(rng.uniform(-np.pi, np.pi))
