import numpy as np

from core.projections import ProjectionOperations, TrigState

Gradient = np.ndarray | float


class BaseNet:
    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights: np.ndarray | None = weights

    @staticmethod
    def with_weights(w_arr: np.ndarray) -> "BaseNet":
        return BaseNet(weights=w_arr)

    def compute_beta(self, x_mat: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("weights not set")
        return np.asarray(x_mat @ self.weights)

    def predict(self, x_mat: np.ndarray) -> np.ndarray:
        beta_value: np.ndarray = self.compute_beta(x_mat)
        return np.cos(beta_value)


class ClassNet(BaseNet):
    @staticmethod
    def with_weights(w_arr: np.ndarray) -> "ClassNet":
        return ClassNet(weights=w_arr)

    def predict(self, x_mat: np.ndarray) -> np.ndarray:
        beta_value: np.ndarray = self.compute_beta(x_mat)
        return np.tanh(beta_value)


class RegNet:
    def __init__(
        self,
        w_alpha: np.ndarray | None = None,
        w_beta: np.ndarray | None = None,
        w_gamma: np.ndarray | None = None,
        theta: float = 0.0,
        omega: float = 0.0,
    ) -> None:
        self.w_alpha: np.ndarray | None = w_alpha
        self.w_beta: np.ndarray | None = w_beta
        self.w_gamma: np.ndarray | None = w_gamma
        self.theta: float = theta
        self.omega: float = omega

    @staticmethod
    def with_parameters(
        w_alpha: np.ndarray,
        w_beta: np.ndarray,
        w_gamma: np.ndarray,
        theta: float,
        omega: float,
    ) -> "RegNet":
        return RegNet(w_alpha, w_beta, w_gamma, theta, omega)

    def _get_angles(
        self, x_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.w_alpha is None or self.w_beta is None or self.w_gamma is None:
            raise ValueError("weights missing")
        return x_mat @ self.w_alpha, x_mat @ self.w_beta, x_mat @ self.w_gamma

    def predict(self, x_mat: np.ndarray) -> np.ndarray:
        alpha, beta, gamma = self._get_angles(x_mat)
        state: TrigState = ProjectionOperations.make_state(
            alpha, beta, gamma, self.theta, self.omega
        )
        return ProjectionOperations.compute_output(state)

    def compute_gradients(
        self, x_mat: np.ndarray, y_raw: np.ndarray
    ) -> tuple[dict[str, Gradient], float]:
        alpha, beta, gamma = self._get_angles(x_mat)
        state: TrigState = ProjectionOperations.make_state(
            alpha, beta, gamma, self.theta, self.omega
        )

        y_line: np.ndarray = ProjectionOperations.compute_output(state)
        diff_value: np.ndarray = y_line - y_raw

        mse_loss: float = float(np.mean(diff_value**2))

        da, db, dg, dt, dw = ProjectionOperations.compute_derivatives(state)

        n_samples: int = x_mat.shape[0]
        factor: float = 2.0 / n_samples

        g_alpha: np.ndarray = factor * (diff_value * da) @ x_mat
        g_beta: np.ndarray = factor * (diff_value * db) @ x_mat
        g_gamma: np.ndarray = factor * (diff_value * dg) @ x_mat

        g_theta: float = float(factor * np.sum(diff_value * dt))
        g_omega: float = float(factor * np.sum(diff_value * dw))

        gradients: dict[str, Gradient] = {
            "w_alpha": g_alpha,
            "w_beta": g_beta,
            "w_gamma": g_gamma,
            "theta": g_theta,
            "omega": g_omega,
        }

        return gradients, mse_loss
