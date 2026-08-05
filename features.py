import numpy as np

class FeatureMap:
    def __init__(self, degree: int = 1) -> None:
        self.degree: int = degree

    @staticmethod
    def with_degree(degree: int) -> "FeatureMap":
        return FeatureMap(degree=degree)

    def build_matrix(self, x_raw: np.ndarray) -> np.ndarray:
        n_rows, n_cols = x_raw.shape
        out_cols = 1 + n_cols * self.degree

        x_mat: np.ndarray = np.ones((n_rows, out_cols))

        for k in range(1, self.degree + 1):
            start_idx: int = 1 + (k - 1) * n_cols
            end_idx: int = 1 + k * n_cols
            x_pow: np.ndarray = np.power(x_raw, k)
            x_mat[:, start_idx:end_idx] = x_pow

        return x_mat
