import enum

import numpy as np


class BasisType(enum.Enum):
    STANDARD = enum.auto()
    CHEBYSHEV = enum.auto()
    FOURIER = enum.auto()


class FeatureMap:
    def __init__(self, degree: int = 1, basis: BasisType = BasisType.STANDARD) -> None:
        self.degree: int = degree
        self.basis: BasisType = basis

    @staticmethod
    def with_config(degree: int, basis: BasisType) -> "FeatureMap":
        return FeatureMap(degree=degree, basis=basis)

    def build_matrix(self, x_raw: np.ndarray) -> np.ndarray:
        if self.basis == BasisType.STANDARD:
            return self._build_standard(x_raw)
        if self.basis == BasisType.CHEBYSHEV:
            return self._build_chebyshev(x_raw)
        return self._build_fourier(x_raw)

    def _build_standard(self, x_raw: np.ndarray) -> np.ndarray:
        n_rows, n_cols = x_raw.shape
        out_cols: int = 1 + n_cols * self.degree
        x_mat: np.ndarray = np.ones((n_rows, out_cols))

        for k in range(1, self.degree + 1):
            start_idx: int = 1 + (k - 1) * n_cols
            end_idx: int = 1 + k * n_cols
            x_pow: np.ndarray = np.power(x_raw, k)
            x_mat[:, start_idx:end_idx] = x_pow

        return x_mat

    def _build_chebyshev(self, x_raw: np.ndarray) -> np.ndarray:
        n_rows, n_cols = x_raw.shape
        out_cols: int = 1 + n_cols * self.degree
        x_mat: np.ndarray = np.ones((n_rows, out_cols))

        if self.degree >= 1:
            # T_1(x) = x
            x_mat[:, 1 : 1 + n_cols] = x_raw

        # T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
        for k in range(2, self.degree + 1):
            start_idx: int = 1 + (k - 1) * n_cols
            end_idx: int = 1 + k * n_cols

            prev_start: int = 1 + (k - 2) * n_cols
            prev_end: int = 1 + (k - 1) * n_cols

            t_prev1: np.ndarray = x_mat[:, prev_start:prev_end]
            t_prev2: np.ndarray = (
                np.ones((n_rows, n_cols))
                if k == 2
                else x_mat[:, 1 + (k - 3) * n_cols : 1 + (k - 2) * n_cols]
            )

            x_mat[:, start_idx:end_idx] = 2.0 * x_raw * t_prev1 - t_prev2

        return x_mat

    def _build_fourier(self, x_raw: np.ndarray) -> np.ndarray:
        n_rows, n_cols = x_raw.shape
        # fourier generates two features (sin, cos) per degree per column
        out_cols: int = 1 + 2 * n_cols * self.degree
        x_mat: np.ndarray = np.ones((n_rows, out_cols))

        for k in range(1, self.degree + 1):
            phase: np.ndarray = k * np.pi * x_raw

            sin_start: int = 1 + 2 * (k - 1) * n_cols
            sin_end: int = sin_start + n_cols
            x_mat[:, sin_start:sin_end] = np.sin(phase)

            cos_start: int = sin_end
            cos_end: int = cos_start + n_cols
            x_mat[:, cos_start:cos_end] = np.cos(phase)

        return x_mat
