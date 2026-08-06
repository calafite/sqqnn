from typing import Any

import numpy as np
from sklearn.datasets import make_moons

from data.features import FeatureMap
from training.optimisers import AdamOptimiser
from training.pipeline import ClassificationPipeline, RegressionPipeline
from training.solvers import AdamSolver, LlsSolver
from utils.styling import OutputFormatter


class Tester:
    @staticmethod
    def run_xor() -> None:
        x_raw: np.ndarray = Helpers.xor_X()
        y_raw: np.ndarray = Helpers.xor_Y()

        feature_map: FeatureMap = FeatureMap(degree=2)
        optimiser: AdamOptimiser = AdamOptimiser(learning_rate=0.1, l2_penalty=0.001)
        solver: AdamSolver = AdamSolver(optimiser=optimiser, max_epochs=500)

        pipeline: RegressionPipeline = RegressionPipeline().with_config(
            feature_map=feature_map, solver=solver, scale_x=True
        )

        fitted_pipeline: RegressionPipeline = pipeline.fit(x_raw, y_raw)
        mse_value: float = fitted_pipeline.evaluate(x_raw, y_raw)

        OutputFormatter.metric("MSE", mse_value)

    @staticmethod
    def run_sinc() -> None:
        x_train: np.ndarray = np.linspace(-15, 15, 800).reshape(-1, 1)
        y_train: np.ndarray = np.where(
            x_train == 0, 1.0, np.sin(x_train) / x_train
        ).flatten()

        x_test: np.ndarray = np.linspace(-15, 15, 100).reshape(-1, 1)
        y_test: np.ndarray = np.where(
            x_test == 0, 1.0, np.sin(x_test) / x_test
        ).flatten()

        feature_map: FeatureMap = FeatureMap(degree=1)
        optimiser: AdamOptimiser = AdamOptimiser(learning_rate=0.01, l2_penalty=0.0001)
        solver: AdamSolver = AdamSolver(optimiser=optimiser, max_epochs=1000)

        pipeline: RegressionPipeline = RegressionPipeline().with_config(
            feature_map=feature_map, solver=solver
        )
        fitted_pipeline: RegressionPipeline = pipeline.fit(x_train, y_train)

        mse_value: float = fitted_pipeline.evaluate(x_test, y_test)

        OutputFormatter.metric("MSE", mse_value)

    @staticmethod
    def run_moons() -> None:
        x_raw, y_raw = make_moons(n_samples=1100, noise=0.07, random_state=67)

        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_raw, np.ndarray)

        y_trans: np.ndarray = np.where(y_raw == 0, -1.0, 1.0)

        x_train: np.ndarray = x_raw[:1000]
        y_train: np.ndarray = y_trans[:1000]

        x_test: np.ndarray = x_raw[1000:]
        y_test: np.ndarray = y_trans[1000:]

        feature_map: FeatureMap = FeatureMap(degree=4)
        solver: LlsSolver = LlsSolver()

        pipeline: ClassificationPipeline = ClassificationPipeline().with_config(
            feature_map=feature_map, solver=solver
        )
        fitted_pipeline: ClassificationPipeline = pipeline.fit(x_train, y_train)

        stats: dict[str, float] = fitted_pipeline.evaluate(x_test, y_test)

        for metric_name, score in stats.items():
            OutputFormatter.metric(metric_name, score)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        OutputFormatter.title("XOR GATE | REGRESSION")
        self.run_xor()
        OutputFormatter.title("SINC FUNCTION | REGRESSION")
        self.run_sinc()
        OutputFormatter.title("TWO MOONS | CLASSIFICATION")
        self.run_moons()


class Helpers:
    @staticmethod
    def xor_X() -> np.ndarray:
        x_mat: np.ndarray = np.zeros((4, 2))
        index = 0
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                x_mat[index] = [x1, x2]
                index += 1
        return x_mat

    @staticmethod
    def xor_Y() -> np.ndarray:
        y_vec: np.ndarray = np.array([-1.0, 1.0, 1.0, -1.0])
        return y_vec


if __name__ == "__main__":
    tester = Tester()
    tester()
