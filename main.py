import numpy as np
from sklearn.datasets import make_moons

from data.features import FeatureMap
from training.pipeline import ClassificationPipeline, RegressionPipeline
from training.solvers import GdSolver, LlsSolver
from utils.styling import OutputFormatter


class Tester:
    @staticmethod
    def run_xor() -> None:
        x_raw: np.ndarray = Helpers.xor_X()
        y_raw: np.ndarray = Helpers.xor_Y()
        feature_map: FeatureMap = FeatureMap(degree=1)
        solver: GdSolver = GdSolver(step_size=0.1, max_epochs=200)

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
        solver: GdSolver = GdSolver(step_size=0.01, max_epochs=70000)

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
    OutputFormatter.title("XOR GATE | REGRESSION")
    Tester.run_xor()

    OutputFormatter.title("SINC FUNCTION | REGRESSION")
    Tester.run_sinc()

    OutputFormatter.title("TWO MOONS | CLASSIFICATION")
    Tester.run_moons()
