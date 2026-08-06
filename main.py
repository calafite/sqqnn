from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import KFold

from data.features import BasisType, FeatureMap
from training.optimisers import AdamOptimiser
from training.pipeline import ClassificationPipeline, RegressionPipeline
from training.solvers import AdamSolver, LlsSolver
from utils.styling import OutputFormatter


class Tester:
    @staticmethod
    def run_xor() -> None:
        OutputFormatter.title("XOR GATE | REGRESSION")
        OutputFormatter.config("Basis", "Standard")
        OutputFormatter.config("Degree", 1)
        OutputFormatter.config("Optimiser", "Adam")
        OutputFormatter.config("Learning Rate", 0.1)
        OutputFormatter.config("L2 Penalty", 0.0)
        OutputFormatter.config("Epochs", 100)
        print("")

        x_raw: np.ndarray = Helpers.xor_X()
        y_raw: np.ndarray = Helpers.xor_Y()

        feature_map: FeatureMap = FeatureMap(degree=1, basis=BasisType.STANDARD)
        optimiser: AdamOptimiser = AdamOptimiser(learning_rate=0.1, l2_penalty=0.0)
        solver: AdamSolver = AdamSolver(optimiser=optimiser, max_epochs=100)

        pipeline: RegressionPipeline = RegressionPipeline().with_config(
            feature_map=feature_map, solver=solver, scale_x=False
        )

        fitted_pipeline: RegressionPipeline = pipeline.fit(x_raw, y_raw)
        mse_value: float = fitted_pipeline.evaluate(x_raw, y_raw)

        OutputFormatter.metric("MSE", mse_value)

    @staticmethod
    def run_sinc() -> None:
        OutputFormatter.title("SINC FUNCTION | REGRESSION")
        OutputFormatter.config("Basis", "Standard")
        OutputFormatter.config("Degree", 1)
        OutputFormatter.config("Optimiser", "Adam")
        OutputFormatter.config("Learning Rate", 0.01)
        OutputFormatter.config("Epochs", 30000)
        OutputFormatter.config("Scale X", False)
        print("")

        x_train: np.ndarray = np.linspace(-15, 15, 800).reshape(-1, 1)
        y_train: np.ndarray = np.divide(
            np.sin(x_train), x_train, out=np.ones_like(x_train), where=x_train != 0
        ).flatten()

        x_test: np.ndarray = np.linspace(-15, 15, 100).reshape(-1, 1)
        y_test: np.ndarray = np.divide(
            np.sin(x_test), x_test, out=np.ones_like(x_test), where=x_test != 0
        ).flatten()

        feature_map: FeatureMap = FeatureMap(degree=1, basis=BasisType.STANDARD)
        optimiser: AdamOptimiser = AdamOptimiser(learning_rate=0.01)
        solver: AdamSolver = AdamSolver(optimiser=optimiser, max_epochs=30000)

        # scale_x=False preserves the -15 to 15 domain, giving the initialized
        # phase angles the required frequency reach to map the high oscillations.
        pipeline: RegressionPipeline = RegressionPipeline().with_config(
            feature_map=feature_map, solver=solver, scale_x=False
        )
        fitted_pipeline: RegressionPipeline = pipeline.fit(x_train, y_train)

        mse_value: float = fitted_pipeline.evaluate(x_test, y_test)

        OutputFormatter.metric("MSE", mse_value)

    @staticmethod
    def run_moons() -> None:
        OutputFormatter.title("TWO MOONS | CLASSIFICATION")
        OutputFormatter.config("Basis", "Standard")
        OutputFormatter.config("Degree", 5)
        OutputFormatter.config("Solver", "LLS (Pseudo-Inverse)")
        OutputFormatter.config("L2 Penalty", 0.0)
        OutputFormatter.config("Scale X", False)
        print("")

        x_raw, y_raw = make_moons(n_samples=1100, noise=0.07, random_state=67)

        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_raw, np.ndarray)

        y_trans: np.ndarray = np.where(y_raw == 0, -1.0, 1.0)

        x_train: np.ndarray = x_raw[:1000]
        y_train: np.ndarray = y_trans[:1000]

        x_test: np.ndarray = x_raw[1000:]
        y_test: np.ndarray = y_trans[1000:]

        feature_map: FeatureMap = FeatureMap(degree=5, basis=BasisType.STANDARD)
        solver: LlsSolver = LlsSolver(l2_penalty=0.0)

        pipeline: ClassificationPipeline = ClassificationPipeline().with_config(
            feature_map=feature_map, solver=solver, scale_x=False
        )
        fitted_pipeline: ClassificationPipeline = pipeline.fit(x_train, y_train)

        stats: dict[str, float] = fitted_pipeline.evaluate(x_test, y_test)

        for metric_name, score in stats.items():
            OutputFormatter.metric(metric_name, score)

    @staticmethod
    def run_wbcd() -> None:
        OutputFormatter.title("WBCD | CLASSIFICATION")
        OutputFormatter.config("Basis", "Standard")
        OutputFormatter.config("Degree", 4)
        OutputFormatter.config("Solver", "LLS (Ridge)")
        OutputFormatter.config("L2 Penalty", 0.001)
        OutputFormatter.config("Validation", "10-Fold CV")
        print("")

        data: Any = load_breast_cancer()
        x_raw: np.ndarray = data.data
        y_raw: np.ndarray = data.target

        y_trans: np.ndarray = np.where(y_raw == 0, -1.0, 1.0)
        k_fold = KFold(n_splits=10, shuffle=True, random_state=67)

        metrics_history: dict[str, list[float]] = {
            "accuracy": [],
            "precision": [],
            "sensitivity": [],
            "specificity": [],
            "f1_score": [],
        }

        for train_idx, test_idx in k_fold.split(x_raw):
            x_train, x_test = x_raw[train_idx], x_raw[test_idx]
            y_train, y_test = y_trans[train_idx], y_trans[test_idx]

            feature_map: FeatureMap = FeatureMap(degree=4, basis=BasisType.STANDARD)
            solver: LlsSolver = LlsSolver(l2_penalty=1e-3)

            pipeline: ClassificationPipeline = ClassificationPipeline().with_config(
                feature_map=feature_map, solver=solver, scale_x=True
            )
            fitted_pipeline: ClassificationPipeline = pipeline.fit(x_train, y_train)
            stats: dict[str, float] = fitted_pipeline.evaluate(x_test, y_test)

            for key_name, score in stats.items():
                metrics_history[key_name].append(score)

        for metric_name, scores in metrics_history.items():
            mean_val: float = float(np.mean(scores))
            std_val: float = float(np.std(scores))
            display_str: str = f"{mean_val:.3f} ± {std_val:.3f}"
            OutputFormatter.metric(metric_name, display_str)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.run_xor()
        self.run_sinc()
        self.run_moons()
        self.run_wbcd()


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
        return np.array([-1.0, 1.0, 1.0, -1.0])


if __name__ == "__main__":
    tester = Tester()
    tester()
