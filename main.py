import numpy as np
from sklearn.datasets import make_moons

from features import FeatureMap
from pipeline import ClassificationPipeline, RegressionPipeline
from solvers import GdSolver, LlsSolver


class Tester:
    @staticmethod
    def run_xor() -> None:
        x_raw: np.ndarray = Helpers.xor_X()
        y_raw: np.ndarray = Helpers.xor_Y()
        feature_map: FeatureMap = FeatureMap(degree=1)
        solver: GdSolver = GdSolver(step_size=0.1, max_epochs=200)

        # activate input scaling exclusively for logic gates
        # to ensure origin-centered symmetry.
        pipeline: RegressionPipeline = RegressionPipeline().with_config(
            feature_map=feature_map, solver=solver, scale_x=True
        )

        fitted_pipeline: RegressionPipeline = pipeline.fit(x_raw, y_raw)
        mse_value: float = fitted_pipeline.evaluate(x_raw, y_raw)
        print(f"xor mse: {mse_value:.4f}")

    @staticmethod
    def run_sinc() -> None:
        # bounds defined by leandro for the sinc function experiment
        x_train: np.ndarray = np.linspace(-15, 15, 800).reshape(-1, 1)
        # handles division by zero at the origin safely
        y_train: np.ndarray = np.where(
            x_train == 0, 1.0, np.sin(x_train) / x_train
        ).flatten()

        x_test: np.ndarray = np.linspace(-15, 15, 100).reshape(-1, 1)
        y_test: np.ndarray = np.where(
            x_test == 0, 1.0, np.sin(x_test) / x_test
        ).flatten()

        feature_map: FeatureMap = FeatureMap(degree=1)
        solver: GdSolver = GdSolver(step_size=0.01, max_epochs=70000)

        # scale_x defaults to False for regressions, preserving the domain scale
        # required for learning the frequencies of the sinc function
        pipeline: RegressionPipeline = RegressionPipeline().with_config(
            feature_map=feature_map, solver=solver
        )
        fitted_pipeline: RegressionPipeline = pipeline.fit(x_train, y_train)

        mse_value: float = fitted_pipeline.evaluate(x_test, y_test)
        print(f"sinc test mse: {mse_value:.4f}")

    @staticmethod
    def run_moons() -> None:
        # synthetic dataset generation
        x_raw, y_raw = make_moons(n_samples=1100, noise=0.07, random_state=67)

        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_raw, np.ndarray)

        # recalibrates labels from [0, 1] to [-1, 1]
        y_trans: np.ndarray = np.where(y_raw == 0, -1.0, 1.0)

        x_train: np.ndarray = x_raw[:1000]
        y_train: np.ndarray = y_trans[:1000]

        x_test: np.ndarray = x_raw[1000:]
        y_test: np.ndarray = y_trans[1000:]

        # increasing the degree provides the non-linearity needed to separate the moons
        feature_map: FeatureMap = FeatureMap(degree=2)
        solver: LlsSolver = LlsSolver()

        # ClassificationPipeline defaults to scale_x=True to avoid ill-conditioned matrices
        pipeline: ClassificationPipeline = ClassificationPipeline().with_config(
            feature_map=feature_map, solver=solver
        )
        fitted_pipeline: ClassificationPipeline = pipeline.fit(x_train, y_train)

        stats: dict[str, float] = fitted_pipeline.evaluate(x_test, y_test)

        print("two moons stats:")
        for metric, score in stats.items():
            print(f"  {metric}: {score:.4f}")


class Helpers:
    @staticmethod
    def xor_X() -> np.ndarray:
        X: np.ndarray = np.zeros((4, 2))
        k = 0
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                X[k] = [x1, x2]
                k += 1
        return X

    @staticmethod
    def xor_Y() -> np.ndarray:
        # targets rescaled to match the quantum measurement bounds
        Y: np.ndarray = np.array([-1.0, 1.0, 1.0, -1.0])
        return Y


if __name__ == "__main__":
    print("evaluating regression on xor gate...")
    Tester.run_xor()

    print("\nevaluating regression on sinc function...")
    Tester.run_sinc()

    print("\nevaluating classification on two moons...")
    Tester.run_moons()
