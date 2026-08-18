from typing import Any, Sequence

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, make_moons
from sklearn.model_selection import train_test_split

from tracking.records import Hyperparameters, RunRecord
from tracking.tracker import ExperimentTracker
from training.pipeline import ClassificationPipeline, RegressionPipeline
from utils.search import GridSearch
from utils.styling import OutputFormatter
from visualisation.plots import Visualiser


class Helpers:
    @staticmethod
    def get_xor_data() -> tuple[np.ndarray, np.ndarray]:
        x_mat: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_mat: np.ndarray = np.array([-1.0, 1.0, 1.0, -1.0])
        return x_mat, y_mat

    @staticmethod
    def get_sinc_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train: np.ndarray = np.linspace(-15, 15, 800).reshape(-1, 1)
        y_train: np.ndarray = np.divide(
            np.sin(x_train), x_train, out=np.ones_like(x_train), where=x_train != 0
        ).flatten()

        x_test: np.ndarray = np.linspace(-15, 15, 100).reshape(-1, 1)
        y_test: np.ndarray = np.divide(
            np.sin(x_test), x_test, out=np.ones_like(x_test), where=x_test != 0
        ).flatten()

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_moons_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_raw, y_raw = make_moons(n_samples=1100, noise=0.07, random_state=3)

        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_raw, np.ndarray)

        y_trans: np.ndarray = np.where(y_raw == 0, -1.0, 1.0)

        x_tr, x_te, y_tr, y_te = train_test_split(x_raw, y_trans, test_size=100, random_state=42)

        assert isinstance(x_tr, np.ndarray) and isinstance(x_te, np.ndarray)
        assert isinstance(y_tr, np.ndarray) and isinstance(y_te, np.ndarray)

        return x_tr, y_tr, x_te, y_te

    @staticmethod
    def get_wbcd_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_raw, y_labels = load_breast_cancer(return_X_y=True)

        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_labels, np.ndarray)

        y_raw: np.ndarray = np.where(y_labels == 0, -1.0, 1.0)

        x_tr, x_te, y_tr, y_te = train_test_split(x_raw, y_raw, test_size=0.2, random_state=42)

        assert isinstance(x_tr, np.ndarray) and isinstance(x_te, np.ndarray)
        assert isinstance(y_tr, np.ndarray) and isinstance(y_te, np.ndarray)

        return x_tr, y_tr, x_te, y_te

    @staticmethod
    def get_digits_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_raw, y_raw = load_digits(return_X_y=True)

        assert isinstance(x_raw, np.ndarray)
        assert isinstance(y_raw, np.ndarray)

        x_tr, x_te, y_tr, y_te = train_test_split(x_raw, y_raw, test_size=0.2, random_state=42)

        assert isinstance(x_tr, np.ndarray) and isinstance(x_te, np.ndarray)
        assert isinstance(y_tr, np.ndarray) and isinstance(y_te, np.ndarray)

        return x_tr, y_tr, x_te, y_te


class Tester:
    def _run_search(
        self,
        title: str,
        pipeline: Any,
        configs: Sequence[Hyperparameters],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        maximise: bool,
    ) -> tuple[RunRecord, ExperimentTracker]:
        OutputFormatter.title(title)
        tracker = ExperimentTracker(maximise_metric=maximise)

        for config in configs:
            record: RunRecord = pipeline.execute(config, x_train, y_train, x_val, y_val)
            tracker.log_run(record)

        best_run: RunRecord | None = tracker.get_best_run()
        if best_run is None:
            raise RuntimeError("No configurations were successfully evaluated.")

        OutputFormatter.config("Best Degree", best_run.config.degree)

        if not maximise:
            OutputFormatter.config("Best Learning Rate", best_run.config.learning_rate)
            OutputFormatter.config("Best Epochs", best_run.config.epochs)

        OutputFormatter.metric("Primary Score", best_run.metrics.primary_score)
        print("")

        return best_run, tracker

    def run_xor(self) -> None:
        x_data, y_data = Helpers.get_xor_data()
        configs = GridSearch.generate(
            degree=[1],
            epochs=[79],
            learning_rate=[0.05]
        )

        best_run, _ = self._run_search(
            title="XOR GATE | REGRESSION",
            pipeline=RegressionPipeline(),
            configs=configs,
            x_train=x_data,
            y_train=y_data,
            x_val=x_data,
            y_val=y_data,
            maximise=False
        )
        Visualiser.plot_convergence(best_run, title="XOR Convergence Over Time")

    def run_sinc(self) -> None:
        x_train, y_train, x_test, y_test = Helpers.get_sinc_data()
        configs = GridSearch.generate(
            degree=[1],
            epochs=[2000],
            learning_rate=[0.01, 0.001],
            scale_x=[False, True]
        )

        best_run, _ = self._run_search(
            title="SINC FUNCTION | REGRESSION",
            pipeline=RegressionPipeline(),
            configs=configs,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            maximise=False
        )

        Visualiser.plot_convergence(best_run, title="Sinc Convergence Over Time")

    def run_moons(self) -> None:
        x_train, y_train, x_test, y_test = Helpers.get_moons_data()
        configs = GridSearch.generate(degree=[1, 2, 3, 4])

        best_run, tracker = self._run_search(
            title="TWO MOONS | CLASSIFICATION",
            pipeline=ClassificationPipeline(),
            configs=configs,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            maximise=True
        )
        Visualiser.plot_hyperparameter_comparison(
            tracker.all_runs(), "degree", "Moons Dataset: Impact of Basis Degree"
        )
        Visualiser.plot_decision_boundary(
            best_run.model, x_test, y_test, title="Two Moons Decision Boundary"
        )

    def run_wbcd(self) -> None:
        x_train, y_train, x_test, y_test = Helpers.get_wbcd_data()
        configs = GridSearch.generate(
            degree=[1, 2, 3, 4],
            scale_x=[True, False]
        )

        best_run, tracker = self._run_search(
            title="WBCD | CLASSIFICATION",
            pipeline=ClassificationPipeline(),
            configs=configs,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            maximise=True
        )
        Visualiser.plot_hyperparameter_comparison(
            tracker.all_runs(), "degree", "WBCD: Impact of Basis Degree"
        )

        for key, value in best_run.metrics.additional_stats.items():
            OutputFormatter.metric(key.title(), value)

    def run_digits(self) -> None:
        x_train, y_train, x_test, y_test = Helpers.get_digits_data()
        configs = GridSearch.generate(
            degree=[1, 2, 3],
            scale_x=[True]
        )

        _, tracker = self._run_search(
            title="DIGITS | MULTI-CLASS CLASSIFICATION",
            pipeline=ClassificationPipeline(),
            configs=configs,
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            maximise=True
        )

        Visualiser.plot_hyperparameter_comparison(
            tracker.all_runs(), "degree", "Digits: Impact of Basis Degree"
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.run_xor()
        self.run_sinc()
        self.run_moons()
        self.run_wbcd()
        self.run_digits()


if __name__ == "__main__":
    tester = Tester()
    tester()
