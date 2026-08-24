from collections.abc import Sequence

import numpy as np

from tracking.records import Hyperparameters, RunMetrics, RunRecord
from training.pipeline import ClassificationPipeline, RegressionPipeline
from utils.metrics import ModelMetrics
from utils.shared import SharedRNG


class ClassificationEnsemble:
    def __init__(
        self,
        config: Hyperparameters,
        pipelines: Sequence[ClassificationPipeline] | None = None,
        classes: np.ndarray | None = None,
    ) -> None:
        self.config: Hyperparameters = config
        self.pipelines: Sequence[ClassificationPipeline] = pipelines or []
        self.classes: np.ndarray | None = classes

    @classmethod
    def from_config(cls, config: Hyperparameters) -> "ClassificationEnsemble":
        return cls(config=config)

    def fit(
        self, x_raw: np.ndarray, y_raw: np.ndarray
    ) -> tuple["ClassificationEnsemble", Sequence[float]]:
        n_samples: int = x_raw.shape[0]
        rng = SharedRNG.rng
        unique_classes: np.ndarray = np.unique(y_raw)

        fitted_pipelines: list[ClassificationPipeline] = []

        for _ in range(self.config.n_estimators):
            indices: np.ndarray = rng.choice(n_samples, size=n_samples, replace=True)
            x_boot: np.ndarray = x_raw[indices]
            y_boot: np.ndarray = y_raw[indices]

            pipeline: ClassificationPipeline = ClassificationPipeline.from_config(
                self.config, classes=unique_classes
            )
            fitted_pipeline, _ = pipeline.fit(x_boot, y_boot)
            fitted_pipelines.append(fitted_pipeline)

        ensemble = ClassificationEnsemble(
            config=self.config, pipelines=fitted_pipelines, classes=unique_classes
        )

        y_encoded: np.ndarray = (
            y_raw
            if len(unique_classes) <= 2
            else self._encode_multiclass(y_raw, unique_classes)
        )

        train_loss: float = ModelMetrics.hinge(
            y_encoded, ensemble.decision_function(x_raw)
        )
        return ensemble, [train_loss]

    @staticmethod
    def _encode_multiclass(y_raw: np.ndarray, classes: np.ndarray) -> np.ndarray:
        n_samples, n_classes = len(y_raw), len(classes)
        y_encoded: np.ndarray = -np.ones((n_samples, n_classes))
        for i, class_val in enumerate(classes):
            y_encoded[y_raw == class_val, i] = 1.0
        return y_encoded

    def decision_function(self, x_raw: np.ndarray) -> np.ndarray:
        if not self.pipelines:
            raise RuntimeError("Ensemble is not fitted.")

        raw_preds: np.ndarray = np.stack(
            [p.decision_function(x_raw) for p in self.pipelines]
        )
        return np.mean(raw_preds, axis=0)

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        avg_pred: np.ndarray = self.decision_function(x_raw)
        if self.classes is not None and len(self.classes) > 2:
            class_indices: np.ndarray = np.argmax(avg_pred, axis=1)
            return np.asarray(self.classes[class_indices])
        return avg_pred

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, float]:
        y_pred: np.ndarray = self.predict(x_raw)
        is_multi: bool = self.classes is not None and len(self.classes) > 2
        return ModelMetrics.compute_stats(y_raw, y_pred, is_multiclass=is_multi)

    def execute(
        self,
        config: Hyperparameters,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> RunRecord:
        ensemble = self.from_config(config)
        fitted_ensemble, history = ensemble.fit(x_train, y_train)

        stats: dict[str, float] = fitted_ensemble.evaluate(x_val, y_val)
        primary_score: float = stats.get("accuracy", 0.0)

        metrics = RunMetrics(
            primary_score=primary_score,
            additional_stats=stats,
            history={"loss": history},
        )
        return RunRecord(config=config, metrics=metrics, model=fitted_ensemble)


class RegressionEnsemble:
    def __init__(
        self,
        config: Hyperparameters,
        pipelines: Sequence[RegressionPipeline] | None = None,
    ) -> None:
        self.config: Hyperparameters = config
        self.pipelines: Sequence[RegressionPipeline] = pipelines or []

    @classmethod
    def from_config(cls, config: Hyperparameters) -> "RegressionEnsemble":
        return cls(config=config)

    def fit(
        self, x_raw: np.ndarray, y_raw: np.ndarray
    ) -> tuple["RegressionEnsemble", Sequence[float]]:
        n_samples: int = x_raw.shape[0]
        rng = SharedRNG.rng

        fitted_pipelines: list[RegressionPipeline] = []
        ensemble_history: list[float] = []

        for _ in range(self.config.n_estimators):
            indices: np.ndarray = rng.choice(n_samples, size=n_samples, replace=True)
            x_boot: np.ndarray = x_raw[indices]
            y_boot: np.ndarray = y_raw[indices]

            pipeline: RegressionPipeline = RegressionPipeline.from_config(self.config)
            fitted_pipeline, history = pipeline.fit(x_boot, y_boot)
            fitted_pipelines.append(fitted_pipeline)

            if not ensemble_history:
                ensemble_history = list(history)
            else:
                ensemble_history = [e + h for e, h in zip(ensemble_history, history)]

        if ensemble_history:
            ensemble_history = [e / self.config.n_estimators for e in ensemble_history]

        fitted_ensemble = RegressionEnsemble(
            config=self.config, pipelines=fitted_pipelines
        )
        return fitted_ensemble, ensemble_history

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        if not self.pipelines:
            raise RuntimeError("Ensemble is not fitted.")

        preds: np.ndarray = np.stack([p.predict(x_raw) for p in self.pipelines])
        return np.mean(preds, axis=0)

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> float:
        y_pred: np.ndarray = self.predict(x_raw)
        return ModelMetrics.mse(y_raw, y_pred)

    def execute(
        self,
        config: Hyperparameters,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> RunRecord:
        ensemble = self.from_config(config)
        fitted_ensemble, history = ensemble.fit(x_train, y_train)

        mse_score: float = fitted_ensemble.evaluate(x_val, y_val)

        metrics = RunMetrics(
            primary_score=mse_score,
            additional_stats={"mse": mse_score},
            history={"loss": history},
        )
        return RunRecord(config=config, metrics=metrics, model=fitted_ensemble)
