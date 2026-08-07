from typing import Sequence

import numpy as np

from core.network import ClassNet, RegNet
from data.features import BasisType, FeatureMap
from data.scaling import DataScaler
from tracking.records import Hyperparameters, RunMetrics, RunRecord
from training.optimisers import AdamOptimiser
from training.solvers import AdamSolver, LlsSolver
from utils.metrics import ModelMetrics


class ClassificationPipeline:
    def __init__(
        self,
        feature_map: FeatureMap | None = None,
        x_scale: DataScaler | None = None,
        solver: LlsSolver | None = None,
        net: ClassNet | None = None,
        scale_x: bool = True,
    ) -> None:
        self.feature_map: FeatureMap = feature_map or FeatureMap(degree=1)
        self.x_scale: DataScaler = x_scale or DataScaler()
        self.solver: LlsSolver = solver or LlsSolver()
        self.net: ClassNet = net or ClassNet()
        self.scale_x: bool = scale_x

    @classmethod
    def from_config(cls, config: Hyperparameters) -> "ClassificationPipeline":
        basis_enum: BasisType = BasisType[config.basis.upper()]
        feature_map = FeatureMap(degree=config.degree, basis=basis_enum)
        solver = LlsSolver(l2_penalty=config.l2_penalty)
        return cls(feature_map=feature_map, solver=solver, scale_x=config.scale_x)

    def _prepare_x(
        self, x_raw: np.ndarray, fit_mode: bool = False
    ) -> tuple[np.ndarray, DataScaler]:
        if not self.scale_x:
            return x_raw, self.x_scale

        current_scale: DataScaler = (
            self.x_scale.fit_bounds(x_raw) if fit_mode else self.x_scale
        )
        return current_scale.scale_data(x_raw), current_scale

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> tuple["ClassificationPipeline", Sequence[float]]:
        x_proc, new_scale = self._prepare_x(x_raw, fit_mode=True)
        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)

        weights: np.ndarray = self.solver.fit(x_mat, y_raw)
        new_net: ClassNet = self.net.with_weights(weights)

        fitted_pipeline = ClassificationPipeline(
            self.feature_map, new_scale, self.solver, new_net, self.scale_x
        )

        # evaluate static training loss (LLS calculates optimum directly in one epoch)
        y_pred: np.ndarray = fitted_pipeline.predict(x_raw)
        train_loss: float = ModelMetrics.hinge(y_raw, y_pred)

        return fitted_pipeline, [train_loss]

    def execute(
        self,
        config: Hyperparameters,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> RunRecord:
        pipeline: ClassificationPipeline = self.from_config(config)
        fitted_pipeline, history = pipeline.fit(x_train, y_train)

        stats: dict[str, float] = fitted_pipeline.evaluate(x_val, y_val)
        primary_score: float = stats.get("accuracy", 0.0)

        metrics: RunMetrics = RunMetrics(
            primary_score=primary_score,
            additional_stats=stats,
            history={"loss": history}
        )
        return RunRecord(config=config, metrics=metrics, model=fitted_pipeline)

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_proc, _ = self._prepare_x(x_raw, fit_mode=False)
        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)
        return self.net.predict(x_mat)

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, float]:
        y_pred: np.ndarray = self.predict(x_raw)
        return ModelMetrics.compute_stats(y_raw, y_pred)


class RegressionPipeline:
    def __init__(
        self,
        feature_map: FeatureMap | None = None,
        x_scale: DataScaler | None = None,
        y_scale: DataScaler | None = None,
        solver: AdamSolver | None = None,
        net: RegNet | None = None,
        scale_x: bool = False,
    ) -> None:
        self.feature_map: FeatureMap = feature_map or FeatureMap(degree=1)
        self.x_scale: DataScaler = x_scale or DataScaler()
        self.y_scale: DataScaler = y_scale or DataScaler()
        self.solver: AdamSolver = solver or AdamSolver()
        self.net: RegNet = net or RegNet()
        self.scale_x: bool = scale_x

    @classmethod
    def from_config(cls, config: Hyperparameters) -> "RegressionPipeline":
        basis_enum: BasisType = BasisType[config.basis.upper()]
        feature_map = FeatureMap(degree=config.degree, basis=basis_enum)

        optimiser = AdamOptimiser(learning_rate=config.learning_rate, l2_penalty=config.l2_penalty)
        solver = AdamSolver(optimiser=optimiser, max_epochs=config.epochs)

        return cls(feature_map=feature_map, solver=solver, scale_x=config.scale_x)

    def _prepare_x(
        self, x_raw: np.ndarray, fit_mode: bool = False
    ) -> tuple[np.ndarray, DataScaler]:
        if not self.scale_x:
            return x_raw, self.x_scale

        current_scale: DataScaler = (
            self.x_scale.fit_bounds(x_raw) if fit_mode else self.x_scale
        )
        return current_scale.scale_data(x_raw), current_scale

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> tuple["RegressionPipeline", Sequence[float]]:
        x_proc, new_x_scale = self._prepare_x(x_raw, fit_mode=True)

        y_col: np.ndarray = y_raw.reshape(-1, 1)
        new_y_scale: DataScaler = self.y_scale.fit_bounds(y_col)
        y_norm: np.ndarray = new_y_scale.scale_data(y_col).flatten()

        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)
        new_net, history = self.solver.fit(self.net, x_mat, y_norm)

        fitted_pipeline = RegressionPipeline(
            self.feature_map,
            new_x_scale,
            new_y_scale,
            self.solver,
            new_net,
            self.scale_x,
        )

        return fitted_pipeline, history

    def execute(
        self,
        config: Hyperparameters,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> RunRecord:
        pipeline: RegressionPipeline = self.from_config(config)
        fitted_pipeline, history = pipeline.fit(x_train, y_train)

        mse_score: float = fitted_pipeline.evaluate(x_val, y_val)

        metrics: RunMetrics = RunMetrics(
            primary_score=mse_score,
            additional_stats={"mse": mse_score},
            history={"loss": history}
        )
        return RunRecord(config=config, metrics=metrics, model=fitted_pipeline)

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_proc, _ = self._prepare_x(x_raw, fit_mode=False)
        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)

        y_norm: np.ndarray = self.net.predict(x_mat)
        y_col: np.ndarray = y_norm.reshape(-1, 1)
        return self.y_scale.invert_data(y_col).flatten()

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> float:
        y_pred: np.ndarray = self.predict(x_raw)
        return ModelMetrics.mse(y_raw, y_pred)
