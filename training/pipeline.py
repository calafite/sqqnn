import numpy as np

from data.features import FeatureMap
from utils.metrics import ModelMetrics
from core.network import ClassNet, RegNet
from data.scaling import DataScaler
from training.solvers import GdSolver, LlsSolver


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

    def with_config(
        self,
        feature_map: FeatureMap | None = None,
        solver: LlsSolver | None = None,
        scale_x: bool | None = None,
    ) -> "ClassificationPipeline":
        new_map: FeatureMap = feature_map or self.feature_map
        new_solver: LlsSolver = solver or self.solver
        use_x_scale: bool = self.scale_x if scale_x is None else scale_x
        return ClassificationPipeline(
            new_map, self.x_scale, new_solver, self.net, use_x_scale
        )

    def _prepare_x(
        self, x_raw: np.ndarray, fit_mode: bool = False
    ) -> tuple[np.ndarray, DataScaler]:
        if not self.scale_x:
            return x_raw, self.x_scale

        current_scale: DataScaler = (
            self.x_scale.fit_bounds(x_raw) if fit_mode else self.x_scale
        )
        norm_x: np.ndarray = current_scale.scale_data(x_raw)
        return norm_x, current_scale

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> "ClassificationPipeline":
        x_proc, new_scale = self._prepare_x(x_raw, fit_mode=True)
        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)

        weights: np.ndarray = self.solver.fit(x_mat, y_raw)
        new_net: ClassNet = self.net.with_weights(weights)

        return ClassificationPipeline(
            self.feature_map, new_scale, self.solver, new_net, self.scale_x
        )

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
        solver: GdSolver | None = None,
        net: RegNet | None = None,
        scale_x: bool = False,
    ) -> None:
        self.feature_map: FeatureMap = feature_map or FeatureMap(degree=1)
        self.x_scale: DataScaler = x_scale or DataScaler()
        self.y_scale: DataScaler = y_scale or DataScaler()
        self.solver: GdSolver = solver or GdSolver()
        self.net: RegNet = net or RegNet()
        self.scale_x: bool = scale_x

    def with_config(
        self,
        feature_map: FeatureMap | None = None,
        solver: GdSolver | None = None,
        scale_x: bool | None = None,
    ) -> "RegressionPipeline":
        new_map: FeatureMap = feature_map or self.feature_map
        new_solver: GdSolver = solver or self.solver
        use_x_scale: bool = self.scale_x if scale_x is None else scale_x
        return RegressionPipeline(
            new_map, self.x_scale, self.y_scale, new_solver, self.net, use_x_scale
        )

    def _prepare_x(
        self, x_raw: np.ndarray, fit_mode: bool = False
    ) -> tuple[np.ndarray, DataScaler]:
        if not self.scale_x:
            return x_raw, self.x_scale

        current_scale: DataScaler = (
            self.x_scale.fit_bounds(x_raw) if fit_mode else self.x_scale
        )
        norm_x: np.ndarray = current_scale.scale_data(x_raw)
        return norm_x, current_scale

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> "RegressionPipeline":
        x_proc, new_x_scale = self._prepare_x(x_raw, fit_mode=True)

        y_col: np.ndarray = y_raw.reshape(-1, 1)
        new_y_scale: DataScaler = self.y_scale.fit_bounds(y_col)
        y_norm: np.ndarray = new_y_scale.scale_data(y_col).flatten()

        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)
        new_net: RegNet = self.solver.fit(self.net, x_mat, y_norm)

        return RegressionPipeline(
            self.feature_map,
            new_x_scale,
            new_y_scale,
            self.solver,
            new_net,
            self.scale_x,
        )

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_proc, _ = self._prepare_x(x_raw, fit_mode=False)
        x_mat: np.ndarray = self.feature_map.build_matrix(x_proc)

        y_norm: np.ndarray = self.net.predict(x_mat)
        y_col: np.ndarray = y_norm.reshape(-1, 1)
        return self.y_scale.invert_data(y_col).flatten()

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> float:
        y_pred: np.ndarray = self.predict(x_raw)
        return ModelMetrics.mse(y_raw, y_pred)
