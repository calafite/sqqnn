import numpy as np

from features import FeatureMap
from metrics import ModelMetrics
from network import ClassNet, RegNet
from scaling import DataScaler
from solvers import GdSolver, LlsSolver


class ClassificationPipeline:
    def __init__(
        self,
        feature_map: FeatureMap | None = None,
        solver: LlsSolver | None = None,
        net: ClassNet | None = None,
    ) -> None:
        self.feature_map: FeatureMap = feature_map or FeatureMap(degree=1)
        self.solver: LlsSolver = solver or LlsSolver()
        self.net: ClassNet = net or ClassNet()

    def with_config(
        self,
        feature_map: FeatureMap | None = None,
        solver: LlsSolver | None = None,
    ) -> "ClassificationPipeline":
        new_map: FeatureMap = feature_map or self.feature_map
        new_solver: LlsSolver = solver or self.solver
        return ClassificationPipeline(new_map, new_solver, self.net)

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> "ClassificationPipeline":
        x_mat: np.ndarray = self.feature_map.build_matrix(x_raw)
        weights: np.ndarray = self.solver.fit(x_mat, y_raw)
        new_net: ClassNet = self.net.with_weights(weights)

        return ClassificationPipeline(self.feature_map, self.solver, new_net)

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_mat: np.ndarray = self.feature_map.build_matrix(x_raw)
        return self.net.predict(x_mat)

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, float]:
        y_pred: np.ndarray = self.predict(x_raw)
        return ModelMetrics.compute_stats(y_raw, y_pred)


class RegressionPipeline:
    def __init__(
        self,
        feature_map: FeatureMap | None = None,
        y_scale: DataScaler | None = None,
        solver: GdSolver | None = None,
        net: RegNet | None = None,
    ) -> None:
        self.feature_map: FeatureMap = feature_map or FeatureMap(degree=1)
        self.y_scale: DataScaler = y_scale or DataScaler()
        self.solver: GdSolver = solver or GdSolver()
        self.net: RegNet = net or RegNet()

    def with_config(
        self,
        feature_map: FeatureMap | None = None,
        solver: GdSolver | None = None,
    ) -> "RegressionPipeline":
        new_map: FeatureMap = feature_map or self.feature_map
        new_solver: GdSolver = solver or self.solver
        return RegressionPipeline(new_map, self.y_scale, new_solver, self.net)

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> "RegressionPipeline":
        y_col: np.ndarray = y_raw.reshape(-1, 1)
        new_y_scale: DataScaler = self.y_scale.fit_bounds(y_col)
        y_norm: np.ndarray = new_y_scale.scale_data(y_col).flatten()

        x_mat: np.ndarray = self.feature_map.build_matrix(x_raw)
        new_net: RegNet = self.solver.fit(self.net, x_mat, y_norm)

        return RegressionPipeline(
            self.feature_map, new_y_scale, self.solver, new_net
        )

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_mat: np.ndarray = self.feature_map.build_matrix(x_raw)
        y_norm: np.ndarray = self.net.predict(x_mat)

        y_col: np.ndarray = y_norm.reshape(-1, 1)
        return self.y_scale.invert_data(y_col).flatten()

    def evaluate(self, x_raw: np.ndarray, y_raw: np.ndarray) -> float:
        y_pred: np.ndarray = self.predict(x_raw)
        return ModelMetrics.mse(y_raw, y_pred)
