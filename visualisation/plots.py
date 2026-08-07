from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tracking.records import RunRecord


class Visualiser:
    def __init__(self) -> None:
        sns.set_theme(style="whitegrid", palette="muted")

    @staticmethod
    def plot_convergence(
        record: RunRecord, metric_key: str = "loss", title: str = "Model Convergence"
    ) -> None:
        history: Sequence[float] = record.metrics.history.get(metric_key, [])
        if not history:
            print(f"No history found for metric: {metric_key}")
            return

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=range(1, len(history) + 1), y=history, linewidth=2)

        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel(metric_key.capitalize(), fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_hyperparameter_comparison(
        records: Sequence[RunRecord], param_name: str, title: str = "Hyperparameter Search"
    ) -> None:
        if not records:
            return

        x_values: list[Any] = []
        y_values: list[float] = []

        for record in records:
            config_dict = record.config.to_dict()
            if param_name in config_dict:
                x_values.append(config_dict[param_name])
                y_values.append(record.metrics.primary_score)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=x_values, y=y_values, color="#4C72B0")

        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(param_name.replace("_", " ").title(), fontsize=12)
        plt.ylabel("Primary Score", fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_decision_boundary(
        model: Any, x_data: np.ndarray, y_data: np.ndarray, title: str = "Decision Boundary"
    ) -> None:
        x_min: float = float(x_data[:, 0].min()) - 0.5
        x_max: float = float(x_data[:, 0].max()) + 0.5
        y_min: float = float(x_data[:, 1].min()) - 0.5
        y_max: float = float(x_data[:, 1].max()) + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        mesh_points: np.ndarray = np.c_[xx.ravel(), yy.ravel()]

        predictions: np.ndarray = model.predict(mesh_points)
        zz: np.ndarray = np.where(predictions >= 0.0, 1.0, -1.0).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
        sns.scatterplot(
            x=x_data[:, 0],
            y=x_data[:, 1],
            hue=y_data,
            palette="coolwarm",
            edgecolor="k",
            legend=False
        )

        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.tight_layout()
        plt.show()
