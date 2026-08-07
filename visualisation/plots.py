import os
from typing import Any, Sequence

os.environ["GSK_RENDERER"] = "gl"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.container import BarContainer

from tracking.records import RunRecord


class Visualiser:
    def __init__(self) -> None:
        sns.set_theme(
            style="white",
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "--",
                "axes.titlesize": 16,
                "axes.labelsize": 12,
                "axes.titleweight": "bold",
                "axes.titlepad": 18,
                "figure.figsize": (9, 6),
            },
        )

    @staticmethod
    def plot_convergence(
        record: RunRecord, metric_key: str = "loss", title: str = "Model Convergence"
    ) -> None:
        history: Sequence[float] = record.metrics.history.get(metric_key, [])
        if not history:
            print(f"No history found for metric: {metric_key}")
            return

        x_epochs: list[int] = list(range(1, len(history) + 1))

        if len(history) > 1000:
            step: int = max(1, len(history) // 1000)
            x_epochs = x_epochs[::step]
            history = history[::step]

        fig, ax = plt.subplots()

        sns.lineplot(x=x_epochs, y=history, color="#2563EB", linewidth=2.5, ax=ax)
        ax.fill_between(x_epochs, history, color="#2563EB", alpha=0.1)

        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric_key.capitalize())

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

        fig, ax = plt.subplots()

        sns.barplot(
            x=x_values,
            y=y_values,
            hue=x_values,
            palette="mako",
            legend=False,
            ax=ax,
        )

        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(
                    container,
                    fmt="%.4f",
                    padding=3,
                    fontsize=10,
                    color="#333333",
                )

        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.05)

        ax.set_title(title)
        ax.set_xlabel(param_name.replace("_", " ").title())
        ax.set_ylabel("Primary Score")

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

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        mesh_points: np.ndarray = np.c_[xx.ravel(), yy.ravel()]

        predictions: np.ndarray = model.predict(mesh_points)
        zz: np.ndarray = np.where(predictions >= 0.0, 1.0, -1.0).reshape(xx.shape)

        fig, ax = plt.subplots()

        ax.contourf(xx, yy, zz, alpha=0.4, cmap="RdBu_r")

        sns.scatterplot(
            x=x_data[:, 0],
            y=x_data[:, 1],
            hue=y_data,
            palette="RdBu_r",
            edgecolor="white",
            linewidth=1.2,
            s=70,
            legend=False,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        plt.tight_layout()
        plt.show()
