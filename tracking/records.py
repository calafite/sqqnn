import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Hyperparameters:
    degree: int = 1
    basis: str = "standard"
    optimiser: str = "adam"
    learning_rate: float = 0.1
    l2_penalty: float = 0.0
    epochs: int = 100
    solver: str = "lls"
    scale_x: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "basis": self.basis,
            "optimiser": self.optimiser,
            "learning_rate": self.learning_rate,
            "l2_penalty": self.l2_penalty,
            "epochs": self.epochs,
            "solver": self.solver,
            "scale_x": self.scale_x,
        }


@dataclass(frozen=True)
class RunMetrics:
    primary_score: float
    additional_stats: Mapping[str, float] = field(default_factory=dict)
    history: Mapping[str, Sequence[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    config: Hyperparameters
    metrics: RunMetrics
    model: Any
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
