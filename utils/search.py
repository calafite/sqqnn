import itertools
from typing import Any, Sequence

from tracking.records import Hyperparameters

class GridSearch:
    @staticmethod
    def generate(**kwargs: Any) -> Sequence[Hyperparameters]:
        keys: list[str] = list(kwargs.keys())

        # ensures scalar values are wrapped in a list for the cartesian product
        values: list[Any] = [
            value if isinstance(value, (list, tuple)) else [value] for value in kwargs.values()
        ]

        combinations = itertools.product(*values)
        return [Hyperparameters(**dict(zip(keys, combo))) for combo in combinations]
