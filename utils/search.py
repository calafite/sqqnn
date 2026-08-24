import itertools
from collections.abc import Sequence
from typing import Any, cast

from tracking.records import ConfigValue, Hyperparameters


class GridSearch:
    @staticmethod
    def generate(**kwargs: Sequence[ConfigValue] | ConfigValue) -> list[Hyperparameters]:
        keys: list[str] = list(kwargs.keys())

        values: list[Sequence[ConfigValue]] = [
            list(value) if isinstance(value, Sequence) and not isinstance(value, str) else [value]
            for value in kwargs.values()
        ]

        combinations = itertools.product(*values)
        combos = cast(
            "list[dict[str, Any]]",
            [dict(zip(keys, combo)) for combo in combinations],
        )
        return [Hyperparameters(**combo) for combo in combos]
