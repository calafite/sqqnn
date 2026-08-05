import numpy as np
from typing import Optional

class DataScaler:
    def __init__(self, minimum: Optional[np.ndarray] = None, maximum: Optional[np.ndarray] = None) -> None:
        self.minimum_value = minimum
        self.maximum_value = maximum

    def fit_bounds(self, data: np.ndarray) -> "DataScaler":
       minimum_array: np.ndarray = np.min(data, axis=0)
       maximum_array: np.ndarray = np.max(data, axis=0)
       return DataScaler(minimum=minimum_array, maximum=maximum_array)

    def scale_data(self, data: np.ndarray) -> np.ndarray:
        if self.minimum_value is None or self.maximum_value is None:
            raise ValueError("bounds missing")

        value_range: np.ndarray = self.maximum_value - self.minimum_value
        # to prevent division on zero on constant features
        value_range[value_range == 0] = 1.0

        norm_data: np.ndarray = (data - self.minimum_value) / value_range
        # maps from unit to target interval
        return norm_data * 2.0 - 1.0

    def invert_data(self, data: np.ndarray) -> np.ndarray:
        if self.minimum_value is None or self.maximum_value is None:
            raise ValueError("bounds missing")

        value_range: np.ndarray = self.maximum_value - self.minimum_value

        # maps back from target interval to unit interval
        norm_data: np.ndarray = (data + 1.0) / 2.0
        return norm_data * value_range + self.minimum_value
