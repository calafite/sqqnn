import numpy as np


class SharedRNG:
    # homage to jay pee
    rng: np.random.Generator = np.random.default_rng(42)
