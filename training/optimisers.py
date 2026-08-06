from typing import Any, Protocol

import numpy as np


class Optimiser(Protocol):
    def update(
        self, parameters: dict[str, Any], gradients: dict[str, Any]
    ) -> dict[str, Any]: ...


class GradientDescentOptimiser:
    def __init__(self, learning_rate: float = 0.1, l2_penalty: float = 0.0) -> None:
        self.learning_rate: float = learning_rate
        self.l2_penalty: float = l2_penalty

    def update(
        self, parameters: dict[str, Any], gradients: dict[str, Any]
    ) -> dict[str, Any]:
        updated_parameters: dict[str, Any] = {}

        for key, parameter in parameters.items():
            gradient: Any = gradients[key]

            if self.l2_penalty > 0.0:
                gradient = gradient + self.l2_penalty * parameter

            step_update: Any = self.learning_rate * gradient
            updated_parameters[key] = parameter - step_update

        return updated_parameters


class AdamOptimiser:
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_penalty: float = 0.0,
    ) -> None:
        self.learning_rate: float = learning_rate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.l2_penalty: float = l2_penalty
        self.m_state: dict[str, Any] = {}
        self.v_state: dict[str, Any] = {}
        self.time_step: int = 0

    def update(
        self, parameters: dict[str, Any], gradients: dict[str, Any]
    ) -> dict[str, Any]:
        self.time_step += 1
        updated_parameters: dict[str, Any] = {}

        for key, parameter in parameters.items():
            gradient: Any = gradients[key]

            if self.l2_penalty > 0.0:
                gradient = gradient + self.l2_penalty * parameter

            if key not in self.m_state:
                self.m_state[key] = np.zeros_like(parameter)
                self.v_state[key] = np.zeros_like(parameter)

            updated_state: Any = (
                self.beta1 * self.m_state[key] + (1.0 - self.beta1) * gradient
            )
            self.m_state[key] = updated_state

            updated_state = self.beta2 * self.v_state[key] + (1.0 - self.beta2) * (
                gradient**2
            )
            self.v_state[key] = updated_state

            m_line: Any = self.m_state[key] / (1.0 - self.beta1**self.time_step)
            v_line: Any = self.v_state[key] / (1.0 - self.beta2**self.time_step)

            step_update: Any = (
                self.learning_rate * m_line / (np.sqrt(v_line) + self.epsilon)
            )
            updated_parameters[key] = parameter - step_update

        return updated_parameters
