from typing import Any, Protocol

import abc

import jax

class LossFn(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        params: Any,
        inputs: jax.Array,
        targets: jax.Array,
    ) -> jax.Array: ...
