from collections.abc import Callable as Callable

import chex

def get_variable(type_var: str): ...

class RandomTest(chex.TestCase):
    def test_tree_split_key_like(self) -> None: ...
    def test_tree_random_like(
        self,
        sampler: Callable[[chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array],
        dtype: str,
        type_var: str,
    ): ...
