from typing import TypeVar

import chex

T = TypeVar("T")

def tree_cast_like(tree: T, other_tree: chex.ArrayTree) -> T: ...
def tree_cast(
    tree: chex.ArrayTree,
    dtype: chex.ArrayDType | None,
) -> chex.ArrayTree: ...
def tree_dtype(
    tree: chex.ArrayTree,
    mixed_dtype_handler: str | None = None,
) -> chex.ArrayDType: ...
