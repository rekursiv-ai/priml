from collections.abc import (
    Iterable,
    Set as AbstractSet,
)
from typing import Any

from _typeshed import Incomplete

class Effect: ...

type Effects = AbstractSet[Effect]

class JaxprInputEffect(Effect):
    input_index: Incomplete
    def __init__(self, input_index: Any) -> None: ...
    def replace(self, *, input_index: Any | None = None): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class EffectTypeSet:
    def __init__(self) -> None: ...
    def add_type(self, effect_type: type[Effect]): ...
    def contains(self, eff: Effect) -> bool: ...
    def filter_in(self, effects: Iterable[Effect]) -> list[Effect]: ...
    def filter_not_in(self, effects: Iterable[Effect]) -> list[Effect]: ...

no_effects: Effects
ordered_effects: EffectTypeSet
shardable_ordered_effects: EffectTypeSet
lowerable_effects: EffectTypeSet
control_flow_allowed_effects: EffectTypeSet
custom_derivatives_allowed_effects: EffectTypeSet
remat_allowed_effects: EffectTypeSet
partial_eval_kept_effects: EffectTypeSet
