from _typeshed import Incomplete
from jax._src.dtypes import (
    iinfo as iinfo,
    issubdtype as issubdtype,
)
from jax._src.lib import xla_client as xc
from jax._src.named_sharding import AUTO as AutoSharding
from jax._src.sharding import Sharding as Sharding
from jax._src.util import tuple_insert as tuple_insert

type Shape = tuple[int, ...]

class AutoLayout: ...

class Layout:
    major_to_minor: tuple[int, ...]
    tiling: tuple[tuple[int, ...], ...] | None
    sub_byte_element_size_in_bits: int
    AUTO: Incomplete
    def __init__(
        self,
        major_to_minor: tuple[int, ...],
        tiling: tuple[tuple[int, ...], ...] | None = None,
        sub_byte_element_size_in_bits: int = 0,
    ) -> None: ...
    @staticmethod
    def from_pjrt_layout(pjrt_layout: xc.PjRtLayout): ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def update(self, **kwargs): ...
    def check_compatible_aval(self, aval_shape: Shape): ...

type LayoutOptions = Layout | AutoLayout | None
type ShardingOptions = Sharding | AutoSharding | None

class Format:
    layout: Incomplete
    sharding: Incomplete
    def __init__(
        self,
        layout: LayoutOptions = None,
        sharding: ShardingOptions = None,
    ) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...

def get_layout_for_vmap(dim: int, layout: Layout) -> Layout: ...
