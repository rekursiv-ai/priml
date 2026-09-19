from jax._src import (
    api_util as api_util,
    core as jax_core,
    tree_util as tree_util,
)
from jax._src.pallas.fuser import (
    fusible_dtype as fusible_dtype,
    fusion as fusion_lib,
)
from jax._src.pallas.fuser.fusible import fusible_p as fusible_p
from jax._src.traceback_util import api_boundary as api_boundary

def fuse(f=None, *, resolve_fusion_dtypes: bool = True, debug: bool = False): ...
def construct_fusion(
    candidate_values,
    jaxpr: jax_core.Jaxpr,
    outvars,
    *invars,
    **kwargs,
) -> fusion_lib.Fusion: ...
def fuse_jaxpr(jaxpr: jax_core.Jaxpr, out_tree: tree_util.PyTreeDef, consts, *args): ...
