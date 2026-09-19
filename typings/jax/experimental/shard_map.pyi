from jax._src import traceback_util as traceback_util

@traceback_util.api_boundary
def shard_map(f, mesh, in_specs, out_specs, check_rep: bool = True): ...
