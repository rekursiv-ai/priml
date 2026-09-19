from jax._src import (
    config as config,
    core as core,
    source_info_util as source_info_util,
    sourcemap as sourcemap,
)
from jax.experimental.source_mapper import common as common

def compile_jaxpr(work_dir, f, f_args, f_kwargs, **_): ...
def canonicalize_filename(file_name: str): ...
def make_jaxpr_dump(jaxpr: core.Jaxpr, **_) -> common.SourceMapDump: ...
