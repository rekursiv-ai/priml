from typing import NamedTuple

import types

from _typeshed import Incomplete
from jax._src import (
    compiler as compiler,
    traceback_util as traceback_util,
)
from jax._src.lib import xla_client as xla_client
from jax._src.lib.mlir import ir as ir

LOCATION_PATTERN: Incomplete
FRAME_PATTERN: Incomplete
MLIR_ERR_PREFIX: str

class RawFrame(NamedTuple):
    func_name: Incomplete
    filename: Incomplete
    lineno: Incomplete
    colno: Incomplete

class MosaicError(Exception): ...

class VerificationError(MosaicError):
    def __init__(self, message: str) -> None: ...

def mlir_error_to_verification_error(base_err: ir.MLIRError) -> VerificationError: ...
def redact_locations(err_msg: str) -> str: ...
def parse_location_string(location_string: str) -> tuple[str, list[RawFrame]]: ...
def traceback_from_raw_frames(frames: list[RawFrame]) -> types.TracebackType: ...
