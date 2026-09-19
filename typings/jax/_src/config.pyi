from collections.abc import (
    Callable as Callable,
    Iterator,
    Sequence,
)
from typing import Any, Generic, NoReturn, Protocol

import contextlib
import enum
import types

from _typeshed import Incomplete
from jax._src import (
    deprecations as deprecations,
    logging_config as logging_config,
)
from jax._src.lib import (
    guard_lib as guard_lib,
    jax_jit as jax_jit,
    xla_client as xla_client,
)

config_ext: Incomplete
logger: Incomplete

class EffortLevel(enum.Enum):
    UNKNOWN = 0
    O0 = 9
    O1 = 19
    O2 = 29
    O3 = 39

def bool_env(varname: str, default: bool) -> bool: ...
def int_env(varname: str, default: int) -> int: ...

class ValueHolder(Protocol[_T]):
    value: _T

class Config:
    meta: Incomplete
    use_absl: bool
    def __init__(self) -> None: ...
    def update(self, name, val) -> None: ...
    def read(self, name): ...
    @property
    def values(self): ...
    def add_option(self, name, holder, opt_type, meta_args, meta_kwargs) -> None: ...
    absl_flags: Incomplete
    def config_with_absl(self): ...
    def complete_absl_config(self, absl_flags) -> None: ...
    def parse_flags_with_absl(self): ...

register_trace_context_callback: Incomplete
trace_context: Incomplete
config: Incomplete
update: Incomplete
parse_flags_with_absl: Incomplete

class NoDefault: ...

no_default: Incomplete
config_states: Incomplete

class State(config_ext.Config[_T]):
    __doc__: Incomplete
    def __init__(
        self,
        name: str,
        default: _T,
        help,
        update_global_hook: Callable[[_T], None] | None = None,
        update_thread_local_hook: Callable[[_T | None], None] | None = None,
        parser: Callable[[Any], Any] | None = None,
        extra_description: str = "",
        default_context_manager_value: Any = ...,
        include_in_jit_key: bool = False,
        include_in_trace_context: bool = False,
    ) -> None: ...
    def __bool__(self) -> NoReturn: ...
    def __call__(self, new_val: Any = ...): ...

class StateContextManager(contextlib.ContextDecorator):
    state: Incomplete
    new_val: Incomplete
    def __init__(self, state, new_val) -> None: ...
    prev: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

UPGRADE_BOOL_HELP: str
UPGRADE_BOOL_EXTRA_DESC: str

def bool_state(
    name: str,
    default: bool,
    help: str,
    *,
    update_global_hook: Callable[[bool], None] | None = None,
    update_thread_local_hook: Callable[[bool | None], None] | None = None,
    upgrade: bool = False,
    extra_description: str = "",
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    validator: Callable[[str], None] | None = None,
) -> State[bool]: ...
def enum_state(
    name: str,
    enum_values: Sequence[str],
    default: str,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    extra_validator: Callable[[str], None] | None = None,
) -> State[str]: ...
def optional_enum_state(
    name: str,
    enum_values: Sequence[str],
    default: str | None,
    help: str,
    *,
    update_global_hook: Callable[[str | None], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
) -> State[str | None]: ...
def enum_class_state(
    name: str,
    enum_class: type[_ET],
    default: _ET,
    help: str,
    *,
    update_global_hook: Callable[[_ET], None] | None = None,
    update_thread_local_hook: Callable[[_ET | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    extra_validator: Callable[[_ET], None] | None = None,
) -> State[_ET]: ...
def int_state(
    name: str,
    default: int,
    help: str,
    *,
    update_global_hook: Callable[[int], None] | None = None,
    update_thread_local_hook: Callable[[int | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    validator: Callable[[Any], None] | None = None,
) -> State[int]: ...
def float_state(
    name: str,
    default: float,
    help: str,
    *,
    update_global_hook: Callable[[float], None] | None = None,
    update_thread_local_hook: Callable[[float | None], None] | None = None,
) -> State[float]: ...
def string_state(
    name: str,
    default: str,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
) -> State[str]: ...
def optional_string_state(
    name: str,
    default: str | None,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_trace_context: bool = False,
) -> State[str | None]: ...
def string_or_object_state(
    name: str,
    default: Any,
    help: str,
    *,
    update_global_hook: Callable[[Any], None] | None = None,
    update_thread_local_hook: Callable[[Any], None] | None = None,
    validator: Callable[[Any], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
) -> State[Any]: ...

class Flag(Generic[_T]):
    value: _T
    def __init__(
        self,
        name: str,
        default: _T,
        update_hook: Callable[[Any], None] | None = None,
    ) -> None: ...
    def __bool__(self) -> NoReturn: ...

def bool_flag(name, default, *args, **kwargs) -> Flag[bool]: ...
def int_flag(name, default, *args, **kwargs) -> Flag[int]: ...
def float_flag(name, default, *args, **kwargs) -> Flag[float]: ...
def string_flag(name, default, *args, **kwargs) -> Flag[str]: ...
def enum_flag(name, default, *args, **kwargs) -> Flag[str]: ...

already_configured_with_absl: bool
trace_state: Incomplete
axis_env_state: Incomplete
mesh_context_manager: Incomplete
abstract_mesh_context_manager: Incomplete
device_context: Incomplete
compute_on_context_manager: Incomplete
xla_metadata_context_manager: Incomplete
pallas_tpu_interpret_mode_context_manager: Incomplete

class UserConfig:
    def __init__(self, default_value) -> None: ...
    @property
    def value(self): ...
    def __call__(self, new_value): ...

class UserContext:
    def __init__(self, config, new_value) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...

def make_user_context(default_value=None): ...

jax2tf_associative_scan_reductions: Incomplete
jax2tf_default_native_serialization: Incomplete
jax_serialization_version: Incomplete
jax_export_calling_convention_version: Incomplete
export_ignore_forward_compatibility: Incomplete
jax_platforms: Incomplete
jax_pjrt_client_create_options: Incomplete
enable_checks: Incomplete
debug_key_reuse: Incomplete
check_tracer_leaks: Incomplete
checking_leaks: Incomplete
check_static_indices: Incomplete
captured_constants_warn_bytes: Incomplete
captured_constants_report_frames: Incomplete
debug_nans: Incomplete
debug_infs: Incomplete
log_compiles: Incomplete
explain_cache_misses: Incomplete
log_checkpoint_residuals: Incomplete
pmap_shmap_merge: Incomplete
custom_vjp3: Incomplete
distributed_debug: Incomplete
random_seed_offset: Incomplete

class LegacyPrngKeyState(enum.StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    ERROR = "error"

legacy_prng_key: Incomplete
enable_custom_prng: Incomplete
default_prng_impl: Incomplete
threefry_partitionable: Incomplete
threefry_gpu_kernel_lowering: Incomplete
use_direct_linearize: Incomplete
use_simplified_jaxpr_constants: Incomplete
remove_size_one_mesh_axis_from_type: Incomplete
softmax_custom_jvp: Incomplete
enable_custom_vjp_by_custom_transpose: Incomplete
raise_persistent_cache_errors: Incomplete
persistent_cache_min_compile_time_secs: Incomplete
persistent_cache_min_entry_size_bytes: Incomplete
persistent_cache_enable_xla_caches: Incomplete
compilation_cache_include_metadata_in_key: Incomplete
hlo_source_file_canonicalization_regex: Incomplete
include_full_tracebacks_in_locations: Incomplete
traceback_in_locations_limit: Incomplete
share_binary_between_hosts: Incomplete
share_binary_between_hosts_timeout_ms: Incomplete
enable_pgle: Incomplete
pgle_profiling_runs: Incomplete
pgle_aggregation_percentile: Incomplete
enable_compilation_cache: Incomplete
compilation_cache_dir: Incomplete
compilation_cache_check_contents: Incomplete
compilation_cache_expect_pgle: Incomplete
compilation_cache_max_size: Incomplete
remove_custom_partitioning_ptr_from_cache_key: Incomplete
default_dtype_bits: Incomplete

class ExplicitX64Mode(enum.IntEnum):
    WARN = ...
    ERROR = ...
    ALLOW = ...

explicit_x64_dtypes: Incomplete

class NumpyDtypePromotion(enum.StrEnum):
    STANDARD = "standard"
    STRICT = "strict"

numpy_dtype_promotion: Incomplete
disallow_mesh_context_manager: Incomplete
error_checking_behavior_nan: Incomplete
error_checking_behavior_divide: Incomplete
error_checking_behavior_oob: Incomplete
enable_x64: Incomplete
default_device: Incomplete
disable_jit: Incomplete
numpy_rank_promotion: Incomplete
default_matmul_precision: Incomplete
allow_f16_reductions: Incomplete
traceback_filtering: Incomplete
bcoo_cusparse_lowering: Incomplete
eager_constant_folding: Incomplete
enable_remat_opt_pass: Incomplete
no_tracing: Incomplete
no_execution: Incomplete
disable_vmap_shmap_error: Incomplete
mutable_array_checks: Incomplete
refs_to_pins: Incomplete
disable_bwd_checks: Incomplete
xla_runtime_errors: Incomplete
jax_xla_profile_version: Incomplete

@contextlib.contextmanager
def explicit_device_put_scope() -> Iterator[None]: ...
@contextlib.contextmanager
def explicit_device_get_scope() -> Iterator[None]: ...

transfer_guard_host_to_device: Incomplete
transfer_guard_device_to_device: Incomplete
transfer_guard_device_to_host: Incomplete

@contextlib.contextmanager
def transfer_guard(new_val: str) -> Iterator[None]: ...

array_garbage_collection_guard: Incomplete
thread_guard: Incomplete

class RuntimeTracebackMode(enum.StrEnum):
    OFF = "off"
    ON = "on"
    FULL = "full"
    def as_cpp_enum(self): ...

send_traceback_to_runtime: Incomplete
use_shardy_partitioner: Incomplete
gpu_use_magma: Incomplete
exec_time_optimization_effort: Incomplete
memory_fitting_effort: Incomplete
optimization_level: Incomplete
memory_fitting_level: Incomplete
DEFAULT_CPU_COLLECTIVES_IMPL: str
cpu_collectives_implementation: Incomplete
use_high_dynamic_range_gumbel: Incomplete
jax_dump_ir_to: Incomplete
jax_include_debug_info_in_dumps: Incomplete
jax_dump_ir_modes: Incomplete
jax_ragged_dot_use_ragged_dot_instruction: Incomplete
jax_pallas_verbose_errors: Incomplete
