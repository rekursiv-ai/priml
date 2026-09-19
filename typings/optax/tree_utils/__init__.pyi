from _typeshed import Incomplete
from optax.tree_utils._casting import (
    tree_cast as tree_cast,
    tree_cast_like as tree_cast_like,
    tree_dtype as tree_dtype,
)
from optax.tree_utils._random import (
    tree_random_like as tree_random_like,
    tree_split_key_like as tree_split_key_like,
    tree_unwrap_random_key_data as tree_unwrap_random_key_data,
)
from optax.tree_utils._state_utils import (
    NamedTupleKey as NamedTupleKey,
    tree_get as tree_get,
    tree_get_all_with_path as tree_get_all_with_path,
    tree_map_params as tree_map_params,
    tree_set as tree_set,
)
from optax.tree_utils._tree_math import (
    tree_add as tree_add,
    tree_add_scale as tree_add_scale,
    tree_allclose as tree_allclose,
    tree_batch_shape as tree_batch_shape,
    tree_bias_correction as tree_bias_correction,
    tree_clip as tree_clip,
    tree_conj as tree_conj,
    tree_div as tree_div,
    tree_full_like as tree_full_like,
    tree_max as tree_max,
    tree_min as tree_min,
    tree_mul as tree_mul,
    tree_norm as tree_norm,
    tree_ones_like as tree_ones_like,
    tree_real as tree_real,
    tree_scale as tree_scale,
    tree_size as tree_size,
    tree_sub as tree_sub,
    tree_sum as tree_sum,
    tree_update_infinity_moment as tree_update_infinity_moment,
    tree_update_moment as tree_update_moment,
    tree_update_moment_per_elem_norm as tree_update_moment_per_elem_norm,
    tree_vdot as tree_vdot,
    tree_where as tree_where,
    tree_zeros_like as tree_zeros_like,
)

tree_scalar_mul = tree_scale
tree_add_scalar_mul = tree_add_scale
tree_l1_norm: Incomplete
tree_l2_norm = tree_norm
tree_linf_norm: Incomplete
