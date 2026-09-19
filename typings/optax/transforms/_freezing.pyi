from optax._src import base as base
from optax.transforms._combining import partition as partition
from optax.transforms._masking import masked as masked

import chex

def freeze(mask: bool | chex.ArrayTree) -> base.GradientTransformation: ...
def selective_transform(
    optimizer: base.GradientTransformation,
    *,
    freeze_mask: bool | chex.ArrayTree,
) -> base.GradientTransformation: ...
