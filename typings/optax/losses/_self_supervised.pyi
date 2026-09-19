import chex

def ntxent(
    embeddings: chex.Array,
    labels: chex.Array,
    temperature: chex.Numeric = 0.07,
) -> chex.Numeric: ...
def triplet_margin_loss(
    anchors: chex.Array,
    positives: chex.Array,
    negatives: chex.Array,
    axis: int = -1,
    norm_degree: chex.Numeric = 2,
    margin: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-06,
) -> chex.Array: ...
