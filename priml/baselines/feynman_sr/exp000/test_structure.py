"""Structure test for exp000 baseline - verifies imports and config structure.

Run this for quick CI verification without full GA execution.
"""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[4]),
)

from priml.baselines.feynman_sr.exp000.data import (
    generate_data_for_equation,
    load_feynman_10,
)
from priml.baselines.feynman_sr.exp000.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.feynman_sr.exp000.model import (
    DimVec,
    FeynmanSRModel,
    GAConfig,
    Grammar,
)
from priml.baselines.feynman_sr.exp000.train_step import check_exact_recovery

dataset = load_feynman_10()
for _ in dataset:
    pass

eq = dataset[0]  # feynman-i.12.1
variables, target = generate_data_for_equation(eq, n_samples=10, seed=42)

var_dims = {name: DimVec.parse(dim) for name, dim in eq.variables.items()}
target_dim = DimVec.parse(eq.target_dim)
grammar = Grammar(var_dims, target_dim, max_depth=6)

exprs = grammar.generate_dimension(target_dim, max_nodes=20, n_samples=5)
for _e in exprs[:3]:
    pass

model = FeynmanSRModel(
    ga_config=GAConfig(population_size=10, max_generations=2, seed=42),
)

cfg000 = exp000()

cfg001 = exp001()

cfg002 = exp002()

cfg003 = exp003()


assert check_exact_recovery("x0 * x1", "x0*x1") is True
assert check_exact_recovery("x0 * x1", "x0 * x2") is False
