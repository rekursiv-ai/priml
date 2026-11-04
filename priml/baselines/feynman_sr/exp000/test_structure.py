"""
Structure test for exp000 baseline - verifies imports and config structure.
Run this for quick CI verification without full GA execution.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

print("Testing imports...")
from priml.baselines.feynman_sr.exp000.data import (
    generate_data_for_equation,
    load_feynman_10,
)

print("✓ data.py imports OK")

from priml.baselines.feynman_sr.exp000.model import (
    DimVec,
    FeynmanSRModel,
    GAConfig,
    Grammar,
)

print("✓ model.py imports OK")

from priml.baselines.feynman_sr.exp000.train_step import check_exact_recovery

print("✓ train_step.py imports OK")

from priml.baselines.feynman_sr.exp000.experiments import (
    ALL_EXPERIMENTS,
    exp000,
    exp001,
    exp002,
    exp003,
)

print("✓ experiments.py imports OK")

print("\nTesting data loading...")
dataset = load_feynman_10()
print(f"✓ Loaded {len(dataset)} equations")
for eq in dataset:
    print(f"  - {eq.eq_id}: {eq.formula} (target_dim={eq.target_dim})")

print("\nTesting data generation...")
eq = dataset[0]  # feynman-i.12.1
variables, target = generate_data_for_equation(eq, n_samples=10, seed=42)
print(f"✓ Generated data for {eq.eq_id}")
print(f"  Variables: {list(variables.keys())}")
print(f"  Target shape: {target.shape}")

print("\nTesting grammar...")
var_dims = {name: DimVec.parse(dim) for name, dim in eq.variables.items()}
target_dim = DimVec.parse(eq.target_dim)
grammar = Grammar(var_dims, target_dim, max_depth=6)
print(f"✓ Grammar built for target_dim={target_dim}")

exprs = grammar.generate_dimension(target_dim, max_depth=6, n_samples=5, max_nodes=20)
print(f"✓ Generated {len(exprs)} sample expressions:")
for e in exprs[:3]:
    print(f"  - {e} (nodes={e.count_nodes()}, dim={e.dim})")

print("\nTesting model...")
model = FeynmanSRModel(
    ga_config=GAConfig(population_size=10, max_generations=2, seed=42)
)
print("✓ Model created with GAConfig")

print("\nTesting experiment configs...")
cfg000 = exp000()
print(
    f"✓ exp000: split={cfg000.dataset_split}, pop={cfg000.step.model.ga_config.population_size}"
)

cfg001 = exp001()
print(f"✓ exp001: generations={cfg001.step.model.ga_config.max_generations}")

cfg002 = exp002()
print(f"✓ exp002: population={cfg002.step.model.ga_config.population_size}")

cfg003 = exp003()
print(f"✓ exp003: split={cfg003.dataset_split}")

print(f"\n✓ All {len(ALL_EXPERIMENTS)} experiment configs registered")

print("\nTesting exact recovery check...")
assert check_exact_recovery("x0 * x1", "x0*x1") is True
assert check_exact_recovery("x0 * x1", "x0 * x2") is False
print("✓ Exact recovery check works")

print("\n" + "=" * 50)
print("ALL STRUCTURE TESTS PASSED!")
print("=" * 50)
print("\nThe exp000 baseline structure is correctly set up.")
print("For full GA run, use: python -m priml.baselines.feynman_sr.exp000.experiments")
