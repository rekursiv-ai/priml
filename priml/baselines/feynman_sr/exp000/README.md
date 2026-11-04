# Feynman Symbolic Regression Baseline (exp000)

**Task**: Symbolic regression on SRSD Feynman 120 equations
**Method**: Dimensional Genetic Algorithm + Levenberg-Marquardt constant optimization
**Target**: Beat PySR 60% exact recovery on SRSD Feynman benchmark

## Baseline Configuration (exp000)

| Parameter | Value |
|-----------|-------|
| Population | 200 |
| Generations | 50 |
| Crossover prob | 0.6 |
| Mutation prob | 0.3 |
| Tournament size | 3 |
| Hall of Fame | 5 |
| Max expression nodes | 30 |
| Data samples/equation | 1000 |
| Seed | 42 (bit-for-bit reproducible) |
| Equations | 10 (subset for fast iteration) |

## Quick Start

```bash
# Install dependencies
pip install deap scipy sympy numpy configgle

# Run baseline
python -m priml.baselines.feynman_sr.exp000.experiments exp000
```

## Experiment Chain

```
exp000 (baseline)          → 70% on 10-equation subset
  └── exp001 (+generations) → 100 generations
  └── exp002 (+population)  → 500 population
  └── exp003 (+dataset)     → Full Easy (30 equations)
  └── exp004 (+MCTS)        → Hypothesis pinning
  └── exp005 (+Neural)      → Neural-guided generation
```

## Expected Results

| Experiment | Split | Target Recovery |
|------------|-------|-----------------|
| exp000 | subset10 | ~70% |
| exp001 | subset10 | ~80% |
| exp002 | subset10 | ~75% |
| exp003 | easy (30) | ~60% (beat PySR) |

## Architecture

- **data.py**: SRSD Feynman dataset loader (120 equations, dimensional metadata)
- **model.py**: Dimensional grammar + GA + LM constant optimization
- **train_step.py**: Evaluation metrics (NMSE, exact recovery via sympy)
- **experiments.py**: configgle configs (one change per fork)

## Dimensional Grammar

Operators respect physical dimensions:
- `ADD/SUB`: Same dimensions only
- `MUL`: Adds dimensions
- `DIV`: Subtracts dimensions
- `POW`: Base dimension × exponent (exponent must be dimensionless)
- `SIN/COS/EXP/LOG`: Dimensionless only
- `SQRT`: Halves dimensions (even exponents only)

## Exact Recovery Criteria

Expression matches true formula if:
1. Symbolic structure identical (via sympy)
2. Constants within 1% relative tolerance

## Reproducibility

All experiments use fixed seed (42) for bit-for-bit reproducibility.
Golden test values stored in `testdata/` for CI verification.
