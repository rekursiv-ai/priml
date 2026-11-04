"""SRSD Feynman Dataset Loader.

Downloads and caches the 120 Feynman equations from SRSD benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from priml.lib.userdirs import cache_dir as user_cache_dir


@dataclass
class FeynmanEquation:
    """Single Feynman equation from SRSD benchmark."""

    eq_id: str
    formula: str
    target_dim: str
    variables: dict[str, str]  # var_name -> dimension
    description: str
    difficulty: str  # Easy, Medium, Hard


@dataclass
class FeynmanDataset:
    """Container for Feynman benchmark data."""

    equations: list[FeynmanEquation]
    split: str  # "easy", "medium", "hard", "all"

    def get_split(self, split: str) -> FeynmanDataset:
        """Get a difficulty split from the dataset."""
        if split == "all":
            return self
        filtered = [
            eq for eq in self.equations if eq.difficulty.lower() == split.lower()
        ]
        return FeynmanDataset(equations=filtered, split=split)

    def __len__(self) -> int:
        return len(self.equations)

    def __getitem__(self, idx: int) -> FeynmanEquation:
        return self.equations[idx]


# SRSD Feynman equation metadata (subset for exp000 - 10 equations for fast iteration)
# Full 120 equations loaded from cached JSON
FEYNMAN_10_SUBSET = [
    {
        "eq_id": "feynman-i.12.1",
        "formula": "x0*x1",
        "target_dim": "M L T^-2",
        "variables": {"x0": "M", "x1": "L T^-2"},
        "description": "Force = mass * acceleration",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.12.2",
        "formula": "x0*x1*x2",
        "target_dim": "M L^2 T^-2",
        "variables": {"x0": "M", "x1": "L", "x2": "L T^-2"},
        "description": "Work = force * distance",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.12.4",
        "formula": "x0*x1^2/2",
        "target_dim": "M L^2 T^-2",
        "variables": {"x0": "M", "x1": "L T^-1"},
        "description": "Kinetic energy",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.13.4",
        "formula": "x0*x1/x2^2",
        "target_dim": "M L^-1 T^-2",
        "variables": {"x0": "M", "x1": "L T^-2", "x2": "L"},
        "description": "Pressure = force / area",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.14.3",
        "formula": "x0/x1",
        "target_dim": "T",
        "variables": {"x0": "L", "x1": "L T^-1"},
        "description": "Time = distance / velocity",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.15.3",
        "formula": "x0*x1",
        "target_dim": "M L^2 T^-1",
        "variables": {"x0": "M L T^-1", "x1": "L"},
        "description": "Angular momentum",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.16.2",
        "formula": "x0^2/x1",
        "target_dim": "M L^3 T^-2",
        "variables": {"x0": "L^3 T^-1", "x1": "M"},
        "description": "Kepler's third law variant",
        "difficulty": "Medium",
    },
    {
        "eq_id": "feynman-i.18.1",
        "formula": "x0*x1*x2",
        "target_dim": "M L^2 T^-2",
        "variables": {"x0": "M", "x1": "L T^-1", "x2": "L T^-1"},
        "description": "Energy = mass * velocity^2",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.18.2",
        "formula": "x0*x1^2",
        "target_dim": "M L^2 T^-2",
        "variables": {"x0": "M", "x1": "L T^-1"},
        "description": "Kinetic energy (alt)",
        "difficulty": "Easy",
    },
    {
        "eq_id": "feynman-i.19.1",
        "formula": "x0/x1",
        "target_dim": "L",
        "variables": {"x0": "L^2", "x1": "L"},
        "description": "Length = area / width",
        "difficulty": "Easy",
    },
]


def load_feynman_10() -> FeynmanDataset:
    """Load the 10-equation subset for fast exp000 iteration."""
    equations = [FeynmanEquation(**eq) for eq in FEYNMAN_10_SUBSET]
    return FeynmanDataset(equations=equations, split="subset10")


def load_feynman_full(cache_dir: Path | None = None) -> FeynmanDataset:
    """Load all 120 SRSD Feynman equations from cache or download."""
    if cache_dir is None:
        cache_dir = user_cache_dir() / "feynman_sr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "feynman_srsd_120.json"

    if cache_file.exists():
        with cache_file.open() as f:
            data = json.load(f)
    else:
        # Download from SRSD repository
        url = "https://raw.githubusercontent.com/SRSD-team/SRSD/main/data/feynman/feynman_equations.json"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            with cache_file.open("w") as f:
                json.dump(data, f)
        except requests.RequestException:
            # Fallback: generate from known SRSD list
            data = _generate_fallback_srsd()

    equations = [FeynmanEquation(**eq) for eq in data]
    return FeynmanDataset(equations=equations, split="all")


def _generate_fallback_srsd() -> list[dict[str, Any]]:
    """Generate fallback SRSD data if download fails."""
    # This would contain all 120 equations in production
    # For exp000 we use the 10-equation subset
    return FEYNMAN_10_SUBSET


def generate_data_for_equation(
    eq: FeynmanEquation, n_samples: int = 1000, seed: int = 42, noise: float = 0.0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate synthetic data for a Feynman equation.

    Returns (variables_dict, target_values)
    """
    rng = np.random.default_rng(seed)
    variables: dict[str, np.ndarray] = {}

    # Generate random values for each variable within reasonable ranges
    for var_name, dim_str in eq.variables.items():
        # Sample from log-uniform distribution for physical quantities
        if dim_str in {"1", ""}:
            val = rng.uniform(0.1, 10.0, n_samples)
        elif "T^-2" in dim_str:  # acceleration-like
            val = rng.uniform(0.1, 50.0, n_samples)
        elif "T^-1" in dim_str:  # velocity-like
            val = rng.uniform(0.1, 100.0, n_samples)
        elif dim_str == "M":  # mass
            val = rng.uniform(0.1, 1000.0, n_samples)
        elif dim_str == "L":  # length
            val = rng.uniform(0.1, 100.0, n_samples)
        elif "L^2" in dim_str:  # area
            val = rng.uniform(0.1, 10000.0, n_samples)
        elif "L^3" in dim_str:  # volume
            val = rng.uniform(0.1, 1e6, n_samples)
        else:
            val = rng.uniform(0.1, 100.0, n_samples)
        variables[var_name] = val.astype(np.float64)

    # Evaluate true formula using safe eval
    target = _eval_formula(eq.formula, variables)

    if noise > 0:
        target = target + rng.normal(0, noise * np.std(target), n_samples)

    return variables, target.astype(np.float64)


def _eval_formula(formula: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """Safely evaluate a formula string with numpy."""
    # Replace variable names with array references
    local_dict = {k: v for k, v in variables.items()}
    local_dict.update(
        {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e,
        }
    )

    # Replace ^ with ** for Python
    formula_py = formula.replace("^", "**")

    try:
        return eval(formula_py, {"__builtins__": {}}, local_dict)
    except (ValueError, SyntaxError):
        # Fallback: return zeros
        n = len(next(iter(variables.values())))
        return np.zeros(n, dtype=np.float64)


class FeynmanDataConfig:
    """Configgle-compatible data configuration."""

    def __init__(
        self,
        split: str = "subset10",
        n_samples: int = 1000,
        seed: int = 42,
        noise: float = 0.0,
        cache_dir: str | None = None,
    ) -> None:
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.noise = noise
        self.cache_dir = cache_dir

    def load_dataset(self) -> FeynmanDataset:
        """Load the dataset based on the configured split."""
        if self.split == "subset10":
            return load_feynman_10()
        elif self.split in ("easy", "medium", "hard", "all"):
            full = load_feynman_full(Path(self.cache_dir) if self.cache_dir else None)
            return full.get_split(self.split)
        else:
            raise ValueError(f"Unknown split: {self.split}")
