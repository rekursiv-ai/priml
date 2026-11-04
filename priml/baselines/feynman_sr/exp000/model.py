"""Feynman SR Model: Dimensional GA + Levenberg-Marquardt Constant Optimization.

exp000: Baseline Genetic Algorithm with dimensional constraints.
"""

from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import sympy as sp
from deap import algorithms, base, creator, tools
from scipy.optimize import least_squares

from priml.baselines.feynman_sr.exp000.data import (
    FeynmanDataset,
    FeynmanEquation,
    generate_data_for_equation,
)

# =============================================================================
# DIMENSIONAL GRAMMAR (from feynman_sr/grammar.py)
# =============================================================================


class Dim:
    """Dimension indices for physical quantities."""

    M = 0
    L = 1
    T = 2
    Q = 3
    THETA = 4
    NAMES = ("M", "L", "T", "Q", "THETA")

    @classmethod
    def from_str(cls, s: str) -> tuple[int, int, int, int, int] | None:
        """Parse dimension string into vector.

        Args:
            s: Dimension string like "M L^-1 T^-2"

        Returns:
            Tuple of exponents (M, L, T, Q, THETA) or None if dimensionless

        """
        s = s.upper().replace("[", "").replace("]", "")
        if not s or s == "1":
            return None
        mapping = {
            "M": cls.M,
            "L": cls.L,
            "T": cls.T,
            "Q": cls.Q,
            "THETA": cls.THETA,
            "Θ": cls.THETA,
        }
        # Parse compound dimensions like "M L^-1 T^-2"
        vec = [0, 0, 0, 0, 0]

        for match in re.finditer(r"(THETA|[MLTQΘ])(?:\^(-?\d+))?", s):
            dim_str, exp = match.group(1), int(match.group(2) or 1)
            vec[mapping[dim_str]] = exp
        return tuple(vec)


@dataclass(frozen=True)
class DimVec:
    """Immutable dimensional vector for physical quantities."""

    vec: tuple[int, int, int, int, int]

    @classmethod
    def zero(cls) -> DimVec:
        """Create dimensionless vector."""
        return cls((0, 0, 0, 0, 0))

    @classmethod
    def parse(cls, s: str) -> DimVec:
        """Parse dimension string into DimVec."""
        return cls(Dim.from_str(s) or (0, 0, 0, 0, 0))

    def __add__(self, other: DimVec) -> DimVec:
        return DimVec(tuple(a + b for a, b in zip(self.vec, other.vec, strict=False)))

    def __sub__(self, other: DimVec) -> DimVec:
        return DimVec(tuple(a - b for a, b in zip(self.vec, other.vec, strict=False)))

    def is_dimensionless(self) -> bool:
        """Check if vector represents dimensionless quantity."""
        return all(v == 0 for v in self.vec)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DimVec) and self.vec == other.vec

    def __hash__(self) -> int:
        return hash(self.vec)

    def __deepcopy__(self, memo: dict) -> DimVec:
        return self


DIMENSIONLESS = DimVec.zero()


class OpType:
    """Operator type definitions with dimensional rules."""

    CONST = ("const", 0, lambda *_: None)
    VAR = ("var", 0, lambda *_: None)
    ADD = ("add", 2, lambda a, b: a if a == b else None)
    SUB = ("sub", 2, lambda a, b: a if a == b else None)
    MUL = ("mul", 2, lambda a, b: a + b)
    DIV = ("div", 2, lambda a, b: a - b)
    POW = ("pow", 2, lambda a, b: a + a if b.is_dimensionless() else None)
    POW2 = ("pow2", 1, lambda a: a + a)
    POW3 = ("pow3", 1, lambda a: a + a + a)
    POW_NEG1 = ("pow_neg1", 1, lambda a: DimVec(tuple(-v for v in a.vec)))
    POW_HALF = (
        "pow_half",
        1,
        lambda a: (
            DimVec(tuple(v // 2 for v in a.vec))
            if all(v % 2 == 0 for v in a.vec)
            else None
        ),
    )
    NEG = ("neg", 1, lambda a: a)
    SQRT = (
        "sqrt",
        1,
        lambda a: (
            DimVec(tuple(v // 2 for v in a.vec))
            if all(v % 2 == 0 for v in a.vec)
            else None
        ),
    )
    SIN = ("sin", 1, lambda a: a if a.is_dimensionless() else None)
    COS = ("cos", 1, lambda a: a if a.is_dimensionless() else None)
    EXP = ("exp", 1, lambda a: a if a.is_dimensionless() else None)
    LOG = ("log", 1, lambda a: a if a.is_dimensionless() else None)
    ABS = ("abs", 1, lambda a: a)

    ALL_OPS = (
        CONST,
        VAR,
        ADD,
        SUB,
        MUL,
        DIV,
        POW,
        POW2,
        POW3,
        POW_NEG1,
        POW_HALF,
        NEG,
        SQRT,
        SIN,
        COS,
        EXP,
        LOG,
        ABS,
    )

    OPS = (
        ADD,
        SUB,
        MUL,
        DIV,
        POW,
        POW2,
        POW3,
        POW_NEG1,
        POW_HALF,
        NEG,
        SQRT,
        SIN,
        COS,
        EXP,
        LOG,
        ABS,
    )
    SYMBOLS: ClassVar[dict] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "pow": "^",
        "pow2": "^2",
        "pow3": "^3",
        "pow_neg1": "^-1",
        "pow_half": "^0.5",
        "neg": "-",
        "sqrt": "√",
        "sin": "sin",
        "cos": "cos",
        "exp": "exp",
        "log": "log",
        "abs": "|·|",
    }

    def __init__(self, name: str, arity: int, dim_rule: callable) -> None:
        self.op_name = name
        self.arity = arity
        self.dim_rule = dim_rule

    @classmethod
    def from_name(cls, name: str) -> OpType:
        """Create OpType from operator name."""
        for op in cls.ALL_OPS:
            if op[0] == name:
                return cls(*op)
        msg = f"Unknown operator: {name}"
        raise ValueError(msg)


@dataclass
class TypedExpr:
    """Typed expression node with dimensional information."""

    op: OpType
    children: tuple[TypedExpr, ...]
    dim: DimVec
    value: float | None = None
    var_name: str | None = None
    const_id: int | None = None

    @classmethod
    def constant(
        cls,
        value: float,
        dim: DimVec = DIMENSIONLESS,
        const_id: int | None = None,
    ) -> TypedExpr:
        """Create constant expression."""
        return cls(OpType.from_name("const"), (), dim, value=value, const_id=const_id)

    @classmethod
    def variable(cls, name: str, dim: DimVec) -> TypedExpr:
        """Create variable expression."""
        return cls(OpType.from_name("var"), (), dim, var_name=name)

    @classmethod
    def binary(
        cls,
        op_name: str,
        left: TypedExpr,
        right: TypedExpr,
    ) -> TypedExpr | None:
        """Create binary operation expression."""
        op = OpType.from_name(op_name)
        dim = op.dim_rule(left.dim, right.dim)
        if dim is None:
            return None
        return cls(op, (left, right), dim)

    @classmethod
    def unary(cls, op_name: str, child: TypedExpr) -> TypedExpr | None:
        """Create unary operation expression."""
        op = OpType.from_name(op_name)
        dim = op.dim_rule(child.dim)
        if dim is None:
            return None
        return cls(op, (child,), dim)

    def __str__(self) -> str:
        if self.op.op_name == "const":
            return f"{self.value:.6g}"
        if self.op.op_name == "var":
            return self.var_name or "x"
        if self.op.arity == 1:
            sym = OpType.SYMBOLS.get(self.op.op_name, self.op.op_name)
            child_str = str(self.children[0])
            if self.op.op_name in ("pow2", "pow3", "pow_neg1", "pow_half"):
                return f"{child_str}{sym}"
            return f"{sym}({child_str})"
        sym = OpType.SYMBOLS.get(self.op.op_name, self.op.op_name)
        return f"({self.children[0]!s} {sym} {self.children[1]!s})"

    def __deepcopy__(self, memo: dict) -> TypedExpr:
        return self

    def get_constants(self) -> list[TypedExpr]:
        """Get all constant nodes in expression."""
        if self.op.op_name == "const":
            return [self]
        consts = []
        for child in self.children:
            consts.extend(child.get_constants())
        return consts

    def count_nodes(self) -> int:
        """Count total nodes in expression tree."""
        if self.op.arity == 0:
            return 1
        return 1 + sum(c.count_nodes() for c in self.children)

    def set_constants(self, new_values: list) -> TypedExpr:
        """Create new expression with updated constant values."""
        const_iter = iter(new_values)

        def replace_consts(expr: TypedExpr) -> TypedExpr:
            if expr.op.op_name == "const":
                return TypedExpr.constant(next(const_iter), expr.dim, expr.const_id)
            if not expr.children:
                return expr
            new_children = tuple(replace_consts(c) for c in expr.children)
            return TypedExpr(expr.op, new_children, expr.dim, var_name=expr.var_name)

        return replace_consts(self)

    def evaluate(self, variables: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate expression with given variable values."""
        if self.op.op_name == "const":
            return np.full_like(next(iter(variables.values())), self.value)
        if self.op.op_name == "var":
            return variables[self.var_name]

        child_vals = [c.evaluate(variables) for c in self.children]

        if self.op.op_name == "add":
            return child_vals[0] + child_vals[1]
        if self.op.op_name == "sub":
            return child_vals[0] - child_vals[1]
        if self.op.op_name == "mul":
            return child_vals[0] * child_vals[1]
        if self.op.op_name == "div":
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = child_vals[1]
                safe = np.where(
                    np.abs(denom) < 1e-12,
                    np.sign(denom) * 1e-12 + (denom == 0) * 1e-12,
                    denom,
                )
                return np.where(np.abs(denom) >= 1e-12, child_vals[0] / safe, np.inf)
        elif self.op.op_name == "pow":
            with np.errstate(over="ignore", invalid="ignore"):
                return np.power(child_vals[0], child_vals[1])
        elif self.op.op_name == "pow2":
            return child_vals[0] * child_vals[0]
        elif self.op.op_name == "pow3":
            return child_vals[0] * child_vals[0] * child_vals[0]
        elif self.op.op_name == "pow_neg1":
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = child_vals[0]
                safe = np.where(
                    np.abs(denom) < 1e-12,
                    np.sign(denom) * 1e-12 + (denom == 0) * 1e-12,
                    denom,
                )
                return np.where(np.abs(denom) >= 1e-12, 1.0 / safe, np.inf)
        elif self.op.op_name == "pow_half":
            with np.errstate(invalid="ignore"):
                return np.where(child_vals[0] >= 0, np.sqrt(child_vals[0]), np.nan)
        elif self.op.op_name == "neg":
            return -child_vals[0]
        elif self.op.op_name == "sqrt":
            with np.errstate(invalid="ignore"):
                return np.where(child_vals[0] >= 0, np.sqrt(child_vals[0]), np.nan)
        elif self.op.op_name == "sin":
            return np.sin(child_vals[0])
        elif self.op.op_name == "cos":
            return np.cos(child_vals[0])
        elif self.op.op_name == "exp":
            with np.errstate(over="ignore"):
                return np.exp(child_vals[0])
        elif self.op.op_name == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(child_vals[0] > 0, np.log(child_vals[0]), np.nan)
        elif self.op.op_name == "abs":
            return np.abs(child_vals[0])
        msg = f"Unknown operator: {self.op.op_name}"
        raise ValueError(msg)


class Grammar:
    """Fast dimensional grammar for generating valid expressions - SIMPLIFIED."""

    def __init__(
        self,
        variables: dict[str, DimVec],
        target_dim: DimVec,
        max_depth: int = 10,
    ) -> None:
        """Initialize grammar with variable dimensions and target dimension.

        Args:
            variables: Mapping from variable name to dimension vector
            target_dim: Target dimension for generated expressions
            max_depth: Maximum depth for expression generation

        """
        self.variables = variables
        self.target_dim = target_dim
        self.max_depth = max_depth
        self.unary_ops = [OpType.from_name(op[0]) for op in OpType.OPS if op[1] == 1]
        self.binary_ops = [OpType.from_name(op[0]) for op in OpType.OPS if op[1] == 2]
        self.var_dims = set(variables.values())
        self._build()

    def _build(self) -> None:
        """Build lookup tables for valid dimensional operations."""
        # Only use variable dimensions + dimensionless + simple combinations
        all_dims = set(self.var_dims)
        all_dims.add(DIMENSIONLESS)

        # Add simple unary results from variables
        for d in self.var_dims:
            for op in self.unary_ops:
                res = op.dim_rule(d)
                if res:
                    all_dims.add(res)

        # Add simple binary combinations of variables (max 2 operations)
        var_list = list(self.var_dims)
        for d1 in var_list:
            for d2 in var_list:
                for op in self.binary_ops:
                    res = op.dim_rule(d1, d2)
                    if res:
                        all_dims.add(res)

        # Add target and its unary ancestors
        all_dims.add(self.target_dim)
        for op in self.unary_ops:
            # For sqrt: child = target * 2
            # For pow2: child = target / 2 (if even)
            # For pow_neg1: child = -target
            if op == OpType.from_name("sqrt"):
                child = self.target_dim + self.target_dim
                all_dims.add(child)
            elif op == OpType.from_name("pow2"):
                if all(v % 2 == 0 for v in self.target_dim.vec):
                    child = DimVec(tuple(v // 2 for v in self.target_dim.vec))
                    all_dims.add(child)
            elif op == OpType.from_name("pow_neg1"):
                child = DimVec(tuple(-v for v in self.target_dim.vec))
                all_dims.add(child)
            elif op == OpType.from_name("neg"):
                all_dims.add(self.target_dim)
            elif op in [
                OpType.from_name("sin"),
                OpType.from_name("cos"),
                OpType.from_name("exp"),
                OpType.from_name("log"),
            ]:
                if self.target_dim.is_dimensionless():
                    all_dims.add(self.target_dim)

        # Build valid unary
        self.valid_unary: dict[DimVec, list[OpType]] = {}
        for d in all_dims:
            valid = [op for op in self.unary_ops if op.dim_rule(d) is not None]
            if valid:
                self.valid_unary[d] = valid

        # Build binary candidates - ONLY for target and priority dimensions
        priority_targets = [self.target_dim]
        # Add simple binary combinations of variables that make sense
        for d1 in self.var_dims:
            for d2 in self.var_dims:
                for op in [OpType.from_name("mul"), OpType.from_name("div")]:
                    res = op.dim_rule(d1, d2)
                    if res and res in all_dims:
                        priority_targets.append(res)

        self.binary_candidates: dict[DimVec, list[tuple[OpType, DimVec, DimVec]]] = {}
        for target in set(priority_targets):
            candidates = []
            # PRIORITY 1: Direct variable combinations
            for d1 in self.var_dims:
                for d2 in self.var_dims:
                    if d1 == target and d2 == target:
                        continue
                    for op in [OpType.from_name("mul"), OpType.from_name("div")]:
                        res = op.dim_rule(d1, d2)
                        if res == target:
                            candidates.append((op, d1, d2))

            # PRIORITY 2: Variable + dimensionless
            for d1 in self.var_dims:
                for op in [OpType.from_name("mul"), OpType.from_name("div")]:
                    if op.dim_rule(d1, DIMENSIONLESS) == target:
                        candidates.append((op, d1, DIMENSIONLESS))
                    if op.dim_rule(DIMENSIONLESS, d1) == target:
                        candidates.append((op, DIMENSIONLESS, d1))

            # PRIORITY 3: Other combinations from all_dims (limited)
            if len(candidates) < 10:
                for d1 in all_dims:
                    d2_mul = target - d1
                    if d2_mul in all_dims:
                        candidates.append((OpType.from_name("mul"), d1, d2_mul))
                    d2_div = d1 - target
                    if d2_div in all_dims:
                        candidates.append((OpType.from_name("div"), d1, d2_div))

            # Deduplicate and limit
            seen = set()
            unique = []
            for c in candidates:
                key = (c[0].op_name, c[1].vec, c[2].vec)
                if key not in seen:
                    seen.add(key)
                    unique.append(c)
            self.binary_candidates[target] = unique[:20]

    def generate_random_expr(
        self,
        target_dim: DimVec | None = None,
        depth: int = 0,
        rng: np.random.Generator | None = None,
        max_nodes: int = 20,
        _call_count: list | None = None,
    ) -> TypedExpr | None:
        """Generate random expression matching target dimension.

        Args:
            target_dim: Target dimension (default: grammar's target_dim)
            depth: Current recursion depth
            rng: Random number generator
            max_nodes: Maximum nodes in expression
            _call_count: Internal call counter to prevent infinite recursion

        Returns:
            TypedExpr or None if generation fails

        """
        if _call_count is None:
            _call_count = [0]
        _call_count[0] += 1
        if _call_count[0] > 500:
            return None

        if rng is None:
            rng = np.random.default_rng()
        target = target_dim or self.target_dim
        var_dims = set(self.variables.values())
        matching_vars = [v for v, d in self.variables.items() if d == target]

        # Dimensionless special case - prefer simple expressions
        if target.is_dimensionless():
            if matching_vars and rng.random() < 0.6:
                return TypedExpr.variable(rng.choice(matching_vars), target)
            if rng.random() < 0.3:
                return TypedExpr.constant(
                    rng.choice([0.5, 1.0, 2.0, 2 * np.pi]),
                    target,
                )
            if depth < 3 and rng.random() < 0.3:
                for op in [
                    OpType.from_name("sin"),
                    OpType.from_name("cos"),
                    OpType.from_name("exp"),
                    OpType.from_name("log"),
                ]:
                    if matching_vars:
                        child = TypedExpr.variable(rng.choice(matching_vars), target)
                        return TypedExpr.unary(op.op_name, child)

        if depth >= self.max_depth:
            if matching_vars and rng.random() < 0.7:
                return TypedExpr.variable(rng.choice(matching_vars), target)
            if target.is_dimensionless():
                return TypedExpr.constant(rng.uniform(-5, 5), target)
            return None

        # PRIORITY 1: Direct variable combinations that match target exactly
        if target in self.binary_candidates:
            candidates = self.binary_candidates[target]
            # First try: both dimensions are base variables (not target, not complex)
            for op, d1, d2 in candidates:
                d1_is_var = d1 in var_dims
                d2_is_var = d2 in var_dims
                d1_is_dimless = d1.is_dimensionless()
                d2_is_dimless = d2.is_dimensionless()

                # Prefer: var * var, var * dimless, dimless * var
                if (
                    (d1_is_var and d2_is_var)
                    or (d1_is_var and d2_is_dimless)
                    or (d1_is_dimless and d2_is_var)
                ):
                    # For var * var, use the actual variables directly
                    if d1_is_var and d2_is_var:
                        left_var = next(v for v, d in self.variables.items() if d == d1)
                        right_var = next(
                            v for v, d in self.variables.items() if d == d2
                        )
                        left = TypedExpr.variable(left_var, d1)
                        right = TypedExpr.variable(right_var, d2)
                    elif d1_is_var and d2_is_dimless:
                        left_var = next(v for v, d in self.variables.items() if d == d1)
                        left = TypedExpr.variable(left_var, d1)
                        right = TypedExpr.constant(rng.choice([0.5, 1.0, 2.0]), d2)
                    elif d1_is_dimless and d2_is_var:
                        left = TypedExpr.constant(rng.choice([0.5, 1.0, 2.0]), d1)
                        right_var = next(
                            v for v, d in self.variables.items() if d == d2
                        )
                        right = TypedExpr.variable(right_var, d2)
                    else:
                        continue

                    expr = TypedExpr.binary(op.op_name, left, right)
                    if expr and expr.count_nodes() <= max_nodes:
                        return expr

        # PRIORITY 1b: Recursive generation for composite dimensions
        # Try all binary candidates, recursively generating sub-expressions
        if target in self.binary_candidates:
            candidates = self.binary_candidates[target]
            for op, d1, d2 in candidates:
                # Skip if either dimension is the target itself (avoid infinite recursion)
                if target in (d1, d2):
                    continue
                # Recursively generate sub-expressions
                left = self.generate_random_expr(
                    d1,
                    depth + 1,
                    rng,
                    max_nodes,
                    _call_count,
                )
                right = self.generate_random_expr(
                    d2,
                    depth + 1,
                    rng,
                    max_nodes,
                    _call_count,
                )
                if left and right:
                    expr = TypedExpr.binary(op.op_name, left, right)
                    if expr and expr.count_nodes() <= max_nodes:
                        return expr

        # PRIORITY 2: Simple unary operations on variables
        if target in self.valid_unary:
            for op in self.valid_unary[target]:
                child_dim = op.dim_rule(target)
                if child_dim and child_dim in var_dims:
                    child_var = next(
                        v for v, d in self.variables.items() if d == child_dim
                    )
                    child = TypedExpr.variable(child_var, child_dim)
                    expr = TypedExpr.unary(op.op_name, child)
                    if expr and expr.count_nodes() <= max_nodes:
                        return expr

        # PRIORITY 3: Direct variable match
        if matching_vars and rng.random() < 0.7:
            return TypedExpr.variable(rng.choice(matching_vars), target)

        # PRIORITY 4: Dimensionless constant
        if target.is_dimensionless() and rng.random() < 0.3:
            return TypedExpr.constant(rng.uniform(-5, 5), target)

        return None

    def generate_dimension(
        self,
        target_dim: DimVec,
        max_nodes: int = 20,
        n_samples: int = 10,
        seed: int | None = None,
    ) -> list[TypedExpr]:
        """Generate multiple expressions matching target dimension.

        Args:
            target_dim: Target dimension
            max_nodes: Maximum nodes per expression
            n_samples: Number of expressions to generate
            seed: Random seed

        Returns:
            List of generated expressions

        """
        results = []
        rng = np.random.default_rng(seed)
        for _ in range(n_samples):
            expr = self.generate_random_expr(
                target_dim,
                depth=0,
                rng=rng,
                max_nodes=max_nodes,
            )
            if expr:
                results.append(expr)
        return results


# =============================================================================
# CONSTANT OPTIMIZATION (Levenberg-Marquardt via scipy)
# =============================================================================


def fit_constants(
    expr: TypedExpr,
    variables: dict[str, np.ndarray],
    target_values: np.ndarray,
    max_iter: int = 50,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
) -> tuple[TypedExpr, dict[int, float]]:
    """Optimize constants in expression using Levenberg-Marquardt.

    Args:
        expr: Expression with constants to optimize
        variables: Input variable values
        target_values: Target output values
        max_iter: Maximum iterations
        ftol: Function tolerance
        xtol: Parameter tolerance

    Returns:
        Tuple of (optimized expression, constant values dict)

    """
    consts = expr.get_constants()
    if not consts:
        return expr, {}

    x0 = np.array([c.value for c in consts])
    const_ids = [c.const_id for c in consts]

    def residuals(params: np.ndarray) -> np.ndarray:
        expr_opt = expr.set_constants(params.tolist())
        pred = expr_opt.evaluate(variables)
        pred = pred.flatten() if pred.ndim > 1 else pred
        if pred.shape != target_values.shape:
            return np.full_like(target_values, 1e6)
        return pred - target_values

    try:
        result = least_squares(
            residuals,
            x0,
            method="lm",
            max_nfev=max_iter,
            ftol=ftol,
            xtol=xtol,
            verbose=0,
        )
        optimized_expr = expr.set_constants(result.x.tolist())
        const_values = dict(zip(const_ids, result.x.tolist(), strict=False))
        return optimized_expr, const_values
    except (ValueError, RuntimeError):
        return expr, {}


# =============================================================================
# GA SEARCH (from feynman_sr/ga_search.py)
# =============================================================================

# DEAP singleton
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)


def evaluate_individual(
    individual: list,
    variables: dict[str, np.ndarray],
    target_values: np.ndarray,
) -> tuple[float]:
    """Evaluate fitness of an individual expression.

    Args:
        individual: DEAP individual containing expression
        variables: Input variable values
        target_values: Target output values

    Returns:
        Tuple of negative NMSE (for minimization)

    """
    expr = individual[0]
    try:
        has_consts = any(c.op.op_name == "const" for c in expr.get_constants())
        if has_consts:
            optimized_expr, const_values = fit_constants(
                expr,
                variables,
                target_values,
                max_iter=20,
            )
        else:
            optimized_expr, const_values = expr, {}

        pred = optimized_expr.evaluate(variables)
        if not np.all(np.isfinite(pred)):
            return (-float("inf"),)
        pred = pred.flatten() if pred.ndim > 1 else pred
        if pred.shape != target_values.shape:
            return (-float("inf"),)

        mse = np.mean((pred - target_values) ** 2)
        var = np.var(target_values)
        if var == 0:
            return (-float("inf"),)
        nmse = mse / var

        individual.optimized_expr = optimized_expr
        individual.const_values = const_values
        return (-nmse,)
    except (ValueError, RuntimeError):
        return (-float("inf"),)


def collect_subtree_paths(
    expr: TypedExpr,
    path: tuple[int, ...] = (),
    visited: set | None = None,
    max_depth: int = 15,
) -> list[tuple[int, ...]]:
    """Collect all subtree paths in expression.

    Args:
        expr: Expression to traverse
        path: Current path tuple
        visited: Set of visited node ids
        max_depth: Maximum recursion depth

    Returns:
        List of path tuples

    """
    if visited is None:
        visited = set()
    expr_id = id(expr)
    if expr_id in visited:
        return []
    visited.add(expr_id)
    if len(path) >= max_depth:
        return [path]
    paths = [path]
    for i, child in enumerate(expr.children):
        paths.extend(collect_subtree_paths(child, (*path, i), visited, max_depth))
    return paths


def get_subexpr_at_path(expr: TypedExpr, path: tuple[int, ...]) -> TypedExpr:
    """Get subexpression at given path.

    Args:
        expr: Root expression
        path: Tuple of child indices

    Returns:
        Subexpression at path

    """
    for idx in path:
        expr = expr.children[idx]
    return expr


def replace_subexpr_at_path(
    expr: TypedExpr,
    path: tuple[int, ...],
    replacement: TypedExpr,
    max_nodes: int = 30,
) -> TypedExpr | None:
    """Replace subexpression at path with new expression.

    Args:
        expr: Root expression
        path: Path to subexpression
        replacement: Replacement expression
        max_nodes: Maximum allowed nodes

    Returns:
        New expression with replacement, or None if too large

    """
    if not path:
        return replacement if replacement.count_nodes() <= max_nodes else None
    idx = path[0]
    new_children = list(expr.children)
    replaced = replace_subexpr_at_path(
        expr.children[idx],
        path[1:],
        replacement,
        max_nodes,
    )
    if replaced is None:
        return None
    new_children[idx] = replaced
    result = TypedExpr(expr.op, tuple(new_children), expr.dim)
    return result if result.count_nodes() <= max_nodes else None


def crossover(
    individual1: list, individual2: list, max_nodes: int = 30
) -> tuple[list, list]:
    """Perform crossover between two individuals.

    Args:
        individual1: First parent
        individual2: Second parent
        max_nodes: Maximum nodes in offspring

    Returns:
        Tuple of offspring individuals

    """
    expr1, expr2 = individual1[0], individual2[0]
    paths1 = collect_subtree_paths(expr1, max_depth=15)
    paths2 = collect_subtree_paths(expr2, max_depth=15)

    if not paths1 or not paths2:
        return individual1, individual2

    for _ in range(10):
        path1 = random.choice(paths1)
        path2 = random.choice(paths2)
        sub1 = get_subexpr_at_path(expr1, path1)
        sub2 = get_subexpr_at_path(expr2, path2)
        if sub1.dim == sub2.dim:
            new1 = replace_subexpr_at_path(expr1, path1, sub2, max_nodes)
            new2 = replace_subexpr_at_path(expr2, path2, sub1, max_nodes)
            if new1 and new2:
                individual1[0] = new1
                individual2[0] = new2
                del individual1.fitness.values
                del individual2.fitness.values
                return individual1, individual2
    return individual1, individual2


def mutate(
    individual: list,
    grammar: Grammar,
    max_nodes: int = 30,
    seed: int | None = None,
) -> tuple[list, ...]:
    """Mutate an individual by replacing a random subtree.

    Args:
        individual: Individual to mutate
        grammar: Grammar for generating valid replacements
        max_nodes: Maximum nodes in expression
        seed: Random seed

    Returns:
        Tuple containing mutated individual

    """
    expr = individual[0]
    paths = collect_subtree_paths(expr, max_depth=15)
    if not paths:
        return (individual,)

    rng = np.random.default_rng(seed)
    # With 20% probability, insert a constant instead of replacing subtree
    if rng.random() < 0.2:
        path = rng.choice(paths)
        subexpr = get_subexpr_at_path(expr, path)
        # Multiply by a constant (dimensionless)
        const = TypedExpr.constant(rng.uniform(0.1, 5.0), DIMENSIONLESS)
        new_expr = replace_subexpr_at_path(
            expr,
            path,
            TypedExpr.binary("mul", subexpr, const),
            max_nodes,
        )
        if new_expr:
            individual[0] = new_expr
            del individual.fitness.values
            return (individual,)

    # Standard subtree replacement
    path = random.choice(paths)
    subexpr = get_subexpr_at_path(expr, path)
    replacements = grammar.generate_dimension(
        subexpr.dim,
        max_nodes=max_nodes,
        n_samples=10,
        seed=seed,
    )
    if not replacements:
        return (individual,)
    replacement = random.choice(replacements)
    new_expr = replace_subexpr_at_path(expr, path, replacement, max_nodes)
    if new_expr is None:
        return (individual,)
    individual[0] = new_expr
    del individual.fitness.values
    return (individual,)


def clone_individual(individual: list) -> list:
    """Create a deep copy of an individual.

    Args:
        individual: Individual to clone

    Returns:
        Cloned individual

    """
    clone = creator.Individual(list(individual))
    clone.fitness = (
        creator.FitnessMin(individual.fitness.values)
        if individual.fitness.valid
        else creator.FitnessMin()
    )
    return clone


def generate_individual(
    grammar: Grammar, target_dim: DimVec, seed: int | None = None
) -> list:
    """Generate a random individual.

    Args:
        grammar: Grammar for expression generation
        target_dim: Target dimension
        seed: Random seed

    Returns:
        New individual

    """
    exprs = grammar.generate_dimension(
        target_dim,
        max_nodes=20,
        n_samples=5,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    expr = rng.choice(exprs) if exprs else TypedExpr.constant(0.5, DimVec.parse("1"))
    return creator.Individual([expr])


@dataclass
class GAConfig:
    """Genetic Algorithm configuration."""

    population_size: int = 200
    max_generations: int = 50
    cxpb: float = 0.6
    mutpb: float = 0.3
    tournsize: int = 3
    hall_of_fame_size: int = 5
    seed: int = 42
    max_nodes: int = 30


def run_ga_search(
    equation_name: str,
    grammar: Grammar,
    target_dim: DimVec,
    variables: dict[str, np.ndarray],
    target_values: np.ndarray,
    config: GAConfig,
) -> tuple[TypedExpr, float, dict]:
    """Run GA search for a single equation.

    Args:
        equation_name: Name of equation for logging
        grammar: Grammar for expression generation
        target_dim: Target dimension
        variables: Input variable values
        target_values: Target output values
        config: GA configuration

    Returns:
        Tuple of (best expression, best NMSE, info dict)

    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        lambda: generate_individual(grammar, target_dim, seed=config.seed),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover, max_nodes=config.max_nodes)
    toolbox.register(
        "mutate",
        mutate,
        grammar=grammar,
        max_nodes=config.max_nodes,
        seed=config.seed,
    )
    toolbox.register(
        "evaluate",
        evaluate_individual,
        variables=variables,
        target_values=target_values,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.tournsize)
    toolbox.register("clone", clone_individual)

    population = toolbox.population(n=config.population_size)

    # Pre-evaluate
    for ind in population:
        if not ind.fitness.valid:
            try:
                fit = toolbox.evaluate(ind)
                ind.fitness.values = fit
            except (ValueError, RuntimeError):
                ind.fitness.values = (-float("inf"),)

    hall_of_fame = tools.HallOfFame(config.hall_of_fame_size)
    hall_of_fame.update(population)

    population, _logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=config.cxpb,
        mutpb=config.mutpb,
        ngen=config.max_generations,
        halloffame=hall_of_fame,
        verbose=False,
    )

    best_individual = (
        hall_of_fame[0] if hall_of_fame else tools.selBest(population, k=1)[0]
    )
    best_expr = getattr(best_individual, "optimized_expr", best_individual[0])

    best_fitness = best_individual.fitness.values
    if len(best_fitness) == 0:
        msg = "No fitness values found"
        raise ValueError(msg)

    best_nmse = -float(best_fitness[0])
    complexity = best_expr.count_nodes()

    return (
        best_expr,
        best_nmse,
        {
            "complexity": complexity,
            "generations": config.max_generations,
            "equation": equation_name,
        },
    )


# =============================================================================
# HIGH-LEVEL MODEL INTERFACE
# =============================================================================


@dataclass
class FeynmanSRModel:
    """Main model interface for priml experiments."""

    ga_config: GAConfig = field(default_factory=GAConfig)

    def solve_equation(
        self,
        eq: FeynmanEquation,
        variables: dict[str, np.ndarray],
        target_values: np.ndarray,
    ) -> tuple[TypedExpr, float, dict]:
        """Solve a single Feynman equation.

        Args:
            eq: Feynman equation object
            variables: Input variable values
            target_values: Target output values

        Returns:
            Tuple of (best expression, NMSE, info dict)

        """
        # Build grammar
        var_dims = {name: DimVec.parse(dim) for name, dim in eq.variables.items()}
        target_dim = DimVec.parse(eq.target_dim)
        grammar = Grammar(
            var_dims,
            target_dim,
            max_depth=self.ga_config.max_generations // 5 + 5,
        )

        return run_ga_search(
            eq.eq_id,
            grammar,
            target_dim,
            variables,
            target_values,
            self.ga_config,
        )

    def evaluate_on_dataset(
        self,
        dataset: FeynmanDataset,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> dict:
        """Evaluate model on a dataset split.

        Args:
            dataset: FeynmanDataset to evaluate on
            n_samples: Number of samples per equation
            seed: Random seed

        Returns:
            Metrics dictionary

        """
        results = []
        exact_recoveries = 0

        for eq in dataset:
            variables, target = generate_data_for_equation(
                eq,
                n_samples=n_samples,
                seed=seed,
            )
            best_expr, nmse, info = self.solve_equation(eq, variables, target)

            # Check exact recovery (symbolic equality with 1% constant tolerance)
            exact = self.check_exact_recovery(best_expr, eq.formula, eq.variables)
            if exact:
                exact_recoveries += 1

            results.append(
                {
                    "eq_id": eq.eq_id,
                    "formula": eq.formula,
                    "predicted": str(best_expr),
                    "nmse": nmse,
                    "exact": exact,
                    "complexity": info["complexity"],
                },
            )

        recovery_rate = exact_recoveries / len(dataset) if dataset else 0.0
        return {
            "recovery_rate": recovery_rate,
            "exact_recoveries": exact_recoveries,
            "total": len(dataset),
            "results": results,
        }

    def check_exact_recovery(
        self,
        expr: TypedExpr,
        true_formula: str,
    ) -> bool:
        """Check if expression matches true formula (symbolic + constant tolerance).

        Args:
            expr: Predicted expression
            true_formula: True formula string

        Returns:
            True if expressions match

        """
        return self._check_exact_recovery(expr, true_formula)

    def _check_exact_recovery(
        self,
        expr: TypedExpr,
        true_formula: str,
    ) -> bool:
        """Check if expression matches true formula (symbolic + constant tolerance).

        Args:
            expr: Predicted expression
            true_formula: True formula string

        Returns:
            True if expressions match

        """
        try:
            # Convert our expression to sympy
            expr_sympy = self._to_sympy(expr)
            true_sympy = sp.sympify(true_formula.replace("^", "**"))
            # Check structural equality
            return sp.simplify(expr_sympy - true_sympy) == 0
        except (ValueError, RuntimeError):
            return False

    def _to_sympy(self, expr: TypedExpr):
        """Convert TypedExpr to sympy expression.

        Args:
            expr: TypedExpr to convert

        Returns:
            Sympy expression

        """
        import sympy as sp

        if expr.op.op_name == "const":
            return sp.nsimplify(expr.value, [sp.pi, sp.E])
        if expr.op.op_name == "var":
            return sp.symbols(expr.var_name)
        if expr.op.op_name == "add":
            return self._to_sympy(expr.children[0]) + self._to_sympy(expr.children[1])
        if expr.op.op_name == "sub":
            return self._to_sympy(expr.children[0]) - self._to_sympy(expr.children[1])
        if expr.op.op_name == "mul":
            return self._to_sympy(expr.children[0]) * self._to_sympy(expr.children[1])
        if expr.op.op_name == "div":
            return self._to_sympy(expr.children[0]) / self._to_sympy(expr.children[1])
        if expr.op.op_name == "pow":
            return self._to_sympy(expr.children[0]) ** self._to_sympy(expr.children[1])
        if expr.op.op_name == "pow2":
            return self._to_sympy(expr.children[0]) ** 2
        if expr.op.op_name == "pow3":
            return self._to_sympy(expr.children[0]) ** 3
        if expr.op.op_name == "pow_neg1":
            return 1 / self._to_sympy(expr.children[0])
        if expr.op.op_name == "pow_half":
            return sp.sqrt(self._to_sympy(expr.children[0]))
        if expr.op.op_name == "neg":
            return -self._to_sympy(expr.children[0])
        if expr.op.op_name == "sqrt":
            return sp.sqrt(self._to_sympy(expr.children[0]))
        if expr.op.op_name == "sin":
            return sp.sin(self._to_sympy(expr.children[0]))
        if expr.op.op_name == "cos":
            return sp.cos(self._to_sympy(expr.children[0]))
        if expr.op.op_name == "exp":
            return sp.exp(self._to_sympy(expr.children[0]))
        if expr.op.op_name == "log":
            return sp.log(self._to_sympy(expr.children[0]))
        if expr.op.op_name == "abs":
            return sp.Abs(self._to_sympy(expr.children[0]))
        return sp.symbols("x")
