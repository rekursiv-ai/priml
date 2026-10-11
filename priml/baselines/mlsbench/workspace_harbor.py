"""Reader and verifier for Harbor bundle pristine manifests (read at runtime, never copied).

A Harbor bundle (``<clone>/harbor/tasks-docker/mls-bench__<task>/``) carries
under ``tests/meta/``:

- ``config.json``: ``files[]`` with ``filename`` and ``edit`` ranges (``start``/``end``,
  1-indexed inclusive; ``-1`` means whole file / to EOF; empty means read-only),
  optional ``guard_exclude`` prefixes;
- ``pristine_manifest.json``: ``{relpath: sha256}`` of every file under a guarded prefix;
- ``pristine/<relpath>``: pristine bytes of each declared file (for the segment diff).

The verifier semantics implemented here (independently, from the documented behaviour):

1. *Fail closed*: a missing, malformed or empty manifest refuses to verify.
2. *Byte-segment diff*: the pristine file is cut into fixed segments between the editable
   ranges; each fixed segment must survive verbatim and in order in the edited file
   (first/last segment anchored at the file ends, intermediate ones at their rightmost
   feasible occurrence). Whatever lies between is the edit, of any length. Unlike upstream's
   ``read_text`` this works on raw bytes, so CRLF or non-UTF-8 changes are not normalised away.
3. *Manifest hash check* for undeclared files under guarded prefixes (modification and,
   when the whole tree is present, deletion). New files are not violations.
4. *Sanitised environment*: fixed ``PATH``, every interpreter-influencing variable dropped.

Extra strictness over upstream (documented deviation): symlinks at or above a guarded path
are violations rather than being followed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import difflib
import hashlib
import json
import re
import sys
import tomllib

from priml.baselines.mlsbench.scripts.prepare_data import UPSTREAM_DIR, default_path


SAFE_PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SCRUB: Final = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONIOENCODING",
    "PYTHONHASHSEED",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
)
_EXEMPT_TOP: Final = frozenset({"_task"})


class BundleError(RuntimeError):
    """The Harbor bundle is missing or malformed (the guard fails closed)."""


@dataclass(frozen=True)
class HarborViolation:
    """One breach found by :func:`guard`; ``kind`` is a stable slug."""

    path: str
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class HarborBundle:
    """Parsed ``tests/meta`` of one Harbor bundle."""

    root: Path
    manifest: dict[str, str]
    editable: dict[str, tuple[tuple[int, int], ...]]  # () = declared read-only
    guard_exclude: tuple[str, ...]
    pristine_root: Path

    @property
    def guarded_prefixes(self) -> frozenset[str]:
        """Top-level directories the guard protects."""
        names = list(self.editable) + list(self.manifest)
        return frozenset(PurePosixPath(n).parts[0] for n in names if n)


def sanitised_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for verifier/edit subprocesses: fixed PATH, interpreter vars dropped."""
    env = {k: v for k, v in (base or {}).items() if k not in _SCRUB}
    env["PATH"] = SAFE_PATH
    env["PYTHONNOUSERSITE"] = "1"
    return env


def bundle_dir(task: str, *, root: Path, flavor: str = "docker") -> Path:
    """Locate ``harbor/tasks-<flavor>/mls-bench__<task>`` under the upstream checkout."""
    base = Path(root) / "harbor" / f"tasks-{flavor}"
    path = base / f"mls-bench__{task}"
    if not (path / "tests" / "meta").is_dir():
        raise BundleError(f"{path}: no Harbor bundle (tests/meta missing)")
    return path


def _safe_rel(rel: object) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise BundleError(f"unsafe path {rel!r}")
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise BundleError(f"unsafe path {rel!r}")
    return p.as_posix()


def load_bundle(bundle: Path) -> HarborBundle:
    """Read ``tests/meta`` from a bundle directory; raise :class:`BundleError` if unusable."""
    meta = Path(bundle) / "tests" / "meta"
    try:
        raw = cast("object", json.loads((meta / "pristine_manifest.json").read_text()))
    except FileNotFoundError as e:
        raise BundleError("pristine_manifest.json missing: refusing to verify") from e
    except json.JSONDecodeError as e:
        raise BundleError("pristine_manifest.json malformed") from e
    if not isinstance(raw, dict) or not raw:
        raise BundleError("pristine_manifest.json empty: refusing to verify")
    entries = cast("dict[str, object]", raw)
    manifest: dict[str, str] = {}
    for k, v in entries.items():
        if not (isinstance(v, str) and len(v) == 64):
            raise BundleError("pristine_manifest.json: values must be sha256 hex")
        manifest[_safe_rel(k)] = v
    try:
        cfg = cast("dict[str, object]", json.loads((meta / "config.json").read_text()))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise BundleError(f"config.json unusable: {e}") from e
    editable: dict[str, tuple[tuple[int, int], ...]] = {}
    for f in cast("list[dict[str, object]]", cfg.get("files", [])):
        ranges = cast("list[dict[str, int]] | None", f.get("edit"))
        editable[_safe_rel(cast("str", f["filename"]))] = tuple(
            (int(r["start"]), int(r["end"])) for r in (ranges or [])
        )
    guard = cast("list[str] | None", cfg.get("guard_exclude", []))
    excl = tuple(_safe_rel(e) for e in guard or [])
    return HarborBundle(Path(bundle), manifest, editable, excl, meta / "pristine")


def find_bundle(task: str, *, root: Path) -> Path | None:
    """Return the Harbor bundle of ``task`` under the upstream clone ``root``, or None.

    ``None`` means the checkout has no bundle for it (a synthetic fixture, or a
    clone without ``harbor/``); it is not an error here, the caller decides.
    """
    try:
        return bundle_dir(task, root=root)
    except BundleError:
        return None


def pristine_source(hb: HarborBundle, rel: str) -> Path | None:
    """Locate the pristine bytes of a declared file inside the bundle.

    Order: ``tests/meta/pristine/<rel>``, then the image scaffold
    ``environment/_scaffold/<rel>`` (whole-file editable files that a task ships
    empty are only in the scaffold). The caller checks the manifest hash.
    """
    for base in (hb.pristine_root, hb.root / "environment" / "_scaffold"):
        p = base / rel
        if p.is_file() and not p.is_symlink():
            return p
    return None


def bundle_gpus(bundle: Path) -> int | None:
    """Read ``[environment] gpus`` from the bundle's ``task.toml``.

    Returns:
      gpus: The declared GPU count; ``0`` for a CPU-only task and, as in
        Harbor, when ``[environment]`` omits ``gpus``. ``None`` when there is no
        readable ``task.toml`` or ``[environment]`` table, or the value is not
        an integer.

    """
    try:
        doc = tomllib.loads((Path(bundle) / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = doc.get("environment")
    table = cast("dict[str, object]", env) if isinstance(env, dict) else None
    value = table.get("gpus", 0) if table is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def allow_internet(bundle: Path) -> bool | None:
    """Read ``[environment] allow_internet`` from the bundle's ``task.toml``.

    Returns:
      allow: The declared value, or ``None`` when absent or unreadable (callers
        must treat that as *no network*).

    """
    try:
        doc = tomllib.loads((Path(bundle) / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = doc.get("environment")
    table = cast("dict[str, object]", env) if isinstance(env, dict) else None
    value = table.get("allow_internet") if table is not None else None
    return value if isinstance(value, bool) else None


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_segments(
    pristine: bytes,
    current: bytes,
    ranges: tuple[tuple[int, int], ...],
) -> str | None:
    """Return None if ``current`` differs from ``pristine`` only inside ``ranges``, else why not.

    ``ranges`` are 1-indexed inclusive pristine lines; ``(-1, -1)`` is whole-file editable,
    an ``end`` of -1 means to EOF, and an empty tuple means read-only.
    """
    if pristine == current:
        return None
    if any(r == (-1, -1) for r in ranges):
        return None
    if not ranges:
        return "modified but file is declared read-only"
    lines = pristine.splitlines(keepends=True)
    n = len(lines)
    norm = sorted((max(s, 1), n if e == -1 else min(e, n)) for s, e in ranges)
    merged: list[tuple[int, int]] = []
    for s, e in norm:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    segs: list[bytes] = []
    cur = 0
    for s, e in merged:
        if s - 1 > cur:
            segs.append(b"".join(lines[cur : s - 1]))
        cur = max(cur, e)
    if cur < n:
        segs.append(b"".join(lines[cur:]))
    lead = merged[0][0] > 1
    trail = merged[-1][1] < n
    span: list[tuple[int, int] | None] = [None] * len(segs)
    if segs and lead:
        if not current.startswith(segs[0]):
            return "leading fixed segment changed"
        span[0] = (0, len(segs[0]))
    if segs and trail:
        if not current.endswith(segs[-1]):
            return "trailing fixed segment changed"
        span[-1] = (len(current) - len(segs[-1]), len(current))
    limit = len(current)
    for i in range(len(segs) - 1, -1, -1):
        got = span[i]
        if got is None:
            idx = current.rfind(segs[i], 0, limit)
            if idx < 0:
                return f"fixed segment #{i + 1} missing or out of order"
            got = span[i] = (idx, idx + len(segs[i]))
        limit = got[0]
    prev = 0
    for i, got in enumerate(span):
        assert got is not None
        if got[0] < prev:
            return f"fixed segment #{i + 1} overlaps an earlier one"
        prev = got[1]
    if len(merged) == 1:
        return None
    editable = {ln for s, e in merged for ln in range(s, e + 1)}
    sm = difflib.SequenceMatcher(
        None,
        lines,
        current.splitlines(keepends=True),
        autojunk=False,
    )
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in {"equal", "insert"}:
            continue
        bad = [ln for ln in range(i1 + 1, i2 + 1) if ln not in editable]
        if bad:
            return f"fixed line(s) {bad[:5]} changed"
    return None


def _excluded(rel: str, excl: tuple[str, ...]) -> bool:
    return any(rel == e or rel.startswith(e + "/") for e in excl)


def _symlinked(root: Path, rel: str) -> bool:
    """Report whether ``rel`` or any parent under ``root`` is a symlink."""
    p = root
    for part in PurePosixPath(rel).parts:
        p = p / part
        if p.is_symlink():
            return True
    return False


def guard(
    hb: HarborBundle,
    workspace: Path,
    *,
    check_deletions: bool = True,
) -> tuple[HarborViolation, ...]:
    """Verify ``workspace`` against a Harbor bundle; empty result means clean.

    ``check_deletions=False`` is for partial workspaces that only hold the declared files.
    """
    ws = Path(workspace)
    out: list[HarborViolation] = []
    guarded = hb.guarded_prefixes

    for rel in sorted(set(hb.manifest) | set(hb.editable)):
        if (
            _excluded(rel, hb.guard_exclude)
            or PurePosixPath(rel).parts[0] not in guarded
        ):
            continue
        if _symlinked(ws, rel):
            out.append(
                HarborViolation(
                    rel,
                    "symlink",
                    "guarded path is or sits under a symlink",
                ),
            )

    for rel, ranges in sorted(hb.editable.items()):
        p = ws / rel
        if _symlinked(ws, rel):
            continue
        pri = hb.pristine_root / rel
        if not p.is_file():
            out.append(HarborViolation(rel, "deleted", "declared file missing"))
            continue
        if not pri.is_file():
            if not any(r == (-1, -1) for r in ranges):
                out.append(
                    HarborViolation(rel, "pristine_missing", "no pristine snapshot"),
                )
            continue
        why = check_segments(pri.read_bytes(), p.read_bytes(), ranges)
        if why is not None:
            kind = "read_only_modified" if not ranges else "outside_range"
            out.append(HarborViolation(rel, kind, why))

    for rel, sha in sorted(hb.manifest.items()):
        if rel in hb.editable or _excluded(rel, hb.guard_exclude):
            continue
        if PurePosixPath(rel).parts[0] in _EXEMPT_TOP or _symlinked(ws, rel):
            continue
        p = ws / rel
        if not p.is_file():
            if check_deletions:
                out.append(HarborViolation(rel, "deleted", "manifest file missing"))
        elif sha256_file(p) != sha:
            out.append(
                HarborViolation(rel, "modified_unlisted", "hash differs from manifest"),
            )
    return tuple(out)


# --- Which eval-script set does the container run? -----------------------------------
#
# Two script sets exist upstream. ``tasks/<t>/scripts`` is what native MLS-Bench
# ran (task dir bind-mounted, no /tests mount) and so what produced every row of
# ``leaderboard.csv``. ``harbor/.../tests/eval/scripts`` is what Harbor's
# ``test.sh`` -> ``score_task.py run-evals --eval-root /tests/eval`` executes; it
# is the native set plus Harbor scaffolding. Our container has no /tests mount
# (no ``_inputgen/apply.py``, no holdout inputs) and no /workspace/_task links,
# so the Harbor-adapted scripts cannot run in it, while the native ones are the
# leaderboard's provenance. Decision: the container runs the NATIVE
# ``tasks/<t>/scripts``, and Harbor's copy is the tamper reference: an executed
# script that is not byte-identical to Harbor's is refused unless upstream
# itself ships that exact file as adapted (``ADAPTED_EVAL_SCRIPTS``, measured
# against the pinned commit; see ``real_clone_test``). Adapted tasks are still
# logged as ``harbor_adapted_scripts`` in workspace.json.
#
# 32 tasks differ on some file (29 on ``*.sh``; the 3 quant-* tasks differ only in
# ``_label_guard.py``). In 12 the difference is only the documented Harbor
# additions (``cd /workspace[/pkg]`` and the ``_inputgen/apply.py`` call, plus
# blank lines): for those the executed script must equal Harbor's after
# normalising exactly that. The other 20 have substantive Harbor patches
# (runtime races, ephemeral inputs, label guards): Harbor's copy cannot vouch for
# them, so the file *set* that differs is pinned AND the sha256 of every native
# ``tasks/<t>/scripts`` file is pinned in ``eval_script_pins.json`` (see
# ``EVAL_SCRIPT_PINS``); the executed native bytes must match the pin.
ADAPTED_EVAL_SCRIPTS: Final[dict[str, tuple[str, ...]]] = {
    "causal-discovery-discrete": (
        "eval_alarm.sh",
        "eval_cancer.sh",
        "eval_child.sh",
        "eval_hailfinder.sh",
        "eval_win95pts.sh",
    ),
    "causal-observational-linear-gaussian": (
        "eval_er10.sh",
        "eval_er10_hard.sh",
        "eval_er10_noisy.sh",
        "eval_er20.sh",
        "eval_er20_hard.sh",
        "eval_er20_noisy.sh",
        "eval_sf50.sh",
        "eval_sf50_hard.sh",
        "eval_sf50_noisy.sh",
    ),
    "causal-observational-linear-non-gaussian": (
        "eval_er10.sh",
        "eval_er15.sh",
        "eval_er20_dense.sh",
        "eval_er30.sh",
        "eval_er50.sh",
        "eval_er50_lowsample.sh",
        "eval_sf100.sh",
        "eval_sf12.sh",
    ),
    "causal-observational-nonlinear": (
        "eval_er12_gp.sh",
        "eval_er12_lowsample.sh",
        "eval_er15_sigmoid.sh",
        "eval_er20_gauss.sh",
        "eval_er20_mixed.sh",
        "eval_er8_mlp.sh",
        "eval_sf10_mixed.sh",
        "eval_sf20_gp.sh",
    ),
    "causal-treatment-effect": (
        "acic_synth.sh",
        "ihdp_synth.sh",
        "jobs_synth.sh",
    ),
    "cv-dbm-sampler": ("_runtime_patch.sh",),
    "cv-dbm-scheduler": (
        "_runtime_patch.sh",
        "run_DIODE.sh",
        "run_Imagenet.sh",
        "run_e2h.sh",
    ),
    "cv-diffusion-conditioning": (
        "_runtime_patch.sh",
        "train_large.sh",
        "train_medium.sh",
        "train_small.sh",
    ),
    "cv-vae-loss": (
        "_runtime_patch.sh",
        "train_large.sh",
        "train_medium.sh",
        "train_small.sh",
    ),
    "llm-rl-advantage": ("train.sh",),
    "llm-rl-importance-sampling": ("train.sh",),
    "llm-rl-kl-estimator": ("train.sh",),
    "llm-rl-reward-normalization": ("train.sh",),
    "meta-inner-loop-optimizer": (
        "_runtime_patch.sh",
        "cifar_fs_5shot.sh",
        "mini_imagenet_1shot.sh",
        "mini_imagenet_5shot.sh",
    ),
    "meta-rl": (
        "_runtime_patch.sh",
        "cheetah_vel.sh",
        "point_robot.sh",
        "sparse_point_robot.sh",
    ),
    "ml-anomaly-detection": (
        "cardio.sh",
        "satellite.sh",
        "shuttle.sh",
        "thyroid.sh",
    ),
    "ml-clustering-algorithm": (
        "blobs.sh",
        "digits.sh",
        "moons.sh",
        "varied_density.sh",
    ),
    "ml-ensemble-boosting": (
        "breast_cancer.sh",
        "california_housing.sh",
        "diabetes.sh",
    ),
    "ml-missing-data-imputation": (
        "breast_cancer.sh",
        "california.sh",
        "wine.sh",
    ),
    "ml-symbolic-regression": (
        "koza3.sh",
        "nguyen10.sh",
        "nguyen7.sh",
    ),
    "optimization-diagonal-net": (
        "d10000_k50_a1e0.sh",
        "d200_k20_a1e3.sh",
        "d500_k10_a1e3.sh",
        "d500_k10_a5e1.sh",
    ),
    "optimization-multi-objective": (
        "dtlz1.sh",
        "dtlz2.sh",
        "zdt1.sh",
        "zdt3.sh",
    ),
    "optimization-nas": (
        "cifar10.sh",
        "cifar100.sh",
        "imagenet16.sh",
    ),
    "optimization-parity": (
        "eval.sh",
        "eval_n16_k4.sh",
        "eval_n50_k8.sh",
        "eval_n64_k12.sh",
        "eval_n64_k8.sh",
    ),
    "optimization-variance-reduction": (
        "conditioned.sh",
        "logistic.sh",
        "mlp.sh",
    ),
    "quant-concept-drift": ("_label_guard.py",),
    "quant-graph-stock": ("_label_guard.py",),
    "quant-stock-prediction": ("_label_guard.py",),
    "rl-reward-learning": (
        "_runtime_patch.sh",
        "halfcheetah.sh",
        "hopper.sh",
        "walker2d.sh",
    ),
    "robo-diffusion-guidance": (
        "_runtime_patch.sh",
        "train_halfcheetah.sh",
        "train_hopper.sh",
        "train_walker2d.sh",
    ),
    "robo-diffusion-policy": (
        "_runtime_patch.sh",
        "train_halfcheetah.sh",
        "train_hopper.sh",
        "train_walker2d.sh",
    ),
    "robo-diffusion-sampling-method": (
        "_runtime_patch.sh",
        "train_halfcheetah.sh",
        "train_hopper.sh",
        "train_walker2d.sh",
    ),
}

NORMALISED_ONLY: Final = frozenset(
    {
        "causal-discovery-discrete",
        "causal-observational-linear-gaussian",
        "causal-observational-linear-non-gaussian",
        "causal-observational-nonlinear",
        "causal-treatment-effect",
        "ml-anomaly-detection",
        "ml-clustering-algorithm",
        "ml-ensemble-boosting",
        "ml-missing-data-imputation",
        "ml-symbolic-regression",
        "optimization-multi-objective",
        "optimization-variance-reduction",
    },
)
"""Adapted tasks whose Harbor scripts differ only by the documented additions."""

_HARBOR_ADDITIONS: Final = (
    re.compile(r"^cd /workspace(/[\w.\-]+)?\s*$"),
    re.compile(r'^python "/tests/eval/_inputgen/apply\.py" "[\w\-]+" /workspace\s*$'),
)


PINS_PATH: Final = Path(__file__).resolve().parent / "eval_script_pins.json"


def load_pins(path: Path = PINS_PATH) -> dict[str, dict[str, str]]:
    """Read ``{task: {relpath: sha256}}`` (digests only, no script contents)."""
    raw = cast("dict[str, dict[str, str]]", json.loads(path.read_text()))
    return {t: dict(files) for t, files in raw.items()}


EVAL_SCRIPT_PINS: Final[dict[str, dict[str, str]]] = load_pins()
"""Native ``tasks/<t>/scripts`` digests of the substantively adapted tasks."""


def pinned_tasks() -> frozenset[str]:
    """Tasks whose native scripts are byte-pinned: adapted and not normalised-only."""
    return frozenset(ADAPTED_EVAL_SCRIPTS) - NORMALISED_ONLY


def generate_pins(root: Path) -> dict[str, dict[str, str]]:
    """Hash every native script file of the pinned tasks under a clone ``root``.

    Regenerate after a deliberate upstream bump (review the diff first)::

        uv --quiet run --frozen python -m priml.baselines.mlsbench.workspace_harbor pin [--root DIR] [--out FILE]  # noqa: E501
    """
    out: dict[str, dict[str, str]] = {}
    for task in sorted(pinned_tasks()):
        d = Path(root) / "tasks" / task / "scripts"
        out[task] = {
            p.relative_to(d).as_posix(): sha256_file(p)
            for p in sorted(x for x in d.rglob("*") if x.is_file())
            if "__pycache__" not in p.parts
        }
    return out


def check_native_pins(
    task: str,
    scripts_dir: Path,
    pins: dict[str, dict[str, str]] | None = None,
) -> tuple[HarborViolation, ...]:
    """Verify a native scripts tree byte-for-byte against the pinned digests."""
    table = EVAL_SCRIPT_PINS if pins is None else pins
    want = table.get(task, {})
    if not want:
        return (HarborViolation("eval_script_pins", "eval_script_unpinned"),)
    base = Path(scripts_dir)
    out: list[HarborViolation] = []
    for rel, digest in sorted(want.items()):
        q = base / rel
        if q.is_symlink() or not q.is_file():
            out.append(HarborViolation(rel, "eval_script_missing"))
        elif sha256_file(q) != digest:
            out.append(HarborViolation(rel, "eval_script_pin_mismatch"))
    out.extend(
        HarborViolation(rel, "eval_script_unpinned")
        for p in sorted(base.rglob("*"))
        if (p.is_file() or p.is_symlink())
        and "__pycache__" not in p.parts
        and (rel := p.relative_to(base).as_posix()) not in want
    )
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    """``pin`` subcommand: write ``eval_script_pins.json`` from a clone."""
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="workspace_harbor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pin = sub.add_parser("pin", help="regenerate eval_script_pins.json")
    pin.add_argument("--root", default=str(default_path(UPSTREAM_DIR)))
    pin.add_argument("--out", default=str(PINS_PATH))
    args = ap.parse_args(argv)
    root = cast("str", args.root)
    out = cast("str", args.out)
    pins = generate_pins(Path(root))
    Path(out).write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    print(f"wrote {sum(map(len, pins.values()))} digests for {len(pins)} tasks")  # noqa: T201
    return 0


class EvalScriptsError(BundleError):
    """The eval scripts about to run are not the ones the Harbor bundle vouches for."""

    def __init__(self, task: str, violations: tuple[HarborViolation, ...]) -> None:
        self.violations = violations
        super().__init__(
            f"{task}: eval scripts refused: "
            + "; ".join(f"{v.kind} {v.path}" for v in violations),
        )


class MissingBundleError(BundleError, FileNotFoundError):
    """A real-clone task was materialized without its Harbor bundle."""


def normalise_harbor_script(text: str) -> list[str]:
    """Strip only the documented Harbor additions and blank lines; return the rest."""
    return [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not any(p.match(line) for p in _HARBOR_ADDITIONS)
    ]


def eval_scripts_intact(
    hb: HarborBundle,
    scripts_dir: Path,
) -> tuple[HarborViolation, ...]:
    """Compare a directory of eval scripts against the bundle's ``tests/eval/scripts``.

    Every regular file of the reference is compared byte for byte (not only ``*.sh``:
    helper modules such as ``_label_guard.py`` are executed too).
    """
    ref = hb.root / "tests" / "eval" / "scripts"
    out: list[HarborViolation] = []
    for p in sorted(x for x in ref.rglob("*") if x.is_file()):
        rel = p.relative_to(ref).as_posix()
        q = Path(scripts_dir) / rel
        if q.is_symlink() or not q.is_file():
            out.append(HarborViolation(rel, "eval_script_missing"))
        elif sha256_file(q) != sha256_file(p):
            out.append(HarborViolation(rel, "eval_script_modified"))
    return tuple(out)


def check_eval_scripts(
    task: str,
    hb: HarborBundle,
    scripts_dir: Path,
) -> tuple[str, ...]:
    """Refuse an edited or swapped eval script; return the names of adapted ones.

    Policy (see the block comment above): a task Harbor does not adapt must match
    byte for byte; a normalised-only task must match after :func:`normalise_harbor_script`
    and differ on exactly the pinned files; a substantively adapted task must differ
    on exactly the pinned files and match the pinned sha256 of every native file
    (:data:`EVAL_SCRIPT_PINS`).

    Returns:
      adapted: Sorted names of scripts that differ from Harbor's, all expected.

    Raises:
      EvalScriptsError: Any deviation from the policy.

    """
    found = eval_scripts_intact(hb, scripts_dir)
    names = tuple(sorted(v.path for v in found))
    expected = ADAPTED_EVAL_SCRIPTS.get(task, ())
    bad: list[HarborViolation] = [v for v in found if v.path not in expected]
    if task in ADAPTED_EVAL_SCRIPTS:
        # a pinned file that now matches Harbor's also signals drift (or a swap)
        bad += [
            HarborViolation(n, "eval_script_expected_adapted")
            for n in expected
            if n not in names
        ]
    if task in NORMALISED_ONLY:
        ref = hb.root / "tests" / "eval" / "scripts"
        for v in found:
            q = Path(scripts_dir) / v.path
            if v.kind == "eval_script_modified" and normalise_harbor_script(
                (ref / v.path).read_text(errors="replace"),
            ) != normalise_harbor_script(q.read_text(errors="replace")):
                bad.append(HarborViolation(v.path, "eval_script_modified"))
    if task in pinned_tasks():
        bad += check_native_pins(task, scripts_dir)
    if bad:
        raise EvalScriptsError(task, tuple(bad))
    return names


if __name__ == "__main__":
    sys.exit(main())
