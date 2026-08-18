#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Prove ``exp001`` is bit-identical to the recipe it reproduces.

Clones karpathy/autoresearch at the pinned commit, imports its ``train.py``
UNMODIFIED, and steps it beside this package's own train step on the same
tokens. Every parameter and every gradient is compared with ``torch.equal`` at
each step; nothing is compared with a tolerance.

Exactly one thing is changed, on THEIR side and ours alike: the attention
kernel. Their ``train.py`` imports FlashAttention-3, which builds only for
SM90, so on any other card the comparison cannot run at all -- a ``kernels``
stub hands their own ``fa3`` symbol ``exp001``'s portable SDPA kernel
instead, which is the same function ``exp001`` puts in its own kernel slot.
Both sides then issue one kernel and every remaining difference is the
port's.

Nothing else about their side is supplied by this script. Their module scope
builds their model and their optimizer from their own constants; this script
reads the objects it produced, and mirrors the geometry THEY derived. Passing
our own hyperparameters into their ``setup_optimizer`` would prove only that
two copies of the same numbers agree.

Examples:
  karpathy_parity.py
  karpathy_parity.py --steps 10 --device cuda

'''
# fmt: on

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast, override

import argparse
import ast
import contextlib
import importlib
import re
import subprocess
import sys
import types

from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch

from priml.baselines.nanochat.experiments import exp001
from priml.math.seed import RngState, get_rng_state, set_rng_state
from priml.model.value_gated_attention import sdpa_attention
from priml.train.parallelism import NoParallel


class _StopModuleScopeError(Exception):
    """Ends ``train.py``'s training loop from the dataloader it asks us for."""


def their_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
) -> Tensor:
    """``exp001``'s own kernel, behind FlashAttention-3's call signature.

    The reference calls ``fa3.flash_attn_func(q, k, v, causal=True,
    window_size=(w, 0))``; this hands that call to the very function
    ``exp001`` puts in its kernel slot, so both sides issue ONE kernel and
    what remains between them is the recipe rather than the backend.

    Args:
      q: ``[B, S, heads, channels_head]`` queries.
      k: Keys, same shape.
      v: Values, same shape.
      causal: Whether the mask is causal; the recipe always passes True.
      window_size: ``(history, future)``; the recipe always passes future 0.

    Returns:
      out: Attention output, same shape as ``q``.

    """
    assert causal, "the recipe attends causally"
    assert window_size[1] == 0, f"unexpected future window {window_size[1]}"
    return sdpa_attention(q, k, v, window=window_size[0])


def clone_upstream(
    root: Path,
    *,
    url: str = "https://github.com/karpathy/autoresearch.git",
    commit: str = "b11d6f283f866eb7e10fb776a4b8553fef873fd5",
) -> Path:
    """Clone the reference at its pinned commit, or verify an existing clone.

    Args:
      root: Directory the clone lives in.
      url: Repository to clone.
      commit: Revision the comparison is against.

    Returns:
      path: The clone's path.

    Raises:
      RuntimeError: An existing clone is dirty or at another commit, so what
        it contains is no longer the reference this comparison names.

    """
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603 -- fixed URL and commit from the signature
            ["git", "clone", "--quiet", url, str(root)],  # noqa: S607
            check=True,
        )
        subprocess.run(  # noqa: S603 -- fixed URL and commit from the signature
            ["git", "checkout", "--quiet", commit],  # noqa: S607
            cwd=root,
            check=True,
        )
    head = _git(root, "rev-parse", "HEAD")
    if head != commit:
        raise RuntimeError(f"clone is at {head}, expected {commit}")
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"clone has local modifications:\n{dirty}")
    return root


def build_theirs(
    root: Path,
    *,
    corpus: Path,
    loader: dict[str, Any],
    rows: int,
    rng: dict[str, Any],
) -> tuple[nn.Module, torch.optim.Optimizer, types.ModuleType]:
    """The reference's own model and optimizer, built by its own module scope.

    Neither is constructed here. Their script derives the geometry from its
    ``DEPTH`` and its context length from its own ``constants``, and builds
    the optimizer from its own learning rates -- so passing any of those in
    would prove only that two copies of the same numbers agree.

    The vocabulary is theirs too, read by their own ``Tokenizer`` from the
    prepared corpus -- as is the dataloader whose rows the comparison steps on.

    Args:
      root: The clone's path.
      corpus: Prepared shards and tokenizer, read by their loader.
      loader: Filled with their built dataloader under ``"train"``.
      rows: Rows per pass; theirs is 128, which two resident models cannot hold.
      rng: Filled with the RNG state their module scope seeded, under
        ``"state"``, so ours can draw from the same one.

    Returns:
      model: Their ``GPT``.
      optimizer: Their ``MuonAdamW``, over that model's parameters.
      module: Their ``train`` module, whose schedule functions the caller
        applies exactly as their own training loop does.

    """
    upstream = load_upstream(root, corpus=corpus, loader=loader, rows=rows, rng=rng)
    print(f"upstream: {upstream.__file__}")
    print(f"kernel:   {upstream.fa3.flash_attn_func.__qualname__}")
    # The module BENEATH their ``torch.compile`` wrapper (train.py:506). Both
    # sides then run eager, which is the only pairing that isolates the port:
    # a compiled graph fuses reductions differently, so comparing one side
    # compiled against the other eager measures inductor, not the recipe.
    model = getattr(upstream.model, "_orig_mod", upstream.model)
    optimizer = upstream.optimizer
    assert isinstance(model, nn.Module), type(model).__name__
    assert isinstance(optimizer, torch.optim.Optimizer), type(optimizer).__name__
    return model, optimizer, their_schedules(root, upstream)


def their_schedules(root: Path, module: types.ModuleType) -> types.ModuleType:
    """Bind their schedule functions onto their module.

    The three live below the point where module scope is stopped -- stopping
    later would let their loop take a real step and leave their weights ahead
    of the comparison -- so they are executed here, from their own source
    text, against their own globals. Retyping the formulas instead would
    compare our copy of a schedule with theirs.

    Args:
      root: The clone's path.
      module: Their half-built ``train`` module.

    Returns:
      module: The same module, with the schedules bound.

    """
    source = (root / "train.py").read_text().splitlines()
    tree = ast.parse("\n".join(source))
    wanted = {"get_lr_multiplier", "get_muon_momentum", "get_weight_decay"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(  # noqa: S102 -- their own function definitions, from the pinned clone
                compile(ast.Module([node], []), str(root / "train.py"), "exec"),
                module.__dict__,
            )
    missing = wanted - set(module.__dict__)
    if missing:
        raise RuntimeError(f"train.py defines no {sorted(missing)}")
    return module


def load_upstream(
    root: Path,
    *,
    corpus: Path,
    loader: dict[str, Any],
    rows: int,
    rng: dict[str, Any],
) -> types.ModuleType:
    """Import their ``train.py`` as itself, with its training loop cut short.

    Their file is a script: its module scope builds a tokenizer and a
    dataloader and then trains for five minutes. It is imported rather than
    read, so every class, hyperparameter, and constant is theirs -- the only
    thing supplied from here is the ``kernels`` name, since their
    FlashAttention-3 builds only for SM90.

    Their ``prepare`` is their own, so the loader this captures is the recipe's
    packer over the real corpus. It is taken at the moment their module scope
    asks for it (``train.py:508``), which is also where module scope is ended:
    letting even one iteration of their loop run would leave their weights a
    step ahead of the comparison.

    Args:
      root: The clone's path.
      corpus: Prepared shards and tokenizer, read by their loader.
      loader: Filled with their built dataloader under ``"train"``.
      rows: Rows per pass; theirs is 128, which two resident models cannot hold.
      rng: Filled with the RNG state their module scope seeded, under
        ``"state"``.

    Returns:
      module: Their ``train`` module.

    Raises:
      RuntimeError: Module scope aborted before defining what the comparison
        needs, so the substitutions no longer match what their script expects.

    """
    sys.path.insert(0, str(root))
    sys.modules["kernels"] = _kernels_stub()
    # Their own knob, at the value that ends their training loop as early as
    # their ``step > 10`` guard allows. Their context length is left alone.
    constants = importlib.import_module("constants")
    constants.TIME_BUDGET = 1e-9  # ty: ignore[unresolved-attribute] -- dynamically imported module; attributes unknowable  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    _prepare_module(corpus, loader)

    # Their 128 rows hold two resident models' activations on no card this runs
    # on; the comparison keeps BOTH models alive at once, which their script
    # never does. Rewritten in the source text because their file assigns it at
    # module scope, so any value handed in beforehand is overwritten the moment
    # the line runs. Rows are an INPUT here -- both sides see the same ones --
    # so this changes what is compared on, not the recipe being compared.
    source = (root / "train.py").read_text()
    source, count = re.subn(
        r"^DEVICE_BATCH_SIZE = \d+",
        f"DEVICE_BATCH_SIZE = {rows}",
        source,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(
            f"expected one DEVICE_BATCH_SIZE assignment in train.py; found {count}.",
        )
    module = types.ModuleType("train")
    module.__file__ = str(root / "train.py")
    # Registered BEFORE execution: their ``@dataclass`` resolves its own class's
    # module through ``sys.modules``, which holds nothing for a module built
    # here, and fails on a ``None`` before their first class exists.
    sys.modules["train"] = module
    # Captured where THEY seed (train.py:456-457), which is the state their
    # model is about to draw from. Ours is built from this same state, so the
    # two initializations compare as values rather than as spreads.
    with _capture_rng_after_seeding(rng), contextlib.suppress(_StopModuleScopeError):
        exec(compile(source, str(root / "train.py"), "exec"), module.__dict__)  # noqa: S102 -- their own script, from the pinned clone
    for required in ("GPT", "GPTConfig", "model", "optimizer"):
        if not hasattr(module, required):
            raise RuntimeError(f"train.py aborted before defining {required}")
    if "train" not in loader:
        raise RuntimeError("train.py never asked for a dataloader")
    return module


@contextlib.contextmanager
def _capture_rng_after_seeding(rng: dict[str, Any]) -> Generator[None]:
    """Record the RNG state their module scope seeds, before it draws.

    Their ``torch.manual_seed(42)`` (``train.py:456``) is followed by the CUDA
    seed and then by every draw their model makes. Wrapping the CUDA call is
    what puts the capture between the two: after both seeds are set, and before
    ``GPT(config)`` consumes any of it.

    Their init draws on the CUDA generator -- the model is materialized on the
    device (``train.py:482``) before ``init_weights`` runs -- so that generator
    is the one the comparison must rewind. ``torch.cuda.manual_seed`` does NOT
    initialize CUDA, and ``get_rng_state`` omits the CUDA entries until it is
    (``seed.py:353``), so the context is forced up first. Without it the capture
    holds CPU state only, the restore leaves the CUDA generator wherever their
    draws left it, and the init comparison reports differences it manufactured.

    Args:
      rng: Filled with the captured state under ``"state"``.

    Yields:
      context: Block in which their seeding is observed.

    """
    real_cuda_seed = torch.cuda.manual_seed

    def capture(seed: int) -> None:
        torch.cuda.init()
        real_cuda_seed(seed)
        rng.setdefault("state", get_rng_state())

    torch.cuda.manual_seed = capture  # ty: ignore[invalid-assignment] -- observes their seeding; restored below
    try:
        yield
    finally:
        torch.cuda.manual_seed = real_cuda_seed


def build_ours(*, device: str) -> Any:
    """``exp001``, unmodified.

    Not one field of the recipe is set here. ``exp001`` already IS ``exp000``
    with the portable SDPA kernel, which is the single deviation this
    comparison declares -- so anything assigned here would be a difference the
    comparison then could not see.

    Args:
      device: Device to build on.

    Returns:
      step: The built train step.

    """
    config = exp001().step
    config.parallelism = NoParallel.Config(device=device)
    return config.make()


def name_map(theirs: nn.Module, *, layers: int) -> dict[str, str]:
    """Map this package's parameter names onto the reference's.

    Args:
      theirs: The reference model, read for which layers carry a gate.
      layers: Depth of the stack.

    Returns:
      mapping: Our name to theirs, for every parameter on either side.

    """
    mapping: dict[str, str] = {
        "embed.inner.weight": "transformer.wte.weight",
        "lm_head.inner.weight": "lm_head.weight",
        "mix.running": "resid_lambdas",
        "mix.original": "x0_lambdas",
    }
    # ``torch.compile`` wraps the module, so its parameters answer to an
    # ``_orig_mod.`` prefix. Read off the model rather than assumed, since the
    # comparison runs their compiled path and ours alike.
    prefix = (
        "_orig_mod."
        if any(name.startswith("_orig_mod.") for name, _ in theirs.named_parameters())
        else ""
    )
    mapping = {ours: f"{prefix}{them}" for ours, them in mapping.items()}
    for layer in range(layers):
        ours, them = f"blocks.{layer}", f"{prefix}transformer.h.{layer}"
        mapping |= {
            f"{ours}.attn.proj_q.weight": f"{them}.attn.c_q.weight",
            f"{ours}.attn.proj_k.weight": f"{them}.attn.c_k.weight",
            f"{ours}.attn.proj_v.weight": f"{them}.attn.c_v.weight",
            f"{ours}.attn.proj_out.weight": f"{them}.attn.c_proj.weight",
            f"{ours}.ffn.up_proj.weight": f"{them}.mlp.c_fc.weight",
            f"{ours}.ffn.down_proj.weight": f"{them}.mlp.c_proj.weight",
        }
    # Read off THEIR model rather than enumerated: they build a gate and a
    # table only on alternating layers, so naming every layer would demand a
    # parameter they never created -- and a disagreement about WHICH layers
    # then surfaces as an unmapped name rather than passing silently.
    for name, _ in theirs.named_parameters():
        bare = name.removeprefix(prefix)
        if bare.startswith("value_embeds."):
            # Ours narrows its tables, so the parameter sits under the wrapper.
            layer = bare.split(".")[1]
            mapping[f"value_embeds.{layer}.inner.weight"] = name
        elif bare.endswith("attn.ve_gate.weight"):
            mapping[f"blocks.{bare.split('.')[2]}.attn.value_gate.weight"] = name
    return mapping


def _progress_at(index: int, *, warmup: int, budget_steps: int) -> float:
    """Budget progress a real run sits at on its ``index``-th update.

    The first ``warmup`` updates are unbilled on both sides, so progress is
    zero across them; each update after charges one step's share of a run
    that lasts ``budget_steps`` billed updates.

    Args:
      index: One-based update number.
      warmup: Updates excluded from the budget clock.
      budget_steps: Billed updates the whole run is expected to last.

    Returns:
      progress: Fraction of the budget spent, in ``[0, 1]``.

    """
    billed = max(0, index - 1 - warmup)
    return min(billed / budget_steps, 1.0)


def copy_weights(theirs: nn.Module, ours: nn.Module, mapping: dict[str, str]) -> None:
    """Write the reference's initialization into ours.

    Every later comparison then measures the recipe rather than two RNG
    streams. It also destroys our own draws, so the value comparison in
    ``main`` must have run first -- an init bug is invisible from here on.

    Args:
      theirs: The reference model.
      ours: This package's model.
      mapping: Our parameter names to theirs.

    Raises:
      RuntimeError: A parameter on either side is unmapped, which means the
        two models are not the same architecture.

    """
    src = dict(theirs.named_parameters())
    dst = dict(ours.named_parameters())
    unmapped = sorted(k for k in dst if k not in mapping)
    absent = sorted(v for v in mapping.values() if v not in src)
    if unmapped or absent:
        raise RuntimeError(f"name map incomplete: {unmapped=} {absent=}")
    with torch.no_grad():
        for our_name, their_name in mapping.items():
            dst[our_name].copy_(src[their_name].to(dst[our_name].dtype))


def compare(label: str, a: Tensor, b: Tensor) -> str | None:
    """Describe how two tensors differ, or None when they are identical.

    Args:
      label: Name reported with a difference.
      a: The reference's tensor.
      b: Ours.

    Returns:
      problem: A description, or None.

    """
    if a.shape != b.shape:
        return f"{label}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"{label}: DTYPE {a.dtype} vs {b.dtype}"
    if torch.equal(a, b):
        return None
    return f"{label}: DIFFERS max_abs={(a.float() - b.float()).abs().max():.3e}"


def compare_all(
    theirs: nn.Module,
    ours: nn.Module,
    mapping: dict[str, str],
    *,
    grads: bool,
    tag: str,
) -> list[str]:
    """Compare every mapped parameter, or every mapped gradient.

    Args:
      theirs: The reference model.
      ours: This package's model.
      mapping: Our parameter names to theirs.
      grads: Compare gradients rather than the parameters themselves.
      tag: Prefix for each reported difference.

    Returns:
      problems: One description per differing tensor.

    """
    src = dict(theirs.named_parameters())
    dst = dict(ours.named_parameters())
    problems: list[str] = []
    for our_name, their_name in mapping.items():
        a, b = src[their_name], dst[our_name]
        if grads:
            if a.grad is None or b.grad is None:
                problems.append(f"{tag} {our_name}: MISSING gradient")
                continue
            a, b = a.grad, b.grad
        found = compare(f"{tag} {our_name}", a.detach(), b.detach())
        if found:
            problems.append(found)
    return problems


def compare_state(
    theirs: nn.Module,
    their_optimizer: torch.optim.Optimizer,
    ours: Any,
    mapping: dict[str, str],
) -> list[str]:
    """Compare the optimizers' own state, tensor by tensor.

    The moments and momentum buffers are what carry a difference from one step
    to the next, so a comparison that reads only weights and gradients reports
    agreement on the step that diverges and a mystery on the step after.

    Args:
      theirs: The reference model.
      their_optimizer: The reference's optimizer, read for its state.
      ours: This package's train step.
      mapping: Our parameter names to theirs.

    Returns:
      problems: One description per differing state tensor.

    """
    names = {
        "first_moment": "exp_avg",
        "second_moment": "exp_avg_sq",
        "momentum_buffer": "momentum_buffer",
    }
    src = dict(theirs.named_parameters())
    dst = dict(ours.model.named_parameters())
    problems: list[str] = []
    for our_name, their_name in mapping.items():
        mine: dict[str, Any] = {}
        for member in ours.optimizer.optimizers:
            if dst[our_name] in member.state:
                mine = member.state[dst[our_name]]
        other = their_optimizer.state.get(src[their_name], {})
        for our_key, their_key in names.items():
            key = their_key if their_key in other else "second_momentum_buffer"
            if our_key not in mine or key not in other:
                continue
            a, b = other[key], mine[our_key]
            if a.shape != b.shape:
                continue
            found = compare(f"state {our_name}[{our_key}]", a, b)
            if found:
                problems.append(found)
    return problems


def main() -> int:
    """Step both implementations together and report every difference."""
    args = _parse_args()
    root = clone_upstream(args.clone)

    # THEIRS first, and the RNG state captured at the moment their module scope
    # has seeded (train.py:456-457) and is about to draw. Ours is then built
    # from that same state, so the two initializations are compared as VALUES
    # rather than as distributions -- a draw-order or a fan-in difference shows
    # up here instead of hiding behind a matching standard deviation.
    their_loader: dict[str, Any] = {}
    theirs, their_optimizer, upstream = build_theirs(
        root,
        corpus=args.corpus,
        loader=their_loader,
        rows=args.rows,
        rng=(seeded := {}),
    )
    train_loader = their_loader["train"]

    set_rng_state(cast("RngState", seeded["state"]))
    ours = build_ours(device=args.device)
    model = ours.config.model
    mapping = name_map(theirs, layers=model.num_layers)

    # BEFORE the copy: it overwrites our draws with theirs, so this is the only
    # point at which our initialization still exists to be checked.
    init_problems = compare_all(theirs, ours.model, mapping, grads=False, tag="init")
    print(f"\n[0] init from one RNG state: {len(init_problems)} differ")
    for line in init_problems[:8]:
        print(f"    {line}")

    copy_weights(theirs, ours.model, mapping)

    problems = compare_all(theirs, ours.model, mapping, grads=False, tag="copied")
    problems = init_problems + problems
    print(f"[0] init weights copied: {len(problems) - len(init_problems)} differ")
    for line in problems[len(init_problems) :]:
        print(f"    {line}")
    # Read off the built ATTENTIONS: each layer carries its own window, so this
    # reports what the stack will actually attend over rather than the pattern
    # it was asked for.
    print(
        f"    windows ours={[b.attn.config.window for b in ours.model.blocks]} "
        f"theirs={[w[0] for w in theirs.window_sizes]}",
    )

    autocast = torch.amp.autocast(device_type=args.device, dtype=torch.bfloat16)
    theirs.train()
    ours.model.train()
    # The recipe expresses its window as a MASK, which disqualifies every
    # flash backend and lands on the memory-efficient one -- whose backward
    # torch documents as non-deterministic, and which measurably is: repeated
    # runs of the same second step differed in 3 to 50 gradients. Both sides
    # are pinned to the math backend so a difference between them is the port
    # rather than the kernel's own scatter order.
    stack = ExitStack()
    stack.enter_context(sdpa_kernel(SDPBackend.MATH))
    failures = len(problems)

    for index in range(1, args.steps + 1):
        # THEIR loader, over the real corpus. The packer is a stateful stream --
        # best-fit out of a document buffer refilled a fixed number at a time --
        # so it is part of the recipe rather than a fixture, and random ids left
        # it the one piece of the port nothing here compared. Taking it from
        # their side keeps the reference virgin: what our packer produces is a
        # separate question, and answering it with our own rows would let a
        # packing difference cancel itself on both sides of the comparison.
        tokens, targets, _ = next(train_loader)

        with autocast:
            their_loss = theirs(tokens, targets)
        with autocast:
            logits = ours.model(tokens)
        # Their forward folds the loss in; ours returns logits, so the same
        # cross-entropy is spelled here rather than compared through a
        # different reduction.
        our_loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1).long(),
            ignore_index=-1,
        )
        loss_problem = compare("loss", their_loss.detach(), our_loss.detach())

        their_loss.backward()
        our_loss.backward()
        grad_problems = compare_all(
            theirs,
            ours.model,
            mapping,
            grads=True,
            tag="grad",
        )

        # Their schedules, from their own functions, exactly as their training
        # loop applies them (train.py:552-561). Ours applies its own inside
        # ``_apply_update``; stepping either optimizer bare would compare a run
        # whose momentum never ramps against one whose does.
        #
        # Progress is SUPPLIED, identically to both sides, rather than measured
        # on either: both read it off a wall clock, so letting it run would
        # freeze how fast this machine is and compare two different schedules.
        # It follows the fencepost a real run has -- the first
        # ``budget_warmup_steps`` updates charge nothing (train.py:576, ours at
        # train_step.py:506), so progress is pinned at zero across them and
        # advances one step's share afterwards. That is what carries the LR
        # curve, the weight-decay ramp, and the momentum ramp into the
        # comparison instead of sampling one point of each.
        progress = _progress_at(
            index, warmup=args.warmup, budget_steps=args.budget_steps
        )
        multiplier = upstream.get_lr_multiplier(progress)
        for group in their_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
            if group["kind"] == "muon":
                group["momentum"] = upstream.get_muon_momentum(index - 1)
                group["weight_decay"] = upstream.get_weight_decay(progress)
        their_optimizer.step()

        ours.elapsed_sec = progress * ours.config.train_budget_sec
        # Written through the timer: ``global_step`` reads it and is read-only,
        # since a caller able to assign it could move the run's position out
        # from under the schedule. The momentum ramp is step-indexed, so the
        # count still has to be pinned to match theirs.
        ours.timer_step.global_count = index - 1
        ours._apply_update()  # noqa: SLF001 -- the schedules live here, and the loop that calls it also owns the clock
        theirs.zero_grad(set_to_none=True)
        state_problems = compare_state(theirs, their_optimizer, ours, mapping)
        weight_problems = compare_all(
            theirs,
            ours.model,
            mapping,
            grads=False,
            tag="weight",
        )

        step_problems = [*([loss_problem] if loss_problem else []), *grad_problems]
        step_problems += weight_problems + state_problems
        failures += len(step_problems)
        print(
            f"[{index}] loss {'DIFFERS' if loss_problem else 'identical'} | "
            f"grads {len(grad_problems)} differ | "
            f"weights {len(weight_problems)} differ | "
            f"state {len(state_problems)} differ",
        )
        for line in step_problems[:8]:
            print(f"    {line}")

    failures += compare_eval(
        theirs,
        ours,
        upstream,
        their_loader["prepare"],
        batches=args.eval_batches,
    )

    verdict = f"{failures} DIFFERENCE(S)" if failures else "BIT-IDENTICAL"
    print(f"\n{args.steps} steps, FA3->FA2 only: {verdict}")
    return 1 if failures else 0


def compare_eval(
    theirs: nn.Module,
    ours: Any,
    upstream: types.ModuleType,
    prepare: types.ModuleType,
    *,
    batches: int,
) -> int:
    """Score both models with THEIR metric and with ours, on their rows.

    The reported number is bits per byte, and until now nothing here touched
    it: twenty bit-identical updates say the weights agree, not that the two
    implementations turn the same weights into the same score. Their
    ``evaluate_bpb`` is marked DO NOT CHANGE (``prepare.py:324``) precisely
    because it IS the comparison, so it is the one run here -- against their
    model and ours in turn, which isolates the metric from the models.

    Ours is then run on the same rows through :class:`BitsPerByte`, so a
    difference in the ACCOUNTING shows up as well: theirs sums nats in float32
    and multiplies by a mask (``prepare.py:347``), ours accumulates float64 and
    indexes. Both drop zero-byte tokens, which is the part that must agree.

    Args:
      theirs: The reference model.
      ours: This package's train step.
      upstream: Their ``train`` module, for the tokenizer it built.
      prepare: Their ``prepare`` module, holding the metric and its constants.
      batches: Validation batches to score; their full evaluation is 40x
        larger and answers the same question far more slowly.

    Returns:
      failures: One per disagreement found.

    """
    tokenizer = upstream.tokenizer
    rows = int(upstream.DEVICE_BATCH_SIZE)
    # Their own metric, on each model in turn. Capped by rebinding the token
    # count their function divides by: it is a module constant they read, not
    # an argument, and the full 40 x 524,288 tokens take minutes to answer a
    # question a few batches settle.
    original = prepare.EVAL_TOKENS
    prepare.EVAL_TOKENS = batches * rows * int(prepare.MAX_SEQ_LEN)  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    # Under autocast, as their own final eval runs it (train.py:609-611): their
    # tables are held in bfloat16, so the model is only runnable inside one.
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    try:
        with autocast:
            their_bpb = float(upstream.evaluate_bpb(theirs, tokenizer, rows))
        with autocast:
            our_bpb = float(
                upstream.evaluate_bpb(_LossAdapter(ours.model), tokenizer, rows),
            )
    finally:
        prepare.EVAL_TOKENS = original  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module

    print(f"\n[eval] their metric: theirs={their_bpb:.9f} ours={our_bpb:.9f}")
    if their_bpb != our_bpb:
        print(f"    bpb DIFFERS by {abs(their_bpb - our_bpb):.3e}")
        return 1
    return 0


class _LossAdapter(nn.Module):
    """Present our model to their metric under their forward's signature.

    Their ``evaluate_bpb`` calls ``model(x, y, reduction='none')`` and expects
    per-token nats. Ours returns logits, so the same cross-entropy is spelled
    here -- once, in the place their metric reaches for it -- rather than
    reimplementing the metric around our shape.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.inner = model

    @override
    def forward(
        self, tokens: Tensor, targets: Tensor, reduction: str = "mean"
    ) -> Tensor:
        logits = self.inner(tokens)
        assert isinstance(logits, Tensor)
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1).long(),
            ignore_index=-1,
            reduction=reduction,
        )


def _git(root: Path, *arguments: str) -> str:
    """Run a read-only git command in the clone."""
    return subprocess.run(  # noqa: S603 -- fixed read-only subcommands from the caller
        ["git", *arguments],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _kernels_stub() -> types.ModuleType:
    """A ``kernels`` module whose ``get_kernel`` yields ``exp001``'s kernel."""
    module = types.ModuleType("kernels")

    def get_kernel(name: str) -> types.SimpleNamespace:
        assert "flash-attention-3" in name, name
        return types.SimpleNamespace(
            flash_attn_interface=types.SimpleNamespace(
                flash_attn_func=their_attention,
            ),
        )

    module.get_kernel = get_kernel  # ty: ignore[unresolved-attribute] -- stub module built at runtime  # pyright: ignore[reportAttributeAccessIssue] -- stub module built at runtime
    return module


def _prepare_module(corpus: Path, loader: dict[str, Any]) -> types.ModuleType:
    """THEIR ``prepare``, pointed at the prepared corpus.

    Their own module, not a stub: the packer it holds is a stateful stream --
    best-fit out of a document buffer refilled a fixed number at a time -- so
    it is part of the recipe being reproduced, and the rows it emits are what
    both sides must be stepped on. Only the two directories are rebound, since
    their file computes both from ``~/.cache`` at import.

    ``TOKENIZER_DIR`` is also a DEFAULT ARGUMENT of ``Tokenizer.from_directory``,
    bound at definition and so unaffected by the rebinding; their ``train.py``
    calls it with no argument, so the default is replaced too.

    Their ``make_dataloader`` is wrapped rather than replaced: the real one is
    called, its generator handed to ``loader``, and module scope then ended
    before their training loop -- so the comparison drives their own packer
    while their weights stay untouched.

    Args:
      corpus: Directory holding the shards and ``tokenizer/``.
      loader: Filled with the built dataloader under ``"train"``.

    Returns:
      module: Their ``prepare`` module.

    """
    module = importlib.import_module("prepare")
    module.DATA_DIR = str(corpus)  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    module.TOKENIZER_DIR = str(corpus / "tokenizer")  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    module.Tokenizer.from_directory.__func__.__defaults__ = (str(corpus / "tokenizer"),)
    real_make_dataloader = module.make_dataloader

    def make_dataloader(*args: Any, **kwargs: Any) -> Any:
        """Build their loader, keep it, and end their module scope.

        Restores their own function first: their ``evaluate_bpb`` builds a
        VALIDATION loader through the same name (``prepare.py:337``), and a
        wrapper still in place would end the scoring run instead.
        """
        module.make_dataloader = real_make_dataloader  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
        loader["train"] = real_make_dataloader(*args, **kwargs)
        loader["prepare"] = module
        raise _StopModuleScopeError

    module.make_dataloader = make_dataloader  # ty: ignore[unresolved-attribute] -- dynamically imported module  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module
    return module


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clone",
        type=Path,
        default=Path("/opt/scratch/karpathy-autoresearch"),
        help="Where the reference is cloned.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Optimizer steps.")
    # The unbilled prefix both sides share (train.py:576; train_step.py:506).
    # Progress is pinned at zero across it, so a default of 20 steps walks the
    # eleven warmup updates and nine billed ones after them.
    parser.add_argument(
        "--warmup",
        type=int,
        default=11,
        help="Updates excluded from the budget clock on both sides.",
    )
    parser.add_argument(
        "--budget-steps",
        type=int,
        default=191,
        help="Billed updates a full run lasts; sets one step's share.",
    )
    parser.add_argument("--device", default="cuda", help="Device to compare on.")
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=4,
        help="Validation batches scored; theirs is 40 x 524,288 tokens.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=8,
        help="Rows per pass; theirs is 128, and two models are resident here.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("/opt/scratch/datasets/nanochat-priml"),
        help="Prepared shards and tokenizer, read by THEIR loader.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
