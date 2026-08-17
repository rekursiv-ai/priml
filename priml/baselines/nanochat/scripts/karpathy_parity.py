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

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import argparse
import ast
import importlib
import subprocess
import sys
import types

from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch

from priml.baselines.nanochat.experiments import exp001
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
    vocab_size: int,
) -> tuple[nn.Module, torch.optim.Optimizer, types.ModuleType]:
    """The reference's own model and optimizer, built by its own module scope.

    Neither is constructed here. Their script derives the geometry from its
    ``DEPTH`` and its context length from its own ``constants``, and builds
    the optimizer from its own learning rates -- so passing any of those in
    would prove only that two copies of the same numbers agree.

    The vocabulary is the one exception: it comes from a tokenizer fitted on a
    corpus, which this box does not have, so the stub reports one.

    Args:
      root: The clone's path.
      vocab_size: Vocabulary their tokenizer stub reports.

    Returns:
      model: Their ``GPT``.
      optimizer: Their ``MuonAdamW``, over that model's parameters.
      module: Their ``train`` module, whose schedule functions the caller
        applies exactly as their own training loop does.

    """
    upstream = load_upstream(root, vocab_size=vocab_size)
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


def load_upstream(root: Path, *, vocab_size: int) -> types.ModuleType:
    """Import their ``train.py`` as itself, with its training loop cut short.

    Their file is a script: its module scope builds a tokenizer and a
    dataloader and then trains for five minutes. It is imported rather than
    read, so every class, hyperparameter, and constant is theirs -- the only
    things supplied from here are the three names it imports from outside
    (``kernels``, ``prepare``, ``constants``), and the dataloader stub ends
    module scope before the training loop by raising.

    Args:
      root: The clone's path.
      vocab_size: Vocabulary their tokenizer stub reports.

    Returns:
      module: Their ``train`` module.

    Raises:
      RuntimeError: Module scope aborted before defining what the comparison
        needs, so the stubs no longer match what their script expects.

    """
    sys.path.insert(0, str(root))
    sys.modules["kernels"] = _kernels_stub()
    sys.modules["prepare"] = _prepare_stub(vocab_size)
    # Their own knob, at the value that ends their training loop as early as
    # their ``step > 10`` guard allows. Their context length is left alone.
    constants = importlib.import_module("constants")
    constants.TIME_BUDGET = 1e-9  # ty: ignore[unresolved-attribute] -- dynamically imported module; attributes unknowable  # pyright: ignore[reportAttributeAccessIssue] -- dynamically imported module

    try:
        module = importlib.import_module("train")
    except _StopModuleScopeError as stop:
        module = _module_from_traceback(stop)
    for required in ("GPT", "GPTConfig", "model", "optimizer"):
        if not hasattr(module, required):
            raise RuntimeError(f"train.py aborted before defining {required}")
    return module


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


def compare_init(
    theirs: nn.Module,
    ours: nn.Module,
    mapping: dict[str, str],
) -> list[str]:
    """Compare the two INITIALIZATIONS, before either is overwritten.

    This has to run before :func:`copy_weights`, which writes their draws into
    ours so every later step compares the recipe rather than the RNG. That copy
    also makes the rest of this script structurally blind to an init bug: it
    erased a token table drawn 1/sqrt(2) too narrow, and five steps of
    bit-identical output said nothing about it.

    Draws are random, so this compares DISTRIBUTIONS rather than values: the
    per-tensor standard deviation, at a tolerance loose enough for sampling
    noise on the smallest tensor here and far tighter than any real init
    mistake, which lands at a ratio like 0.707 or 0.88 rather than 1.001.
    A tensor both sides initialize to a constant is compared exactly, since
    there is nothing random about it.

    Args:
      theirs: The reference model, freshly initialized.
      ours: This package's model, freshly initialized.
      mapping: Our parameter names to theirs.

    Returns:
      problems: One description per tensor whose spread disagrees.

    """
    src = dict(theirs.named_parameters())
    dst = dict(ours.named_parameters())
    problems: list[str] = []
    for our_name, their_name in mapping.items():
        a = src[their_name].detach().float()
        b = dst[our_name].detach().float()
        if a.shape != b.shape:
            problems.append(
                f"init {our_name}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
            )
            continue
        # A constant tensor (the per-layer scalars, a zeroed projection) has no
        # spread to compare, and its std is 0 on both sides -- which every
        # ratio test passes vacuously. Compared by value instead.
        if not a.std() and not b.std():
            if not torch.equal(a, b):
                problems.append(
                    f"init {our_name}: CONSTANT {a.flatten()[0]:.6g} vs "
                    f"{b.flatten()[0]:.6g}",
                )
            continue
        ratio = float(b.std() / a.std()) if a.std() else float("inf")
        if abs(ratio - 1.0) > 0.05:
            problems.append(
                f"init {our_name}: STD {float(a.std()):.6g} vs "
                f"{float(b.std()):.6g} (ours/theirs = {ratio:.4f})",
            )
    return problems


def copy_weights(theirs: nn.Module, ours: nn.Module, mapping: dict[str, str]) -> None:
    """Write the reference's initialization into ours.

    Every later comparison then measures the recipe rather than two RNG
    streams. It also destroys our own draws, so :func:`compare_init` must have
    run first -- an init bug is invisible from here onward.

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

    ours = build_ours(device=args.device)
    model = ours.config.model

    # Their script sizes its model from its own tokenizer, which this box does
    # not have -- so the stub reports OUR vocabulary and context. Every other
    # number on their side stays theirs.
    theirs, their_optimizer, upstream = build_theirs(
        root,
        vocab_size=model.vocab_size,
    )
    mapping = name_map(theirs, layers=model.num_layers)

    # BEFORE the copy: it overwrites our draws with theirs, so this is the only
    # point at which our initialization still exists to be checked.
    init_problems = compare_init(theirs, ours.model, mapping)
    print(f"\n[0] init distributions: {len(init_problems)} differ")
    for line in init_problems:
        print(f"    {line}")

    copy_weights(theirs, ours.model, mapping)

    problems = compare_all(theirs, ours.model, mapping, grads=False, tag="init")
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
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for index in range(1, args.steps + 1):
        shape = (args.rows, model.max_seq_len)
        tokens = torch.randint(
            0,
            model.vocab_size,
            shape,
            generator=generator,
            device=args.device,
        )
        targets = torch.randint(
            0,
            model.vocab_size,
            shape,
            generator=generator,
            device=args.device,
        )

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
        # Progress is written rather than measured: theirs and ours both read
        # it off a wall clock, so a comparison that let it run would freeze how
        # fast this machine is. Zero keeps both at the schedules' start, which
        # is where five steps of a budgeted run actually sit.
        progress = 0.0
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

    verdict = f"{failures} DIFFERENCE(S)" if failures else "BIT-IDENTICAL"
    print(f"\n{args.steps} steps, FA3->FA2 only: {verdict}")
    return 1 if failures else 0


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


def _prepare_stub(vocab_size: int) -> types.ModuleType:
    """The data-side names their ``train.py`` imports.

    None of the three is under test: the comparison is of the model and the
    optimizer, and both sides are fed the same tokens from outside.
    """
    module = types.ModuleType("prepare")

    class Tokenizer:
        """Enough of theirs for their module scope to size a model."""

        @classmethod
        def from_directory(cls) -> Tokenizer:
            return cls()

        def get_vocab_size(self) -> int:
            return vocab_size

    def make_dataloader(*args: Any, **kwargs: Any) -> Any:
        """Ends their module scope at the prefetch, before any training.

        Their own exit is gated on ``step > 10`` (train.py:601), so no budget
        makes it short, and letting even one iteration run would leave their
        weights a step ahead of the comparison. Raising here unwinds at their
        line 509: after every class, kernel, hyperparameter, and the model and
        optimizer they expose -- and before the loop touches any of it.

        Their schedule functions are defined below this point and so are not
        on the module; :func:`their_schedules` reads them out of the source
        instead.
        """
        del args, kwargs
        raise _StopModuleScopeError

    def evaluate_bpb(*args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return 0.0

    module.Tokenizer = Tokenizer  # ty: ignore[unresolved-attribute] -- stub module built at runtime  # pyright: ignore[reportAttributeAccessIssue] -- stub module built at runtime
    module.make_dataloader = make_dataloader  # ty: ignore[unresolved-attribute] -- stub module built at runtime  # pyright: ignore[reportAttributeAccessIssue] -- stub module built at runtime
    module.evaluate_bpb = evaluate_bpb  # ty: ignore[unresolved-attribute] -- stub module built at runtime  # pyright: ignore[reportAttributeAccessIssue] -- stub module built at runtime
    return module


def _module_from_traceback(stop: _StopModuleScopeError) -> types.ModuleType:
    """Recover the half-built ``train`` module from an aborted import.

    A module whose execution raised is removed from ``sys.modules``, so it has
    to come from the traceback: the frame running ``train.py`` holds the
    globals the comparison needs.
    """
    frame = stop.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code.co_filename.endswith("train.py"):
            module = types.ModuleType("train")
            module.__dict__.update(frame.tb_frame.f_globals)
            return module
        frame = frame.tb_next
    raise RuntimeError("train.py frame not found in traceback") from stop


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
    parser.add_argument("--steps", type=int, default=5, help="Optimizer steps.")
    parser.add_argument("--seed", type=int, default=42, help="Batch seed.")
    parser.add_argument("--device", default="cuda", help="Device to compare on.")
    # The reference sizes its model from its own DEPTH, so only what it reads
    # from OUTSIDE its script is set here: the tokenizer's vocabulary and the
    # context length. Both are small because the question is whether two
    # implementations agree, which a divergence answers in the first layer.
    parser.add_argument("--vocab", type=int, default=64, help="Vocabulary size.")
    parser.add_argument("--seq", type=int, default=32, help="Context length.")
    # An input, not the recipe: both models see the same tokens, and the
    # recipe's own batch would need more memory than one card holds while two
    # copies of the model are resident.
    parser.add_argument("--rows", type=int, default=1, help="Sequences compared.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
