#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
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

from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Literal, NoReturn, Protocol, cast, override

import argparse
import ast
import contextlib
import functools
import importlib
import re
import sys
import types

from torch import Tensor, nn
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch

from priml.baselines.nanochat.experiments import exp001
from priml.baselines.nanochat.scripts.karpathy_upstream import (
    clone_upstream,
    kernels_stub,
)
from priml.baselines.nanochat.train_step import NanoChatTrainStep
from priml.math.seed import RngState, get_rng_state, set_rng_state
from priml.optimizers.composite import CompositeOptimizer
from priml.train.parallelism import NoParallel


class _AttentionCallable(Protocol):
    __qualname__: str

    def __call__(self, *args: object, **kwargs: object) -> Tensor: ...


class _AttentionKernel(Protocol):
    flash_attn_func: _AttentionCallable


class _AttentionConfig(Protocol):
    window: int


class _NanoChatConfig(Protocol):
    num_layers: int


class _ReferenceModule(Protocol):
    __file__: str
    __name__: str
    fa3: _AttentionKernel
    model: nn.Module
    optimizer: torch.optim.Optimizer
    window_sizes: list[tuple[int, int]]
    tokenizer: object
    DEVICE_BATCH_SIZE: int
    get_lr_multiplier: Callable[[float], float]
    get_muon_momentum: Callable[[int], float]
    get_weight_decay: Callable[[float], float]
    evaluate_bpb: Callable[[nn.Module, object, int], float]


class _TokenizerMethod(Protocol):
    __func__: types.FunctionType


class _TokenizerClass(Protocol):
    from_directory: _TokenizerMethod


class _PrepareModule(Protocol):
    DATA_DIR: str
    TOKENIZER_DIR: str
    EVAL_TOKENS: int
    MAX_SEQ_LEN: int
    Tokenizer: _TokenizerClass
    make_dataloader: Callable[..., Iterator[tuple[Tensor, Tensor, object]]]


class _EvaluationPrepare(Protocol):
    EVAL_TOKENS: int
    MAX_SEQ_LEN: int


def build_theirs(
    root: Path,
    *,
    corpus: Path,
    loader: dict[str, object],
    rows: int,
    rng: dict[str, object],
) -> tuple[nn.Module, torch.optim.Optimizer, _ReferenceModule]:
    """Return the reference's own model and optimizer, built by its own module scope.

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
    # Module scope ends at their dataloader (train.py:508), after their
    # ``torch.compile`` wrapper (:506). Unwrapped to the module beneath it so
    # both sides run eager, the only pairing that isolates the port: a compiled
    # graph fuses reductions differently, so one side compiled against the
    # other eager measures inductor, not the recipe.
    model = _eager(upstream.model)
    optimizer = upstream.optimizer
    assert isinstance(model, nn.Module), type(model).__name__
    assert isinstance(optimizer, torch.optim.Optimizer), type(optimizer).__name__
    return model, optimizer, their_schedules(root, cast(types.ModuleType, upstream))


def their_schedules(root: Path, module: types.ModuleType) -> _ReferenceModule:
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
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    exec(  # noqa: S102 -- The parity harness executes function definitions from the pinned reference clone.
        compile(tree, str(root / "train.py"), "exec"),
        module.__dict__,
    )
    missing = wanted - set(module.__dict__)
    if missing:
        raise RuntimeError(f"train.py defines no {sorted(missing)}")
    return cast(_ReferenceModule, module)


def load_upstream(
    root: Path,
    *,
    corpus: Path,
    loader: dict[str, object],
    rows: int,
    rng: dict[str, object],
) -> _ReferenceModule:
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
    sys.modules["kernels"] = kernels_stub()
    # Their own knob, at the value that ends their training loop as early as
    # their ``step > 10`` guard allows. Their context length is left alone.
    constants = importlib.import_module("constants")
    constants.TIME_BUDGET = 1e-9  # ty: ignore[unresolved-attribute] -- The module is dynamically imported and has no static attributes.  # pyright: ignore[reportAttributeAccessIssue] -- The module is dynamically imported and has no static attributes.
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
        exec(compile(source, str(root / "train.py"), "exec"), module.__dict__)  # noqa: S102 -- The parity harness executes the pinned reference script unchanged.
    for required in ("GPT", "GPTConfig", "model", "optimizer"):
        if not hasattr(module, required):
            raise RuntimeError(f"train.py aborted before defining {required}")
    if "train" not in loader:
        raise RuntimeError("train.py never asked for a dataloader")
    return cast(_ReferenceModule, module)


def build_ours(*, device: str) -> NanoChatTrainStep:
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
    step = config.make()
    assert isinstance(step, NanoChatTrainStep)
    return step


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
    for layer in range(layers):
        ours, them = f"blocks.{layer}", f"transformer.h.{layer}"
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
        if name.startswith("value_embeds."):
            # Ours narrows its tables, so the parameter sits under the wrapper.
            layer = name.split(".")[1]
            mapping[f"value_embeds.{layer}.inner.weight"] = name
        elif name.endswith("attn.ve_gate.weight"):
            mapping[f"blocks.{name.split('.')[2]}.attn.value_gate.weight"] = name
    return mapping


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
    # Theirs too: a reference parameter no name maps to is never copied or
    # compared, so every later "identical" would be silent about it.
    uncovered = sorted(set(src) - set(mapping.values()))
    if unmapped or absent or uncovered:
        raise RuntimeError(
            f"name map incomplete: {unmapped=} {absent=} {uncovered=}",
        )
    with torch.no_grad():
        for our_name, their_name in mapping.items():
            dst[our_name].copy_(src[their_name])


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
        if found is not None:
            problems.append(found)
    return problems


def compare_state(
    theirs: nn.Module,
    their_optimizer: torch.optim.Optimizer,
    ours: NanoChatTrainStep,
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
      problems: One description per differing, missing, or one-sided state
        tensor.

    """
    src = dict(theirs.named_parameters())
    dst = dict(ours.model.named_parameters())
    # The recipe runs two optimizers; the state lives on each member.
    assert isinstance(ours.optimizer, CompositeOptimizer)
    their_state = cast(dict[Tensor, object], their_optimizer.state)
    # Our name to the names theirs may use. Their AdamW keeps ``exp_avg_sq`` and
    # their Muon ``second_momentum_buffer``; ours names both ``second_moment``. A
    # member holds one or the other, never both.
    names = {
        "first_moment": ("exp_avg",),
        "second_moment": ("exp_avg_sq", "second_momentum_buffer"),
        "momentum_buffer": ("momentum_buffer",),
    }
    problems: list[str] = []
    for our_name, their_name in mapping.items():
        mine = _member_state(ours.optimizer, dst[our_name])
        other_value = their_state.get(src[their_name], {})
        assert isinstance(other_value, dict)
        other = cast(dict[str, Tensor], other_value)
        for our_key, their_keys in names.items():
            problems.extend(
                _compare_state_entry(
                    f"state {our_name}[{our_key}]",
                    theirs=next(
                        (other[key] for key in their_keys if key in other),
                        None,
                    ),
                    ours=mine.get(our_key),
                ),
            )
    return problems


def main() -> int:
    """Step both implementations together and report every difference.

    Returns:
      result: Exit code (0 on success).

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    root = clone_upstream(flags.clone)

    # THEIRS first, and the RNG state captured at the moment their module scope
    # has seeded (train.py:456-457) and is about to draw. Ours is then built
    # from that same state, so the two initializations are compared as VALUES
    # rather than as distributions -- a draw-order or a fan-in difference shows
    # up here instead of hiding behind a matching standard deviation.
    their_loader: dict[str, object] = {}
    theirs, their_optimizer, upstream = build_theirs(
        root,
        corpus=flags.corpus,
        loader=their_loader,
        rows=flags.rows,
        rng=(seeded := {}),
    )
    train_loader = cast(Iterator[tuple[Tensor, Tensor, object]], their_loader["train"])

    set_rng_state(cast(RngState, seeded["state"]))
    ours = build_ours(device=flags.device)
    model = ours.model
    typed_model = cast(_NanoChatModel, model)
    mapping = name_map(theirs, layers=typed_model.config.num_layers)

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
    reference = cast(_UpstreamModel, theirs)
    blocks = cast(Iterator[_ModelBlock], model.blocks)
    print(
        f"    windows ours={[b.attn.window for b in blocks]} "
        f"theirs={[w[0] for w in reference.window_sizes]}",
    )

    theirs.train()
    ours.model.train()
    # The recipe expresses its window as a MASK, which disqualifies every
    # flash backend and lands on the memory-efficient one -- whose backward
    # torch documents as non-deterministic, and which measurably is: repeated
    # runs of the same second step differed in 3 to 50 gradients. Both sides
    # are pinned to the math backend so a difference between them is the port
    # rather than the kernel's own scatter order.
    with sdpa_kernel(SDPBackend.MATH):
        failures = len(problems) + _compare_steps(
            theirs,
            their_optimizer,
            ours,
            upstream=upstream,
            loader=train_loader,
            mapping=mapping,
            flags=flags,
        )
        failures += compare_eval(
            theirs,
            ours,
            upstream,
            cast(_PrepareModule, their_loader["prepare"]),
            batches=flags.eval_batches,
            device=flags.device,
        )

    verdict = f"{failures} DIFFERENCE(S)" if failures else "BIT-IDENTICAL"
    print(
        f"\n{flags.steps} steps, FA3->SDPA (math backend) on both sides: {verdict}",
    )
    return 1 if failures else 0


def compare_eval(
    theirs: nn.Module,
    ours: NanoChatTrainStep,
    upstream: _ReferenceModule,
    prepare: _EvaluationPrepare,
    *,
    batches: int,
    device: str,
) -> int:
    """Score both models with THEIR metric, on their rows.

    The reported number is bits per byte, and until now nothing here touched
    it: twenty bit-identical updates say the weights agree, not that the two
    implementations turn the same weights into the same score. Their
    ``evaluate_bpb`` is marked DO NOT CHANGE (``prepare.py:324``) precisely
    because it IS the comparison, so it is the one run here -- against their
    model and ours in turn, which isolates the models from the metric. Our own
    ``BitsPerByte`` accounting is not exercised here.

    Args:
      theirs: The reference model.
      ours: This package's train step.
      upstream: Their ``train`` module, for the tokenizer it built.
      prepare: Their ``prepare`` module, holding the metric and its constants.
      batches: Validation batches to score; their full evaluation is 40x
        larger and answers the same question far more slowly.
      device: Device type both models run on, for the autocast region.

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
    prepare.EVAL_TOKENS = batches * rows * int(prepare.MAX_SEQ_LEN)
    # Under autocast, as their own final eval runs it (train.py:609-611): their
    # tables are held in bfloat16, so the model is only runnable inside one.
    autocast = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
    try:
        with autocast:
            their_bpb = float(upstream.evaluate_bpb(theirs, tokenizer, rows))
        with autocast:
            our_bpb = float(
                upstream.evaluate_bpb(_LossAdapter(ours.model), tokenizer, rows),
            )
    finally:
        prepare.EVAL_TOKENS = original

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
        self,
        tokens: Tensor,
        targets: Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tensor:
        logits = cast(Tensor, self.inner(tokens))
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1).long(),
            ignore_index=-1,
            reduction=reduction,
        )


# Their own module, not a stub: the packer it holds is a stateful stream -- best-fit out
# of a document buffer refilled a fixed number at a time -- so it is part of the recipe
# being reproduced, and the rows it emits are what both sides must be stepped on. Only
# the two directories are rebound, since their file computes both from ``~/.cache`` at
# import.
#
# ``TOKENIZER_DIR`` is also a DEFAULT ARGUMENT of ``Tokenizer.from_directory``, bound at
# definition and so unaffected by the rebinding; their ``train.py`` calls it with no
# argument, so the default is replaced too.
#
# Their ``make_dataloader`` is wrapped rather than replaced: the real one is called, its
# generator handed to ``loader``, and module scope then ended before their training loop
# -- so the comparison drives their own packer while their weights stay untouched.
def _prepare_module(corpus: Path, loader: dict[str, object]) -> _PrepareModule:
    """THEIR ``prepare``, pointed at the prepared corpus."""
    module = cast(_PrepareModule, importlib.import_module("prepare"))
    module.DATA_DIR = str(corpus)
    module.TOKENIZER_DIR = str(corpus / "tokenizer")
    cast(types.FunctionType, module.Tokenizer.from_directory.__func__).__defaults__ = (
        str(corpus / "tokenizer"),
    )
    module.make_dataloader = functools.partial(
        _capture_loader,
        module=module,
        real=module.make_dataloader,
        loader=loader,
    )
    return module


# Restores their own function first: their ``evaluate_bpb`` builds a VALIDATION loader
# through the same name (``prepare.py:337``), and a wrapper still in place would end the
# scoring run instead.
def _capture_loader(
    *args: object,
    module: _PrepareModule,
    real: Callable[..., Iterator[tuple[Tensor, Tensor, object]]],
    loader: dict[str, object],
    **kwargs: object,
) -> NoReturn:
    """Build their loader, keep it, and end their module scope."""
    module.make_dataloader = real
    loader["train"] = real(*args, **kwargs)
    loader["prepare"] = module
    raise _StopModuleScopeError


class _StopModuleScopeError(Exception):
    """Ends ``train.py``'s training loop from the dataloader it asks us for."""


# Their ``torch.manual_seed(42)`` (``train.py:456``) is followed by the CUDA seed and
# then by every draw their model makes. Wrapping the CUDA call is what puts the capture
# between the two: after both seeds are set, and before ``GPT(config)`` consumes any of
# it.
#
# Their init draws on the CUDA generator -- the model is materialized on the device
# (``train.py:482``) before ``init_weights`` runs -- so that generator is the one the
# comparison must rewind. ``torch.cuda.manual_seed`` does NOT initialize CUDA, and
# ``get_rng_state`` omits the CUDA entries until it is (``seed.py:353``), so the context
# is forced up first. Without it the capture holds CPU state only, the restore leaves
# the CUDA generator wherever their draws left it, and the init comparison reports
# differences it manufactured.
@contextlib.contextmanager
def _capture_rng_after_seeding(rng: dict[str, object]) -> Generator[None]:
    """Record the RNG state their module scope seeds, before it draws."""
    real_cuda_seed = torch.cuda.manual_seed
    torch.cuda.manual_seed = functools.partial(
        _seed_and_capture,
        real=real_cuda_seed,
        rng=rng,
    )
    try:
        yield
    finally:
        torch.cuda.manual_seed = real_cuda_seed


def _seed_and_capture(
    seed: int,
    *,
    real: Callable[[int], None],
    rng: dict[str, object],
) -> None:
    """Seed CUDA as they asked, then record the first state after seeding."""
    torch.cuda.init()
    real(seed)
    rng.setdefault("state", get_rng_state())


def _eager(model: object) -> object:
    """Return the module beneath a ``torch.compile`` wrapper, or ``model`` itself."""
    if not isinstance(model, OptimizedModule):
        return model
    inner = cast(object, model._orig_mod)  # noqa: SLF001 -- torch's only handle on the module it wrapped.
    assert isinstance(inner, nn.Module)
    return inner


# The first ``warmup`` updates are unbilled on both sides, so progress is zero across
# them; each update after charges one step's share of a run that lasts ``budget_steps``
# billed updates.
def _progress_at(index: int, *, warmup: int, budget_steps: int) -> float:
    """Budget progress a real run sits at on its ``index``-th update."""
    billed = max(0, index - 1 - warmup)
    return min(billed / budget_steps, 1.0)


class _ModelBlock(Protocol):
    attn: _AttentionConfig


class _UpstreamModel(Protocol):
    window_sizes: list[tuple[int, int]]

    def __call__(self, tokens: Tensor, targets: Tensor) -> Tensor: ...


class _Flags(Protocol):
    """Parsed command-line flags."""

    clone: Path
    corpus: Path
    rows: int
    device: str
    steps: int
    warmup: int
    budget_steps: int
    eval_batches: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
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


class _NanoChatModel(Protocol):
    config: _NanoChatConfig


def _member_state(
    optimizer: CompositeOptimizer,
    parameter: Tensor,
) -> dict[str, Tensor]:
    """Return the state of the member owning ``parameter``; empty when none does."""
    for member in optimizer.optimizers:
        member_state = cast(dict[Tensor, object], member.state)
        if parameter in member_state:
            value = member_state[parameter]
            assert isinstance(value, dict)
            return cast(dict[str, Tensor], value)
    return {}


def _compare_state_entry(
    label: str,
    *,
    theirs: Tensor | None,
    ours: Tensor | None,
) -> list[str]:
    """Report a one-sided entry as a difference; absent on both sides agrees."""
    if theirs is None and ours is None:
        return []
    if theirs is None or ours is None:
        return [f"{label}: MISSING on {'theirs' if theirs is None else 'ours'}"]
    found = compare(label, theirs, ours)
    return [found] if found is not None else []


def _compare_steps(
    theirs: nn.Module,
    their_optimizer: torch.optim.Optimizer,
    ours: NanoChatTrainStep,
    *,
    upstream: _ReferenceModule,
    loader: Iterator[tuple[Tensor, Tensor, object]],
    mapping: dict[str, str],
    flags: _Flags,
) -> int:
    """Step both sides together on their rows; return the differences found."""
    reference = cast(_UpstreamModel, theirs)
    autocast = torch.amp.autocast(device_type=flags.device, dtype=torch.bfloat16)
    failures = 0
    for index in range(1, flags.steps + 1):
        # THEIR loader, over the real corpus. The packer is a stateful stream --
        # best-fit out of a document buffer refilled a fixed number at a time --
        # so it is part of the recipe rather than a fixture, and random ids left
        # it the one piece of the port nothing here compared. Taking it from
        # their side keeps the reference virgin: what our packer produces is a
        # separate question, and answering it with our own rows would let a
        # packing difference cancel itself on both sides of the comparison.
        tokens, targets, _ = next(loader)

        with autocast:
            their_loss = reference(tokens, targets)
        with autocast:
            logits = cast(Tensor, ours.model(tokens))
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
            index,
            warmup=flags.warmup,
            budget_steps=flags.budget_steps,
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
        ours._apply_update()  # noqa: SLF001 -- The parity comparison must invoke the implementation's private update hook.
        theirs.zero_grad(set_to_none=True)
        state_problems = compare_state(theirs, their_optimizer, ours, mapping)
        weight_problems = compare_all(
            theirs,
            ours.model,
            mapping,
            grads=False,
            tag="weight",
        )

        step_problems = [
            *([loss_problem] if loss_problem is not None else []),
            *grad_problems,
        ]
        step_problems += weight_problems + state_problems
        failures += len(step_problems)
        print(
            f"[{index}] loss {'DIFFERS' if loss_problem is not None else 'identical'} | "
            f"grads {len(grad_problems)} differ | "
            f"weights {len(weight_problems)} differ | "
            f"state {len(state_problems)} differ",
        )
        for line in step_problems[:8]:
            print(f"    {line}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
