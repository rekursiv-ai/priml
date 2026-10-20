# The keyword bus

Every `forward`, every `cost`, and every metric `update` in priml takes an
open set of keyword arguments. Containers pass the unread part down to their
children; leaves, including every metric, only read. This is the bus.

## The rule

A layer declares the keywords it reads as keyword-only parameters. Everything
else lands in `**kwargs: object` and goes to its children unchanged.

```python
def cost(
    self,
    *,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype | None,
    **kwargs: object,
) -> Cost:
    return (
        cost(self.norm, seq_len=seq_len, batch_size=batch_size, dtype=dtype, **kwargs)
        + ...
    )
```

- **Peel:** a declared keyword is read, and may be re-passed with a new value
  (`seq_len=math.prod(grid)`).
- **Forward:** the remainder flows into each child's `**kwargs`.
- **Terminal:** a leaf that reads nothing ignores its `**kwargs`.

There are three buses. The metric bus is one level deep: the train loop
hands each metric the batch, and the metric reads it.

| Bus | Protocol | Typical keys |
|---|---|---|
| forward | `model/custom_types.py` `TensorModule`, `AttentionKernel` | `cache`, `attention_mask`, `window` |
| cost | `custom_types.py` `HasCost`, `cost.cost` | `seq_len`, `batch_size`, `dtype` |
| metric | `metrics/*` `update(logits, **batch)` | batch fields, `step_sec` |

## Why it is open

No protocol lists the keys. From `AttentionKernel`:

> The protocol does not enumerate that open set because doing so would make
> every new modular consumer a change to this shared interface.

A new consumer adds a key at the boundary and a reader at the leaf. No layer
in between changes.

## Correctness

Three properties hold by construction:

1. **Peeled and leftover never overlap.** Python binds a declared keyword to
   its parameter, so it is never in `**kwargs`. A layer cannot pass a key both
   explicitly and through the leftover.
2. **A missing required key fails where it is read.** A reader that needs a
   key declares it required, so a caller that forgets `batch_size` gets a
   `TypeError` instead of a cost for a guessed batch. A key with a natural
   default is declared with one (`cache: LayerCache | None = None`,
   `window: int = -1`).
3. **At a direct call, the leftover cannot reach a typed parameter.** The
   leftover is `object`. pyright and ty reject `object` for a parameter
   typed `int`:

   ```python
   def leaf(*, dtype: str, depth: int, **kwargs: object) -> None: ...
   def mid(*, dtype: str, **kwargs: object) -> None:
       leaf(dtype=dtype, **kwargs)


   # pyright: Argument of type "object" cannot be assigned to parameter "depth" of type "int"
   ```

   Every typed parameter must therefore be fed by an explicit, typed argument.

Some layers also read or edit the leftover directly, instead of declaring a
parameter:

- **Read:** `kwargs.get("puzzle_identifiers")` (`model/prefix.py`),
  `kwargs.get("input_grad", True)` (`model/linear.py`). The value is
  `object`, so the reader narrows it at runtime.
- **Drop:** `kwargs.pop("memory", None)` (`model/transformer/block.py`)
  keeps a key away from a child that would misread it.
- **Add:** `Qwen35.hidden_states` derives `attention_mask` and `positions`
  and puts them on the bus for its blocks (`model/transformer/qwen3_5.py`).

Property 3 still holds: an edited leftover is still `object`.

Injected callees are checked only at runtime. A slot holds whatever the
experiment config puts there, so the parent cannot name the callee's
signature statically. This is the price of injection, not a defect:

- **`cost(config: object, **kwargs: object)`.** The dispatcher costs any
  config. `cost(Linear.Config(...), seq_len="oops", ...)` type-checks and
  fails inside the leaf with `TypeError: rows must be an integer`.
- **The gradual `_Kwargs` ParamSpec.** `TensorModule` defaults its keyword
  signature to `...` so any kernel fits the slot. A parent can forward
  `**kwargs: object` into a kernel declaring `window: int`, and a
  wrong-typed `window` passes the checker.

At these boundaries the reader is the guard. A reader of a non-trivial key
validates it on entry, as `SparsePuzzleEmbedding.forward` does for
`puzzle_identifiers` (`model/prefix.py`) and `Qwen35GatedDeltaNet.forward`
does for `attention_mask` (`model/attention/qwen3_5_delta.py`).

A misspelled key is never read and never rejected. It flows to the leaves
and is dropped.

## Relation to prior work

The paper below does not cover the bus. Its calculus has no way to forward
leftover keys, and its unsoundness does not occur here.

Sun & Oliveira, [Named Arguments as Intersections, Optional Arguments as
Unions](https://doi.org/10.1007/978-3-031-91121-7_14) (ESOP 2025), formalize
first-class named arguments. They show mypy and Sorbet are unsound when a
dictionary's static type has forgotten a key that is still present at run
time and binds to a typed parameter. Their fix rewrites each call site from
the argument's static type, so only declared keys pass and forgotten ones are
dropped.

That fix does not fit the bus:

- **No rest parameter.** Their named-parameter grammar is
  `P ::= · | P; ℓ : A | P; ℓ? : A` (section 4.2): required and optional
  keys only. A callee cannot capture the remaining keys and pass them on.
- **Stripping at every call breaks depth.** A middle layer would drop every
  key it does not read, and no key would reach a leaf two levels down.
- **Their hazard is absent.** It needs an argument type that hides a key.
  The leftover is `object`, so nothing is hidden, and a direct call cannot
  bind it to a typed parameter (property 3).
- **Their fix needs the callee's signature.** Rewriting a call site reads
  the callee's parameter list. An injected callee has none at the call
  site, so the fix cannot apply even in principle.

The bus is closer to row polymorphism: a function typed "my fields plus a
row of whatever else arrives", passed on intact. Sun & Oliveira cite this
approach, Osinski's [row-polymorphic record
calculus](https://cs.nyu.edu/media/publications/osinski_edward.pdf) (2006,
section 3.5), as "free from the type-safety issue". In that notation a layer
is

    forall rho. {seq_len: int, batch_size: int, dtype: DType, rho} -> Cost

Python cannot write `rho`, so priml erases it to `**kwargs: object`.
