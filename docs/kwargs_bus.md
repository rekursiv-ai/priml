# The keyword bus

Every `forward`, every `cost`, and every metric `update` in priml takes an
open set of keyword arguments and passes the unread part down. This is the
bus.

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

There are three buses, all following this rule:

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

Four properties hold by construction:

1. **Peeled and leftover never overlap.** Python binds a declared keyword to
   its parameter, so it is never in `**kwargs`. A layer cannot pass a key both
   explicitly and through the leftover.
2. **A missing key fails where it is read.** Readers declare their keys as
   required keyword-only parameters, so a caller that forgets `batch_size`
   gets a `TypeError` instead of a cost for a guessed batch.
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
4. **Untouched keys stay untouched.** A layer never inspects or rewrites
   keys it did not declare.

Two boundaries are checked only at runtime:

- **`cost(config: object, **kwargs: object)`.** The dispatcher cannot see
  the callee's signature. `cost(Linear.Config(...), seq_len="oops", ...)`
  type-checks and fails inside the leaf with
  `TypeError: rows must be an integer`.
- **The gradual `_Kwargs` ParamSpec.** `TensorModule` defaults its keyword
  signature to `...`, so a parent holding a `TensorModule` can forward
  `**kwargs: object` into a kernel declaring `window: int`. A wrong-typed
  `window` passes the checker.

At these boundaries the reader is the guard. A reader of a non-trivial key
validates it on entry, as `SparsePuzzleEmbedding.forward` does for
`puzzle_identifiers` (`model/prefix.py`) and `Qwen35DeltaAttention.forward`
does for `attention_mask` (`model/attention/qwen3_5_delta.py`).

A misspelled key is never read and never rejected. It flows to the leaves
and is dropped.

## Relation to prior work

Sun & Oliveira, [Named Arguments as Intersections, Optional Arguments as
Unions](https://doi.org/10.1007/978-3-031-91121-7_14) (ESOP 2025), formalize
first-class named arguments. They show mypy and Sorbet are unsound when a
dictionary's static type has forgotten a key that is still present at run
time and binds to a typed parameter. Their fix rewrites each call site from
the argument's static type, so only declared keys pass and forgotten ones are
dropped.

That fix does not fit the bus:

- **No rest parameter.** Their functions have closed parameter lists. A
  callee cannot capture the remaining keys and pass them on.
- **Stripping at every call breaks depth.** A middle layer would drop every
  key it does not read, and no key would reach a leaf two levels down.
- **The hazard is mostly absent here.** The leftover is `object`, not a
  record type that hides a key, so a direct call cannot bind it to a typed
  parameter (property 3). What remains is the two dynamic boundaries above.

The bus is closer to row polymorphism: a function typed "my fields plus a
row of whatever else arrives", passed on intact (Ohori 1995; Osinski 2006,
section 3.5, both cited by Sun & Oliveira). In that notation a layer is

    forall rho. {seq_len: int, batch_size: int, dtype: DType, rho} -> Cost

Python cannot write `rho`, so priml erases it to `**kwargs: object`. The
erasure is what puts the two dynamic boundaries back on runtime checks.
