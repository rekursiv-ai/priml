"""Tests for the bits-per-byte metric."""

from __future__ import annotations

import math

import pytest
import torch

from priml.metrics.bits_per_byte import BitsPerByte


def _metric() -> BitsPerByte:
    return BitsPerByte.Config().make()


def test_it_retains_its_config() -> None:
    config = BitsPerByte.Config()
    assert BitsPerByte(config).config is config


def test_it_converts_nats_per_byte_to_bits() -> None:
    """A known loss over a known byte count has one right answer.

    Pinned against a closed form rather than a recorded number: two tokens of
    one byte each at ln(2) nats is exactly 1 bit per byte.
    """
    metric = _metric()
    metric.update(
        torch.full((2, 3), math.log(2)),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1, 2]),
    )
    assert metric.compute()["bpb"] == pytest.approx(1.0)


def test_a_longer_token_lowers_the_score() -> None:
    """The point of the metric: the same surprise over more bytes is cheaper.

    A per-token score would rank these two equal, which is exactly the bias
    that makes cross-entropy incomparable across tokenizers.
    """

    def score(*, token_bytes: list[int]) -> float:
        metric = _metric()
        metric.update(
            torch.full((2, 3), math.log(2)),
            label=torch.ones(2, 3, dtype=torch.int64),
            token_bytes=torch.tensor(token_bytes),
        )
        return metric.compute()["bpb"]

    assert score(token_bytes=[0, 4]) < score(token_bytes=[0, 1])


def test_zero_byte_tokens_leave_both_sums() -> None:
    """Document markers are formatting, not text.

    Charging the model for predicting one would score a convention, and a
    model cannot be right or wrong about where a document was cut.
    """
    metric = _metric()
    # Token 0 carries no bytes; its enormous loss must not reach the score.
    metric.update(
        torch.tensor([[math.log(2), 1e6]]),
        label=torch.tensor([[1, 0]]),
        token_bytes=torch.tensor([0, 1, 2]),
    )
    assert metric.compute()["bpb"] == pytest.approx(1.0)


def test_counts_accumulate_across_batches() -> None:
    """The ratio is computed once at the end, not averaged per batch.

    A mean of per-batch ratios would weight a short final batch equally with
    a full one.
    """
    metric = _metric()
    token_bytes = torch.tensor([0, 1])
    metric.update(
        torch.full((2, 4), math.log(2)),
        label=torch.ones(2, 4, dtype=torch.int64),
        token_bytes=token_bytes,
    )
    metric.update(
        torch.full((2, 3), 3 * math.log(2)),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=token_bytes,
    )
    # (4 + 3) bits over 5 bytes, not the mean of 1.0 and 3.0.
    assert metric.compute()["bpb"] == pytest.approx(13 / 7)


def test_padding_rows_leave_both_sums() -> None:
    """Rows squaring off a short final batch are not data.

    Marking them with a negative target is NOT enough on its own: a negative
    index reads the byte table from the back and lands on a real length, so
    the padding would be scored. ``valid_count`` is what removes it.
    """
    metric = _metric()
    metric.update(
        torch.full((3, 2), math.log(2)),
        label=torch.tensor([[1, 1], [1, 1], [-1, -1]]),
        token_bytes=torch.tensor([0, 1, 2]),
        valid_count=2,
    )
    # Four scored tokens of one byte each, and the padded row absent.
    assert metric.state_dict() == {"nats": pytest.approx(4 * math.log(2)), "bytes": 4}


def test_zero_valid_rows_are_allowed_and_leave_empty_state() -> None:
    metric = _metric()
    metric.update(
        torch.ones(2, 3),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1, 2]),
        valid_count=0,
    )
    assert metric.state_dict() == {"nats": 0.0, "bytes": 0}
    with pytest.raises(ValueError, match="no scored tokens"):
        metric.compute()


@pytest.mark.parametrize("valid_count", [-1, 3])
def test_a_valid_count_outside_the_batch_is_rejected(valid_count: int) -> None:
    """A count past the rows would index the byte table from the back."""
    metric = _metric()
    with pytest.raises(
        ValueError,
        match=(
            rf"valid_count {valid_count} is outside the batch's 2 rows; "
            r"the padding markers it excludes would otherwise index the byte "
            r"table from the back and be scored\."
        ),
    ):
        metric.update(
            torch.zeros(2, 3),
            label=torch.ones(2, 3, dtype=torch.int64),
            token_bytes=torch.tensor([0, 1, 2]),
            valid_count=valid_count,
        )


def test_update_converts_labels_and_moves_byte_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversions: list[tuple[object, ...]] = []
    original_to = torch.Tensor.to

    def recording_to(
        tensor: torch.Tensor,
        *args: torch.dtype | torch.device,
    ) -> torch.Tensor:
        conversions.append(args)
        arg = args[0]
        if isinstance(arg, torch.dtype):
            return original_to(tensor, arg)
        return original_to(tensor, arg)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    metric = _metric()
    metric.update(
        # Metric updates intentionally use a two-by-three batch/token matrix.
        torch.full((2, 3), math.log(2)),
        label=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        token_bytes=torch.tensor([0, 1, 2]),
    )
    assert (torch.int64,) in conversions
    assert (torch.device("cpu"),) in conversions
    assert metric.state_dict() == {"nats": pytest.approx(6 * math.log(2)), "bytes": 6}


def test_compute_sums_across_ranks_before_dividing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a process group the sums are all-reduced once, then divided.

    This rank holds 2 bits over 5 bytes and its peer 3 bits over 3 bytes:
    summed first that is 5/8; a mean of per-rank ratios would say 0.7.
    """

    def all_reduce(totals: torch.Tensor, op: object) -> None:
        assert totals.dtype == torch.float64
        assert op is torch.distributed.ReduceOp.SUM
        totals.add_(torch.tensor([3 * math.log(2), 3.0], dtype=torch.float64))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    metric = _metric()
    metric.update(
        torch.full((2, 3), math.log(2)),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1, 2, 4]),
    )
    assert metric.compute()["bpb"] == pytest.approx(1.0)


def test_non_gloo_reduction_moves_totals_to_current_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_device = torch.device

    def make_device(*args: object, **kwargs: object) -> torch.device:
        device_calls.append((args, kwargs))
        return real_device("cpu")

    def all_reduce(totals: torch.Tensor, op: object) -> None:
        assert totals.dtype == torch.float64
        assert op is torch.distributed.ReduceOp.SUM
        totals.add_(torch.tensor([3 * math.log(2), 3.0], dtype=torch.float64))

    metric = _metric()
    metric.update(
        torch.full((2, 3), math.log(2)),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1, 2, 4]),
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(torch, "device", make_device)
        scoped.setattr(torch.cuda, "current_device", lambda: 3)
        scoped.setattr(torch.distributed, "is_initialized", lambda: True)
        scoped.setattr(torch.distributed, "get_backend", lambda: "nccl")
        scoped.setattr(torch.distributed, "all_reduce", all_reduce)
        assert metric.compute()["bpb"] == pytest.approx(1.0)

    assert device_calls == [(("cuda", 3), {})]


def test_a_shape_disagreement_is_rejected() -> None:
    """Silently broadcasting would pair each loss with another token's length."""
    metric = _metric()
    with pytest.raises(ValueError, match="targets"):
        metric.update(
            torch.zeros(2, 3),
            label=torch.ones(3, 2, dtype=torch.int64),
            token_bytes=torch.tensor([0, 1, 2]),
        )


def test_an_empty_eval_refuses_rather_than_scoring_zero() -> None:
    """Zero is the BEST possible score, so an empty eval must not report it.

    Lower is better here, so a run whose eval loader yielded nothing -- a cap
    below one batch, a misconfigured split -- would otherwise win every
    comparison it entered, and look like a result rather than a failure.
    """
    with pytest.raises(
        ValueError,
        match=(
            r"bits per byte has no scored tokens: the evaluation produced "
            r"no batches carrying byte-bearing targets\."
        ),
    ):
        _metric().compute()


def test_state_round_trips() -> None:
    metric = _metric()
    metric.update(
        torch.full((2, 3), math.log(2)),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1, 2]),
    )
    restored = _metric()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()


def test_reset_clears_both_sums() -> None:
    """After a reset the metric holds nothing, so it refuses to score."""
    metric = _metric()
    metric.update(
        torch.full((2, 3), math.log(2)),
        label=torch.ones(2, 3, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1, 2]),
    )
    metric.reset()
    assert metric.state_dict() == {"nats": 0.0, "bytes": 0}
    with pytest.raises(ValueError, match="no scored tokens"):
        metric.compute()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
