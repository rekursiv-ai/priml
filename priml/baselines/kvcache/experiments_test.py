"""Tests for the KV-cache decode experiment configs.

``exp000`` is pinned whole by a pprint golden: its prompts, context length,
model, and dtype are the control every KV-cache fork is measured against.
"""

from __future__ import annotations

from priml.baselines.kvcache.experiments import exp000
from priml.testing.golden import assert_pprint_golden


def test_exp000_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp000_decodes_to_a_4096_token_context() -> None:
    cfg = exp000()
    assert cfg.prompt_len + cfg.num_decode_steps == 4096


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
