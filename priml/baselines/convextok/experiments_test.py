"""exp000 is frozen: its finalized recipe must not move."""

from priml.baselines.convextok.experiments import exp000
from priml.testing.golden import assert_pprint_golden


def test_exp000_config_golden() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
