"""Tests for the typed views of the pinned TMax, Ray, and vLLM surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from priml.baselines.tmax.scripts import custom_types


@dataclass(kw_only=True, slots=True)
class _Getter:
    value: object

    def get_runtime_env(self) -> object:
        return self.value


@dataclass(kw_only=True, slots=True)
class _Attribute:
    runtime_env: object


@pytest.mark.parametrize(
    ("context", "getter", "attribute"),
    [
        (_Getter(value={}), True, False),
        (_Attribute(runtime_env={}), False, True),
        (object(), False, False),
    ],
)
def test_runtime_context_protocols_distinguish_ray_api_shapes(
    context: object,
    *,
    getter: bool,
    attribute: bool,
) -> None:
    # The runner branches on these isinstance checks across Ray versions.
    assert isinstance(context, custom_types.HasGetRuntimeEnv) is getter
    assert isinstance(context, custom_types.HasRuntimeEnv) is attribute


def test_protocols_name_exactly_the_members_the_driver_reads() -> None:
    # Each view lists only the upstream members the driver calls or sets.
    assert set(custom_types.StreamingConfig.__annotations__) == {
        "dataset_mixer_list",
        "dataset_mixer_list_splits",
        "dataset_mixer_eval_list",
        "dataset_mixer_eval_list_splits",
        "max_prompt_token_length",
        "response_length",
        "num_unique_prompts_rollout",
        "num_samples_per_prompt_rollout",
        "temperature",
        "active_sampling",
        "filter_zero_std_samples",
        "async_steps",
        "inflight_updates",
        "mask_tool_use",
        "system_prompt_override_file",
        "stop_strings",
    }
    assert set(custom_types.GRPOArgs.__annotations__) == {
        "seed",
        "hf_entity",
        "model_name_or_path",
        "verification_reward",
        "lm_head_fp32",
        "enable_queue_dashboard",
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
