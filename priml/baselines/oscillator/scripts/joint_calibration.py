"""Check part four or five's learnability with a privileged joint fit of each history.

The fit knows every change time, shares the hidden chain and both tones across
regimes, and fits the whole live history (the frozen outage is found from
repeated readings). It reports the robust loss of the truth and of the best
restart, and the best restart's forecast skill against the zero forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

import argparse

import torch

from priml.baselines.oscillator.chained import (
    ChainedData,
    ChainedDynamics,
    ChainedShooting,
)
from priml.baselines.oscillator.forced import (
    ForcedData,
    ForcedDynamics,
    ForcedShooting,
)


if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, kw_only=True, slots=True)
class EpisodeCalibration:
    """One episode's fit, compared with the truth on the same readings."""

    regimes: int
    truth_loss: float
    best_loss: float
    skill: float


class _Recording(ChainedDynamics):
    """Log each step's coefficients; only this evaluator-side check reads them."""

    def __init__(self, config: ForcedDynamics.Config) -> None:
        super().__init__(config)
        self.used: list[Tensor] = []

    @override
    def advance(self, state: Tensor, coefficients: Tensor) -> Tensor:
        self.used.append(coefficients)
        return super().advance(state, coefficients)


class _ForcedRecording(ForcedDynamics):
    """Part four's dynamics with the same coefficient log."""

    def __init__(self, config: ForcedDynamics.Config) -> None:
        super().__init__(config)
        self.used: list[Tensor] = []

    @override
    def advance(self, state: Tensor, coefficients: Tensor) -> Tensor:
        self.used.append(coefficients)
        return super().advance(state, coefficients)


def calibrate(
    data: ForcedData,
    *,
    episodes: int,
    restarts: int,
    iterations: int,
    seed: int,
    near_truth: bool = False,
) -> list[EpisodeCalibration]:
    """Fit each episode jointly; ``near_truth`` starts at the truth with 1% jitter."""
    cfg = data.config
    steps = cfg.seq_len
    dynamics = data.dynamics
    chained = isinstance(dynamics, ChainedDynamics)
    recorder = (_Recording if chained else _ForcedRecording)(dynamics.config)
    data.dynamics = recorder
    try:
        episode = data.sample(count=episodes, seed=seed)
    finally:
        data.dynamics = dynamics
    used = recorder.used
    coef = torch.stack(used[: steps - 1], 1)
    change = torch.cat(
        (
            torch.zeros(episodes, 1, dtype=torch.bool),
            (coef[:, 1:, :6] != coef[:, :-1, :6]).any(-1),
        ),
        1,
    )
    regime = change.long().cumsum(1)
    count = int(regime.max()) + 1
    method = ChainedShooting if chained else ForcedShooting
    shooting = method.Config()
    persistent = data.coefficients - 6
    lo = torch.tensor(shooting.lower, dtype=torch.float64)
    hi = torch.tensor(shooting.upper, dtype=torch.float64)
    readings = episode.windows.media[..., 0]
    live = (readings != readings[:, -1:]).flip(1).cumsum(1).flip(1) > 0
    live[torch.arange(episodes), live.sum(1)] = True
    batch = episodes * restarts
    y = readings.repeat_interleave(restarts, 0)
    mask = live.repeat_interleave(restarts, 0)
    reg = regime.repeat_interleave(restarts, 0)
    generator = torch.Generator().manual_seed(5)

    def uniform(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, dtype=torch.float64)

    if near_truth:
        # Coefficients are constant within a regime, so duplicate writes agree.
        filled = coef[:, -1:, :6].repeat(1, count, 1)
        filled.scatter_(1, regime[..., None].expand(-1, -1, 6), coef[..., :6])

        def jitter(v: Tensor) -> Tensor:
            noise = torch.randn(v.shape, generator=generator, dtype=torch.float64)
            return v * (1 + 0.01 * noise)

        def unit(v: Tensor, a: Tensor, b: Tensor) -> Tensor:
            return torch.logit(((v - a) / (b - a)).clamp(1e-4, 1 - 1e-4))

        obs_raw = unit(jitter(filled.repeat_interleave(restarts, 0)), lo[:6], hi[:6])
        per_raw = unit(
            jitter(coef[:, -1, 6:].repeat_interleave(restarts, 0)),
            lo[6:],
            hi[6:],
        )
        init = jitter(episode.states[:, 0].repeat_interleave(restarts, 0))
    else:
        obs_raw = torch.logit(uniform(batch, count, 6).clamp(0.02, 0.98))
        per_raw = torch.logit(uniform(batch, persistent).clamp(0.02, 0.98))
        init = torch.cat(
            (
                y[:, :1],
                (y[:, 2:3] - y[:, :1]) / (2 * data.dynamics.config.step_size),
                method.latent(uniform(batch, method.latent_draws)),
            ),
            1,
        )
    params = [obs_raw.requires_grad_(), per_raw.requires_grad_(), init.requires_grad_()]

    def rollout() -> tuple[Tensor, Tensor, Tensor]:
        obs = lo[:6] + (hi[:6] - lo[:6]) * torch.sigmoid(obs_raw)
        per = lo[6:] + (hi[6:] - lo[6:]) * torch.sigmoid(per_raw)
        state, path = init, [init[:, 0]]
        current = per
        for t in range(steps - 1):
            row = obs.gather(1, reg[:, t, None, None].expand(-1, 1, 6))[:, 0]
            current = torch.cat((row, per), 1)
            state = data.dynamics.advance(state, current)
            path.append(state[:, 0])
        return torch.stack(path, 1), state, current

    def robust(path: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        error = (path - target).div(shooting.robust).square().log1p()
        return (error * weight).sum(1) / weight.sum(1)

    optimizer = torch.optim.Adam(params, lr=0.03)
    for _ in range(iterations):
        optimizer.zero_grad()
        robust(rollout()[0], y, mask).sum().backward()
        optimizer.step()
    with torch.no_grad():
        path, state, last = rollout()
        loss = robust(path, y, mask).nan_to_num(torch.inf).view(episodes, restarts)
        truth = robust(episode.states[:, :steps, 0], readings, live)
        best = torch.arange(episodes) * restarts + loss.argmin(1)
        future = data.dynamics.forecast(
            state[best],
            coefficients=last[best],
            steps=cfg.pred_len,
        )[..., 0]
        target = episode.states[:, steps:, 0]
        skill = 1 - (future - target).square().mean(1) / target.square().mean(1)
    return [
        EpisodeCalibration(
            regimes=int(regime[e].max()) + 1,
            truth_loss=float(truth[e]),
            best_loss=float(loss[e].min()),
            skill=float(skill[e]),
        )
        for e in range(episodes)
    ]


class _Flags(argparse.Namespace):
    episodes: int
    restarts: int
    iterations: int
    seed: int
    near_truth: bool
    part: str


def main(argv: list[str] | None = None) -> int:
    """Print one line per development episode of exp051's or exp053's generator."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--near-truth", action="store_true")
    parser.add_argument("--part", choices=("four", "five"), default="five")
    flags = cast(_Flags, parser.parse_args(argv))
    torch.set_num_threads(1)
    results = calibrate(
        (ChainedData if flags.part == "five" else ForcedData).Config().make(),
        episodes=flags.episodes,
        restarts=flags.restarts,
        iterations=flags.iterations,
        seed=flags.seed,
        near_truth=flags.near_truth,
    )
    for index, row in enumerate(results):
        print(
            f"episode {index}: regimes={row.regimes} truth_loss={row.truth_loss:.4f}"
            f" best_loss={row.best_loss:.4f} skill={row.skill:.3f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
