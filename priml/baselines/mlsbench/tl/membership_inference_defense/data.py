"""The training split cut in half into members and non-members, as the reference cuts it.

The reference shuffles the training split's indices with Python's generator,
seeded with its ``--seed``, and trains on the first half: the members. The
second half, the non-members, is never trained on. After training, the
membership attack scores the model's confidence on both halves, so the
evaluation pass runs over three splits in order: the test split, the members
and the non-members.

The members are the training loader's whole split, so the vision task's
shuffle and augmentation streams apply to them unchanged. Augmentation is off
in the reference's recipe; :func:`normalize_only` is the slot's value then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

import random

from configgle import Makes

import torch

from priml.baselines.mlsbench.dl.vision.data import (
    Cifar100Data,
    EvalLoader,
    normalize,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor


SPLITS: Final = ("test", "member", "nonmember")
"""The evaluation pass's splits, in the order the reference scores them."""


class MembershipData(Cifar100Data):
    """A dataset that trains on the members and evaluates the test split and both halves."""

    class Config(Makes["MembershipData"], Cifar100Data.Config):
        """The prepared tensors, the loader, and the seed of the member split."""

        split_seed: int = 42
        """Seed of the Python generator that shuffles the training split's indices."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.member_index, self.nonmember_index = member_split(
            len(self.train_label),
            seed=config.split_seed,
        )
        """``(N // 2,)`` and ``(N - N // 2,)`` positions in the training split."""
        media, label = self.train_media, self.train_label
        self.nonmember_media = media[self.nonmember_index]
        self.nonmember_label = label[self.nonmember_index]
        self.train_media = media[self.member_index]
        self.train_label = label[self.member_index]

    @override
    def eval_dataloader(self) -> MembershipEvalLoader:
        """Return the re-iterable evaluation loader over the three splits.

        Returns:
          loader: Each iteration is one sequential pass over each of
            :data:`SPLITS`, every batch tagged with its ``split`` index.

        """
        return MembershipEvalLoader(
            (self.eval_media, self.eval_label),
            member=(self.train_media, self.train_label),
            nonmember=(self.nonmember_media, self.nonmember_label),
            config=self.config,
        )


class MembershipEvalLoader(EvalLoader):
    """The test split, then the members, then the non-members, each a sequential pass.

    Each pass draws its own unused seed from the global generator, as each of
    the reference's three loaders does.

    Args:
      test: ``(N, C, H, W)`` uint8 test images and their ``(N,)`` labels.
      member: The members, likewise.
      nonmember: The non-members, likewise.
      config: The dataset; its ``eval_batch_size``, ``mean`` and ``std``.

    """

    def __init__(
        self,
        test: tuple[Tensor, Tensor],
        *,
        member: tuple[Tensor, Tensor],
        nonmember: tuple[Tensor, Tensor],
        config: Cifar100Data.Config,
    ) -> None:
        super().__init__(*test, config)
        self.member = EvalLoader(*member, config)
        self.nonmember = EvalLoader(*nonmember, config)

    @override
    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield ``media``, ``label`` and ``split`` batches, one split after another.

        ``split`` is a 0-d int64 index into :data:`SPLITS`.
        """
        passes = (super().__iter__(), iter(self.member), iter(self.nonmember))
        for index, batches in enumerate(passes):
            for batch in batches:
                yield batch | {"split": torch.tensor(index)}

    @override
    def __len__(self) -> int:
        """Return the batches of all three passes."""
        return super().__len__() + len(self.member) + len(self.nonmember)


def member_split(count: int, *, seed: int) -> tuple[Tensor, Tensor]:
    """Shuffle ``range(count)`` with Python's generator and cut it in half.

    The process-wide generator is seeded, used and restored, so it draws what
    the reference's ``random.Random(seed)`` draws and leaves the caller's
    stream where it was.

    Args:
      count: Images in the training split.
      seed: Seed of the shuffle.

    Returns:
      member: The first ``count // 2`` shuffled indices, int64.
      nonmember: The rest, in shuffled order.

    """
    indices = list(range(count))
    state = random.getstate()
    random.seed(seed)
    try:
        random.shuffle(indices)
    finally:
        random.setstate(state)
    order = torch.tensor(indices, dtype=torch.int64)
    return order[: count // 2], order[count // 2 :]


def normalize_only(
    images: Tensor,
    stream: torch.Generator,
    /,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Normalize training images without augmenting them: the reference's default.

    Args:
      images: ``(B, C, H, W)`` uint8 pixels.
      stream: Unused; no draw is taken.
      pad: Unused.
      mean: ``(3,)`` float32 channel means.
      std: ``(3,)`` float32 channel standard deviations.

    Returns:
      media: ``(B, C, H, W)`` float32 images.

    """
    del stream, pad
    return normalize(images, mean=mean, std=std)
