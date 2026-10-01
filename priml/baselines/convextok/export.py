"""Export a ConvexTok vocabulary as the nanochat baseline's byte-level tokenizer.

Upstream scores every piece -1 in a Unigram model, so encoding picks the segmentation
with the fewest pieces; the exact score keeps path sums integral, so equal-length
segmentations tie exactly as upstream's do. The layout is the baseline's: the 256
bytes in byte order, then the learned pieces, with no special tokens -- the loader
appends reserved IDs after them.
"""

from collections.abc import Sequence

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from priml.baselines.nanochat.scripts.prepare_tokenizer import byte_alphabet


def export_tokenizer(learned: Sequence[str], *, split_pattern: str) -> Tokenizer:
    """Build the fewest-pieces byte Unigram tokenizer over ``learned``.

    Args:
      learned: Learned pieces of two or more bytes, in ByteLevel characters.
      split_pattern: Pretokenization expression the pieces were learned under.

    Returns:
      tokenizer: Byte-complete Unigram tokenizer.

    Raises:
      ValueError: If a learned piece is a single byte, which the alphabet holds.

    """
    alphabet = byte_alphabet()
    bytes_in_order = [alphabet[value] for value in range(256)]
    if any(len(piece) < 2 for piece in learned):
        raise ValueError(
            "Learned pieces span two or more bytes; single bytes are built in.",
        )
    tokenizer = Tokenizer(
        models.Unigram(
            [(piece, -1.0) for piece in [*bytes_in_order, *learned]],
            unk_id=None,
            byte_fallback=False,
        ),
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(split_pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ],
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer
