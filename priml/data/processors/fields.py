"""Field manipulation processors for data pipelines.

Processors for renaming, setting, and transforming field values in samples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, cast

import logging

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


__all__ = [
    "CopyField",
    "FieldRenameKeys",
    "FieldSetValues",
    "Inline",
]


class CopyField:
    """Copy field values to new fields using a dict mapping.

    Useful for:
    - Duplicating tensors for multiple processing paths
    - Preserving original values before transformation
    - Creating backup copies of fields

    Example:
        copier = CopyField.Config(
            mappings={
                "media_tensor": "media",  # Copy media_tensor to media
                "original_caption": "caption",  # Copy original_caption to caption
            }
        ).make()

        for sample in copier(source):
            # sample["media"] now contains a copy of media_tensor
            # sample["caption"] now contains a copy of original_caption
            # Original fields are preserved
            pass

    """

    class Config(Fig["CopyField"], kw_only=False):
        """Configuration for CopyField."""

        mappings: dict[str, str] = field(default_factory=dict[str, str])
        """Source field to destination field; the source is left in place."""

    Input = dict[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.mappings = config.mappings

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Copy field values to new fields.

        Requires:
          - (source fields in mappings) - fields to copy from

        Adds:
          - (target fields in mappings) - copies of source field values

        """
        if not self.mappings:
            yield from samples
            return

        for sample in samples:
            for source_field, target_field in self.mappings.items():
                sample[target_field] = sample.get(source_field)
            yield sample


class FieldSetValues:
    """Set field values in samples using a dict mapping.

    Useful for:
    - Setting constant values (e.g., resize dimensions, batch size)
    - Injecting configuration values into samples
    - Overriding existing field values

    Example:
        setter = FieldSetValues.Config(
            values={
                "target_height": 224,
                "target_width": 224,
                "batch_size": 32,
            }
        ).make()

        for sample in setter(source):
            # sample["target_height"] = 224
            # sample["target_width"] = 224
            # sample["batch_size"] = 32
            pass

    """

    class Config(Fig["FieldSetValues"], kw_only=False):
        """Configuration for FieldSetValues."""

        values: dict[str, object] | None = None
        """Field names to constant values written onto every sample."""

    Input = dict[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.values = config.values or {}

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Set field values in samples.

        Requires:
          - (any fields) - passes through all fields

        Adds:
          - (configured fields) - set to configured values

        """
        if not self.values:
            # No values configured, pass through unchanged.
            yield from samples
            return

        for sample in samples:
            yield {**sample, **self.values}


class FieldRenameKeys:
    """Rename or delete fields in samples using a dict mapping.

    Useful for:
    - Mapping dataset-specific field names to canonical names
    - Deleting fields by mapping to None (triggers garbage collection)
    - Using "*": None to delete all unmapped fields

    If a source field doesn't exist in a sample, it's skipped. If a target
    field already exists, it's overwritten. If target is None, the source
    field is deleted from the result.

    Special pattern:
    - "*": None - Delete all fields except those explicitly remapped to non-None values

    Example:
        remapper = FieldRenameKeys.Config(
            mappings={
                "re_caption_condition_diverse_topk": "caption",
                "original_width": "width",
                "_tar_handle": None,  # Delete to trigger GC
            }
        ).make()

        for sample in remapper(source):
            # sample["caption"] now contains re_caption_condition_diverse_topk
            # sample["width"] now contains original_width
            # sample no longer has "_tar_handle"
            pass

    Example with glob:
        remapper = FieldRenameKeys.Config(
            mappings={
                "media_tensor": "media",
                "caption": "caption",
                "*": None,  # Delete all other fields
            }
        ).make()

        for sample in remapper(source):
            # sample only has "media" and "caption" fields
            pass

    """

    class Config(Fig["FieldRenameKeys"], kw_only=False):
        """Configuration for FieldRenameKeys."""

        mappings: dict[str, str | None] | None = None
        """Source to destination; ``None`` deletes, and ``"*": None`` drops
        every field not named."""

    Input = dict[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.mappings = config.mappings or {}
        self.has_glob = "*" in self.mappings

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Remap or delete field names in samples.

        Requires:
          - (any fields) - passes through all fields

        Adds:
          - (configured target fields) - remapped from source fields

        Deletes:
          - (source fields mapped to None) - deleted from output
          - (all unmapped fields if "*": None is specified)

        """
        if not self.mappings:
            # No mappings configured, pass through unchanged.
            yield from samples
            return

        for sample in samples:
            if self.has_glob:
                # Build result with only explicitly mapped fields.
                result: dict[str, object] = {}

                # First, apply all explicit mappings (except "*")
                for source_field, target_field in self.mappings.items():
                    if source_field == "*":
                        continue
                    if source_field in sample and target_field is not None:
                        # Keep this field with new name.
                        result[target_field] = sample[source_field]
                    # If target is None, field is explicitly deleted (not added to result)
            else:
                # Start with all fields, then apply mappings.
                result = dict(sample)
                for source_field, target_field in self.mappings.items():
                    if source_field in sample:
                        if target_field is None:
                            # Delete the field (allows GC of the value)
                            del result[source_field]
                        else:
                            # Rename/copy the field.
                            result[target_field] = sample[source_field]

            yield result


class Inline:
    """Apply an inline transformation function to samples or specific fields.

    Useful for:
    - Simple field transformations without writing a full processor
    - Squeezing/unsqueezing tensor dimensions
    - Quick data manipulation in pipelines

    Modes:
    - field_name=None: Transform receives and returns the whole sample dict
    - field_name="field": Transform receives and returns just the field value

    Example (whole sample):
        # Squeeze batch dimension from media_tensor
        def squeeze_batch_dim(sample):
            if "media_tensor" in sample:
                tensor = sample["media_tensor"]
                if tensor.ndim == 5:  # (B, C, F, H, W)
                    sample["media_tensor"] = tensor.squeeze(2)  # -> (B, C, H, W)
            return sample

        inline = Inline.Config(transform=squeeze_batch_dim).make()

        for sample in inline(source):
            # sample["media_tensor"] has squeezed dimension
            pass

    Example (specific field):
        # Squeeze a specific tensor field
        def squeeze_tensor(tensor):
            if tensor.ndim == 5:  # (B, C, F, H, W)
                return tensor.squeeze(2)  # -> (B, C, H, W)
            return tensor

        inline = Inline.Config(
            transform=squeeze_tensor,
            field_name="media_tensor"
        ).make()

        for sample in inline(source):
            # sample["media_tensor"] has been transformed
            pass

    """

    class Config(Fig["Inline"], kw_only=False):
        """Configuration for Inline."""

        transform: Callable[[object], object] | None = None
        """Applied to one field, or to the whole sample when ``field_name``
        is ``None``."""

        field_name: str | None = None
        """Field handed to ``transform``; ``None`` passes the whole sample."""

    Input = dict[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.transform = config.transform
        self.field_name = config.field_name

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply transformation function to samples or specific fields.

        Requires:
          - (any fields) - passes through all fields
          - (field_name if specified) - field to transform

        Adds/Modifies:
          - (depends on transform function)

        """
        if self.transform is None:
            yield from samples
            return

        if self.field_name is None:
            for sample in samples:
                result = self.transform(sample)
                assert isinstance(result, dict)
                yield cast(dict[str, object], result)
        else:
            for sample in samples:
                # Always call transform, passing None if field is missing.
                field_value = sample.get(self.field_name)
                sample[self.field_name] = self.transform(field_value)
                yield sample
