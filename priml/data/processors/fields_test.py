"""Tests for field manipulation processors."""

from __future__ import annotations

from typing import cast

from priml.data.processors.fields import (
    CopyField,
    FieldRenameKeys,
    FieldSetValues,
    Inline,
)


# ============================================================================
# CopyField Tests
# ============================================================================.


def test_copy_field_duplicates_sources_and_keeps_them_in_place():
    copier = CopyField.Config(
        mappings={"media_tensor": "media", "caption": "text"},
    ).make()

    samples = [dict[str, object]({"media_tensor": "t", "caption": "c", "keep": 1})]

    results = list(copier(iter(samples)))

    assert results[0] is samples[0]
    assert results[0] == {
        "media_tensor": "t",
        "caption": "c",
        "keep": 1,
        "media": "t",
        "text": "c",
    }


def test_copy_field_writes_none_for_a_missing_source():
    copier = CopyField.Config(mappings={"absent": "target"}).make()

    results = list(copier(iter([dict[str, object]({"id": 1})])))

    assert results[0] == {"id": 1, "target": None}


def test_copy_field_without_mappings_passes_samples_through():
    copier = CopyField.Config().make()

    samples = [dict[str, object]({"a": 1}), dict[str, object]({"b": 2})]

    results = list(copier(iter(samples)))

    assert results == samples
    assert results[0] is samples[0]


# ============================================================================
# FieldSetValues Tests
# ============================================================================.


def test_field_set_values_basic():
    """Test setting field values in samples."""
    setter = FieldSetValues.Config(
        values={"target_height": 224, "target_width": 224, "batch_size": 32},
    ).make()

    samples = [
        dict[str, object]({"key": "sample1", "existing": "value"}),
        dict[str, object]({"key": "sample2"}),
    ]

    results = list(setter(iter(samples)))

    assert len(results) == 2
    assert results[0]["key"] == "sample1"
    assert results[0]["existing"] == "value"
    assert results[0]["target_height"] == 224
    assert results[0]["target_width"] == 224
    assert results[0]["batch_size"] == 32

    assert results[1]["key"] == "sample2"
    assert results[1]["target_height"] == 224
    assert results[1]["target_width"] == 224
    assert results[1]["batch_size"] == 32


def test_field_set_values_overwrite_existing():
    """Test that set values overwrite existing fields."""
    setter = FieldSetValues.Config(values={"field": "new_value"}).make()

    samples = [dict[str, object]({"field": "old_value", "other": "keep"})]

    results = list(setter(iter(samples)))

    assert results[0]["field"] == "new_value"
    assert results[0]["other"] == "keep"


def test_field_set_values_empty():
    """Test that empty values configuration passes through unchanged."""
    setter = FieldSetValues.Config().make()

    samples = [
        dict[str, object]({"field1": "value1", "field2": "value2"}),
        dict[str, object]({"field3": "value3"}),
    ]

    results = list(setter(iter(samples)))

    assert results == samples


def test_field_set_values_none_value():
    """Test setting None as a field value."""
    setter = FieldSetValues.Config(values={"field": None}).make()

    samples = [dict[str, object]({"existing": "value"})]

    results = list(setter(iter(samples)))

    assert results[0]["field"] is None
    assert results[0]["existing"] == "value"


def test_field_set_values_complex_types():
    """Test setting complex types as field values."""
    complex_value = {"nested": [1, 2, 3], "dict": {"a": "b"}}
    setter = FieldSetValues.Config(
        values={
            "list_field": [1, 2, 3],
            "dict_field": complex_value,
            "tuple_field": (4, 5, 6),
        },
    ).make()

    samples = [dict[str, object]({"key": "sample"})]

    results = list(setter(iter(samples)))

    assert results[0]["list_field"] == [1, 2, 3]
    assert results[0]["dict_field"] == complex_value
    assert results[0]["tuple_field"] == (4, 5, 6)


def test_field_set_values_multiple_samples():
    """Test that values are set consistently across multiple samples."""
    setter = FieldSetValues.Config(values={"constant": 42}).make()

    samples = [dict[str, object]({"id": i}) for i in range(10)]

    results = list(setter(iter(samples)))

    assert len(results) == 10
    for i, result in enumerate(results):
        assert result["id"] == i
        assert result["constant"] == 42


def test_field_set_values_preserves_original():
    """Test that original sample is not modified."""
    setter = FieldSetValues.Config(values={"new_field": "value"}).make()

    original = dict[str, object]({"existing": "original"})
    samples = [original]

    results = list(setter(iter(samples)))

    # Original should not have the new field.
    assert "new_field" not in original
    assert results[0]["new_field"] == "value"
    assert results[0]["existing"] == "original"


# ============================================================================
# FieldRenameKeys Tests
# ============================================================================.


def test_field_rename_keys_basic():
    """Test renaming fields."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "original_width": "width",
            "original_height": "height",
        },
    ).make()

    samples = [
        dict[str, object](
            {"original_width": 640, "original_height": 480, "keep": "value"},
        ),
    ]

    results = list(remapper(iter(samples)))

    assert results[0]["width"] == 640
    assert results[0]["height"] == 480
    assert results[0]["keep"] == "value"
    # Original fields should still be present (remap copies, not moves)
    assert "original_width" in results[0]


def test_field_rename_keys_delete():
    """Test deleting fields by mapping to None."""
    remapper = FieldRenameKeys.Config(
        mappings={"_tar_handle": None, "temp_field": None},
    ).make()

    samples = [
        dict[str, object](
            {"_tar_handle": "handle", "temp_field": 123, "keep": "value"},
        ),
    ]

    results = list(remapper(iter(samples)))

    assert "_tar_handle" not in results[0]
    assert "temp_field" not in results[0]
    assert results[0]["keep"] == "value"


def test_field_rename_keys_no_mappings():
    """Test remapper with no mappings passes through unchanged."""
    remapper = FieldRenameKeys.Config().make()

    samples = [dict[str, object]({"field1": 1, "field2": 2})]

    results = list(remapper(iter(samples)))

    assert results[0] == samples[0]


def test_field_rename_keys_missing_source_field():
    """Test that missing source fields are silently skipped."""
    remapper = FieldRenameKeys.Config(mappings={"nonexistent": "target"}).make()

    samples = [dict[str, object]({"real_field": "value"})]

    results = list(remapper(iter(samples)))

    assert "target" not in results[0]
    assert results[0]["real_field"] == "value"


def test_field_rename_keys_overwrite_existing():
    """Test that remapper can overwrite existing target fields."""
    remapper = FieldRenameKeys.Config(mappings={"source": "target"}).make()

    samples = [dict[str, object]({"source": "new_value", "target": "old_value"})]

    results = list(remapper(iter(samples)))

    # Target should be overwritten with source value.
    assert results[0]["target"] == "new_value"


def test_field_rename_keys_glob_delete_all_unmapped():
    """Test using '*': None to delete all unmapped fields."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "media_tensor": "media",
            "caption": "caption",
            "*": None,
        },
    ).make()

    samples = [
        dict[str, object](
            {
                "media_tensor": "tensor_data",
                "caption": "description",
                "unwanted1": "delete_me",
                "unwanted2": 123,
                "_tar_handle": "handle",
            },
        ),
    ]

    results = list(remapper(iter(samples)))

    # Only explicitly mapped fields should remain.
    assert results[0] == {"media": "tensor_data", "caption": "description"}


def test_field_rename_keys_glob_with_none_mapping():
    """Test glob pattern with explicit None mapping."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "keep_field": "renamed_field",
            "explicit_delete": None,
            "*": None,
        },
    ).make()

    samples = [
        dict[str, object](
            {
                "keep_field": "keep_me",
                "explicit_delete": "delete_me",
                "also_delete": "gone",
            },
        ),
    ]

    results = list(remapper(iter(samples)))

    # Only keep_field (renamed) should remain; explicit_delete mapped to None.
    assert results[0] == {"renamed_field": "keep_me"}


def test_field_rename_keys_glob_missing_source():
    """Test glob pattern when mapped source field doesn't exist."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "nonexistent": "target",
            "*": None,
        },
    ).make()

    samples = [dict[str, object]({"field1": "value1", "field2": "value2"})]

    results = list(remapper(iter(samples)))

    # Nonexistent field isn't in sample, so result is empty.
    assert results[0] == {}


def test_field_rename_keys_rename_same_name():
    """Test renaming a field to itself."""
    remapper = FieldRenameKeys.Config(mappings={"field": "field"}).make()

    samples = [dict[str, object]({"field": "value", "other": "data"})]

    results = list(remapper(iter(samples)))

    assert results[0]["field"] == "value"
    assert results[0]["other"] == "data"


def test_field_rename_keys_multiple_sources_same_target():
    """Test multiple source fields mapping to the same target (last wins)."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "source1": "target",
            "source2": "target",
        },
    ).make()

    samples = [
        dict[str, object](
            {"source1": "first", "source2": "second", "other": "keep"},
        ),
    ]

    results = list(remapper(iter(samples)))

    # Both mappings are applied; the order depends on dict iteration
    # In practice, one will overwrite the other.
    assert results[0]["target"] in ["first", "second"]
    assert results[0]["other"] == "keep"


def test_field_rename_keys_complex_types():
    """Test renaming fields with complex types."""
    remapper = FieldRenameKeys.Config(
        mappings={"old_list": "new_list", "old_dict": "new_dict"},
    ).make()

    samples = [
        dict[str, object](
            {
                "old_list": [1, 2, 3],
                "old_dict": {"nested": {"deep": "value"}},
                "keep": "data",
            },
        ),
    ]

    results = list(remapper(iter(samples)))

    assert results[0]["new_list"] == [1, 2, 3]
    assert results[0]["new_dict"] == {"nested": {"deep": "value"}}
    assert results[0]["keep"] == "data"


def test_field_rename_keys_chain_rename():
    """Test that renaming doesn't chain (a->b, b->c doesn't make a->c)."""
    remapper = FieldRenameKeys.Config(mappings={"a": "b", "b": "c"}).make()

    samples = [dict[str, object]({"a": "value_a", "b": "value_b"})]

    results = list(remapper(iter(samples)))

    # a->b creates "b", b->c creates "c"
    # Original b gets mapped to c, new b (from a) also exists.
    assert results[0]["a"] == "value_a"  # Original a still exists.
    assert results[0]["b"] == "value_a"  # A was copied to b.
    assert results[0]["c"] == "value_b"  # Original b was copied to c.


def test_field_rename_keys_preserves_original():
    """Test that original sample is not modified."""
    remapper = FieldRenameKeys.Config(mappings={"old": "new"}).make()

    original = dict[str, object]({"old": "value", "other": "data"})
    samples = [original]

    results = list(remapper(iter(samples)))

    # Original should not have the new field.
    assert "new" not in original
    assert results[0]["new"] == "value"
    assert results[0]["old"] == "value"


def test_field_rename_keys_empty_sample():
    """Test renaming with empty sample."""
    remapper = FieldRenameKeys.Config(mappings={"field": "renamed"}).make()

    samples = [dict[str, object]({})]

    results = list(remapper(iter(samples)))

    assert results[0] == {}


def test_field_rename_keys_glob_empty_sample():
    """Test glob pattern with empty sample."""
    remapper = FieldRenameKeys.Config(mappings={"*": None}).make()

    samples = [dict[str, object]({})]

    results = list(remapper(iter(samples)))

    assert results[0] == {}


def test_field_rename_keys_multiple_samples():
    """Test renaming across multiple samples."""
    remapper = FieldRenameKeys.Config(mappings={"old_name": "new_name"}).make()

    samples = [dict[str, object]({"old_name": f"value_{i}", "id": i}) for i in range(5)]

    results = list(remapper(iter(samples)))

    assert len(results) == 5
    for i, result in enumerate(results):
        assert result["new_name"] == f"value_{i}"
        assert result["id"] == i


def test_field_rename_keys_delete_multiple_fields():
    """Test deleting multiple fields."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "delete1": None,
            "delete2": None,
            "delete3": None,
        },
    ).make()

    samples = [
        dict[str, object](
            {
                "delete1": "a",
                "delete2": "b",
                "delete3": "c",
                "keep1": "x",
                "keep2": "y",
            },
        ),
    ]

    results = list(remapper(iter(samples)))

    assert results[0] == {"keep1": "x", "keep2": "y"}


# ============================================================================
# Inline Tests
# ============================================================================.


def test_inline_whole_sample():
    """Test Inline with no field_name (transforms whole sample)."""

    def add_field(sample: object) -> object:
        assert isinstance(sample, dict)
        sample_dict = cast(dict[str, object], sample)
        result = dict[str, object](sample_dict)
        value: object = sample_dict.get("value", 0)
        assert isinstance(value, int)
        result["computed"] = value * 2
        return result

    inline = Inline.Config(transform=add_field).make()

    samples = [
        dict[str, object]({"id": 1, "value": 10}),
        dict[str, object]({"id": 2, "value": 20}),
    ]

    results = list(inline(iter(samples)))

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["value"] == 10
    assert results[0]["computed"] == 20
    assert results[1]["id"] == 2
    assert results[1]["value"] == 20
    assert results[1]["computed"] == 40


def test_inline_specific_field():
    """Test Inline with field_name (transforms specific field)."""

    def double(value: object) -> object:
        assert isinstance(value, int)
        return value * 2

    inline = Inline.Config(transform=double, field_name="value").make()

    samples = [
        dict[str, object]({"id": 1, "value": 10, "other": "keep"}),
        dict[str, object]({"id": 2, "value": 20, "other": "keep"}),
    ]

    results = list(inline(iter(samples)))

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["value"] == 20
    assert results[0]["other"] == "keep"
    assert results[1]["id"] == 2
    assert results[1]["value"] == 40
    assert results[1]["other"] == "keep"


def test_inline_field_missing():
    """Test Inline with field_name when field is missing (calls transform with None)."""

    def double_or_zero(value: object) -> object:
        if value is None:
            return 0
        assert isinstance(value, int)
        return value * 2

    inline = Inline.Config(transform=double_or_zero, field_name="missing_field").make()

    samples = [
        dict[str, object]({"id": 1, "value": 10}),
        dict[str, object]({"id": 2, "value": 20}),
    ]

    results = list(inline(iter(samples)))

    # Transform should be called with None and result should be set.
    assert results[0]["id"] == 1
    assert results[0]["value"] == 10
    assert results[0]["missing_field"] == 0
    assert results[1]["id"] == 2
    assert results[1]["value"] == 20
    assert results[1]["missing_field"] == 0


def test_inline_no_transform():
    """Test Inline with no transform (passes through unchanged)."""
    inline = Inline.Config().make()

    samples = [
        dict[str, object]({"field1": "value1"}),
        dict[str, object]({"field2": "value2"}),
    ]

    results = list(inline(iter(samples)))

    assert results == samples


def test_inline_field_complex_type():
    """Test Inline transforming complex field types."""

    def append_to_list(lst: object) -> object:
        assert isinstance(lst, list)
        values: list[int] = []
        for value in cast(list[object], lst):
            assert isinstance(value, int)
            values.append(value)
        return [*values, 999]

    inline = Inline.Config(transform=append_to_list, field_name="items").make()

    samples = [dict[str, object]({"id": 1, "items": [1, 2, 3]})]

    results = list(inline(iter(samples)))

    assert results[0]["id"] == 1
    assert results[0]["items"] == [1, 2, 3, 999]


def test_inline_modifies_inplace():
    """Test that Inline modifies sample in-place when using field_name."""

    def modify_value(value: object) -> object:
        assert isinstance(value, int)
        return value + 10

    inline = Inline.Config(transform=modify_value, field_name="value").make()

    original = dict[str, object]({"value": 5, "other": "keep"})
    samples = [original]

    results = list(inline(iter(samples)))

    # Original is modified in-place.
    assert original["value"] == 15
    assert results[0] is original
    assert results[0]["value"] == 15
    assert results[0]["other"] == "keep"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
