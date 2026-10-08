"""Check the Craftax slot schema against the design's vocabulary."""

import torch

from priml.baselines.craftax.world_model.schema import (
    CellField,
    FrameSchema,
    action_id,
    craftax_schema,
    done_id,
    number_id,
)


def test_craftax_vocabulary_ranges() -> None:
    schema = craftax_schema()
    assert schema.vocab_size == 461
    assert schema.cell_slots == 99
    assert [field.offset for field in schema.cell_fields] == [
        0,
        64,
        72,
        74,
        90,
        106,
        122,
        138,
    ]
    assert [field.size for field in schema.cell_fields] == [
        64,
        8,
        2,
        16,
        16,
        16,
        16,
        16,
    ]
    assert schema.frame_slots == 150
    assert schema.local_slots == 152


def test_craftax_number_and_action_ids() -> None:
    assert number_id(5) == 160
    assert number_id(180) == 335
    assert number_id(-1) == 154
    assert number_id(260) == 415
    assert action_id(15) == 431
    assert done_id(done=False) == 459
    assert done_id(done=True) == 460
    assert len(craftax_schema().scalar_names) == 51


def test_craftax_local_masks_follow_valid_ranges() -> None:
    schema = craftax_schema()
    allowed = schema.local_allowed()
    assert allowed.shape == (152, 461)
    reward = allowed[0].nonzero().flatten().tolist()
    assert reward == list(range(154, 390))
    assert allowed[1].nonzero().flatten().tolist() == [459, 460]
    floor_slot = 2 + 99 + 48
    assert allowed[floor_slot].nonzero().flatten().tolist() == list(range(155, 164))
    health_slot = 2 + 99 + 22
    assert allowed[health_slot].nonzero().flatten().tolist() == list(range(155, 416))
    cell_slot = 2
    assert not bool(allowed[cell_slot].any())


def test_craftax_cell_index_table_masks_reserved_values() -> None:
    schema = craftax_schema()
    table = schema.cell_index_table()
    assert table.shape == (8, 64)
    block = table[0]
    assert block[:37].tolist() == list(range(37))
    assert bool((block[37:] == schema.vocab_size).all())
    item = table[1]
    assert item[:6].tolist() == list(range(64, 70))
    assert bool((item[6:] == schema.vocab_size).all())
    melee = table[3]
    assert melee[:9].tolist() == list(range(74, 83))


def test_schema_is_cached_and_frozen() -> None:
    assert craftax_schema() is craftax_schema()
    assert isinstance(craftax_schema(), FrameSchema)
    assert isinstance(craftax_schema().cell_fields[0], CellField)
    assert craftax_schema().local_allowed().dtype == torch.bool


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
