"""The rows the player state field read: the source codes the page and the demo's clients share."""

from worldcast.engine.inference.field_rows import FieldRows


def test_the_sources_of_the_rows_are_the_wire_values():
    assert (FieldRows.OWN, FieldRows.LIVE, FieldRows.RECORDED, FieldRows.HIDDEN) == (0, 1, 2, 3)
