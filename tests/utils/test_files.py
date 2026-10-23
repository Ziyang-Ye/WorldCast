"""Files: a write leaves a whole file or none; the JSON lines of the data files."""

import math

import pytest

from worldcast.utils.files import atomic_write, json_line, sha256_file, write_jsonl


def test_a_write_leaves_the_whole_file_or_the_one_before(tmp_path):
    path = tmp_path / "made" / "here.txt"
    atomic_write(path, lambda handle: handle.write(b"first"))
    assert path.read_bytes() == b"first"

    def fails(handle):
        handle.write(b"half")
        raise OSError("no space left on device")

    with pytest.raises(OSError, match="no space"):
        atomic_write(path, fails)
    assert path.read_bytes() == b"first"
    assert [p.name for p in path.parent.iterdir()] == ["here.txt"]  # no temporary file is left


def test_json_lines_are_sorted_compact_ascii_and_finite(tmp_path):
    assert json_line({"\u00e9": "\u00e9", "a": 1}) == '{"a":1,"\\u00e9":"\\u00e9"}\n'
    with pytest.raises(ValueError):
        json_line({"a": math.nan})
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, (row for row in [{"b": 2, "a": [1.5]}, {}]))
    assert path.read_text() == '{"a":[1.5],"b":2}\n{}\n'
    assert len(sha256_file(path)) == 64
