"""The config loader: merging, ``inherit``, dotted overrides, typed building and what it refuses."""

from dataclasses import dataclass, field

import pytest

from worldcast.config.loader import (
    build,
    check_at_least,
    check_choice,
    config_to_dict,
    load,
    merge,
    parse_overrides,
    read_yaml,
    require_set,
    set_dotted,
)


@dataclass(frozen=True)
class Inner:
    rate: float = 1.0
    names: tuple[str, ...] = ("a",)


@dataclass(frozen=True)
class Outer:
    inner: Inner = field(default_factory=Inner)
    path: str | None = None
    table: dict[str, int] = field(default_factory=dict)


def outer(data: dict) -> Outer:
    return build(Outer, data)


def test_merge_is_recursive_and_copies():
    base = {"a": {"b": 1, "c": [1]}, "d": 2}
    out = merge(base, {"a": {"b": 3}, "e": 4})
    assert out == {"a": {"b": 3, "c": [1]}, "d": 2, "e": 4}
    out["a"]["c"].append(2)
    assert base["a"]["c"] == [1]


def test_files_merge_in_order_with_inherit(tmp_path):
    (tmp_path / "base.yaml").write_text("inner: {rate: 2.0, names: [x, y]}\npath: base\n")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\npath: child\n")
    (tmp_path / "local.yaml").write_text("inner: {rate: 3}\n")
    assert read_yaml(tmp_path / "child.yaml") == {
        "inner": {"rate": 2.0, "names": ["x", "y"]},
        "path": "child",
    }
    cfg = load([tmp_path / "child.yaml", tmp_path / "local.yaml"], {"table": {"k": 1}}, outer)
    assert cfg == Outer(inner=Inner(rate=3.0, names=("x", "y")), path="child", table={"k": 1})
    assert build(Outer, config_to_dict(cfg)) == cfg


def test_a_number_in_scientific_notation_is_a_number_in_a_file_and_in_an_override(tmp_path):
    (tmp_path / "rate.yaml").write_text("inner: {rate: 1e-5}\n")
    assert read_yaml(tmp_path / "rate.yaml") == {"inner": {"rate": 1e-5}}
    assert load(tmp_path / "rate.yaml", None, outer).inner.rate == 1e-5
    assert parse_overrides(["inner.rate=1e-5", "inner.rate2=1.5e3", "path=1e5x"]) == {
        "inner.rate": 1e-5,
        "inner.rate2": 1500.0,
        "path": "1e5x",
    }


def test_a_key_written_twice_in_a_file_is_refused(tmp_path):
    (tmp_path / "twice.yaml").write_text("inner: {rate: 2}\npath: a\ninner: {names: [x]}\n")
    with pytest.raises(ValueError, match=r"twice.yaml.*keys written twice: \['inner'\]"):
        read_yaml(tmp_path / "twice.yaml")


def test_overrides_are_set_in_the_mapping_before_it_is_built(tmp_path):
    assert parse_overrides(["inner.rate=2", "path=null", "inner.names=[p, q]"]) == {
        "inner.rate": 2,
        "path": None,
        "inner.names": ["p", "q"],
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_overrides(["inner.rate"])
    data = {"inner": {"rate": 2.0}, "table": {"k": 1}}
    assert set_dotted(data, {"inner.rate": 3, "path": "x", "table.m": 2}) == {
        "inner": {"rate": 3},
        "path": "x",
        "table": {"k": 1, "m": 2},
    }
    assert set_dotted(data, {"table": {"n": 5}})["table"] == {"n": 5}  # a whole mapping replaces
    assert data == {"inner": {"rate": 2.0}, "table": {"k": 1}}  # a copy is set
    (tmp_path / "base.yaml").write_text("inner: {rate: 2.0}\n")
    cfg = load(tmp_path / "base.yaml", {"inner.rate": 3, "path": 7, "table.m": 2}, outer)
    assert cfg == Outer(inner=Inner(rate=3.0), path="7", table={"m": 2})
    assert isinstance(cfg.inner.rate, float)  # an integer for a number, a number for a string


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"inner.speed": 1}, r"unknown config keys in inner: \['speed'\]"),
        ({"outer.rate": 1}, r"unknown config keys in <root>: \['outer'\]"),
        ({"path.x": 1}, "'path' is a value, not a section"),
        ({"table.k": "one"}, "table.k must be an integer, got 'one'"),
    ],
)
def test_an_override_of_an_unknown_key_is_refused_like_a_files(tmp_path, overrides, message):
    (tmp_path / "base.yaml").write_text("path: base\n")
    with pytest.raises(ValueError, match=message):
        load(tmp_path / "base.yaml", overrides, outer)


@pytest.mark.parametrize(
    "data",
    [{"unknown": 1}, {"inner": {"rate": "fast"}}, {"inner": {"names": "a"}}, {"path": 3.5}],
)
def test_build_refuses(data):
    with pytest.raises(ValueError):
        build(Outer, data)


def test_the_checks_name_the_key():
    check_choice("optim.clip", "global", ("per_group", "global"))
    with pytest.raises(ValueError, match=r"optim.clip must be one of \('a', 'b'\), got 'c'"):
        check_choice("optim.clip", "c", ("a", "b"))
    check_at_least("run.max_steps", 1, 1)
    with pytest.raises(ValueError, match="run.max_steps must be >= 1, got 0"):
        check_at_least("run.max_steps", 0, 1)
    require_set(Outer(path="x"), "outer", ["path"])
    with pytest.raises(ValueError, match="not set: outer.path, outer.table"):
        require_set(Outer(), "outer", ["path", "table"])
