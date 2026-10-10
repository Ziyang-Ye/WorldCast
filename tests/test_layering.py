"""The dependency direction of the package: ``config, utils -> data, distributed -> modeling ->
sampling -> player_state, scene_state -> engine, hub``. An area imports only the areas before it."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "worldcast"
#: The areas in dependency order; the areas of one tier do not import each other.
TIERS = (
    ("config", "utils"),
    ("data", "distributed"),
    ("modeling",),
    ("sampling",),
    ("player_state", "scene_state"),
    ("engine", "hub"),
)
#: Area -> the areas it may import: those of the earlier tiers.
ALLOWED = {
    area: {earlier for tier in TIERS[:i] for earlier in tier}
    for i, tier in enumerate(TIERS)
    for area in tier
}


def _imported_areas(source: str, package: list[str]) -> set[str]:
    """The ``worldcast`` areas that the source of a module of ``package`` imports, at module level
    or inside a function."""
    areas = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            # ``from worldcast import engine`` names its area as the imported name
            names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "worldcast" and len(parts) > 1:
                areas.add(parts[1])
    return areas


def test_every_spelling_of_an_import_names_its_area():
    spellings = {
        "import worldcast.engine.generator": {"engine"},
        "import torch, worldcast.hub as hub": {"hub"},
        "from worldcast.engine import generator": {"engine"},
        "from worldcast import engine, hub": {"engine", "hub"},
        "from .. import engine": {"engine"},
        "from ..engine.generator import load_vae": {"engine"},
        "from . import window": {"sampling"},
        "from .window import gather_window": {"sampling"},
        "def f():\n    from worldcast import hub": {"hub"},
        "import torch\nfrom collections.abc import Mapping": set(),
    }
    for source, areas in spellings.items():
        assert _imported_areas(source, ["worldcast", "sampling"]) == areas, source


def test_every_area_has_its_tier():
    areas = {path.stem for path in PACKAGE.iterdir() if path.is_dir() or path.suffix == ".py"}
    assert areas - {"__init__", "__pycache__"} == set(ALLOWED)


@pytest.mark.parametrize("area", ALLOWED)
def test_an_area_imports_only_the_areas_before_it(area):
    package = PACKAGE / area
    modules = sorted(package.rglob("*.py")) if package.is_dir() else [package.with_suffix(".py")]
    assert modules
    for path in modules:
        parents = ["worldcast", *path.relative_to(PACKAGE).parts[:-1]]
        above = _imported_areas(path.read_text(), parents) - ALLOWED[area] - {area}
        assert not above, f"{path.relative_to(PACKAGE)} imports {sorted(above)}"
