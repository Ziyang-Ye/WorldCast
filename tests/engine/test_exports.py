"""The engine's packages export the public names of their modules, and import them lazily."""

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

#: The packages with exports; ``training`` exports the names of its recipes too.
PACKAGES = (
    "worldcast.distributed",
    "worldcast.engine",
    "worldcast.engine.checkpoint",
    "worldcast.engine.evaluation",
    "worldcast.engine.inference",
    "worldcast.engine.optim",
    "worldcast.engine.training",
)


def _modules(package: str) -> list[str]:
    """The modules under ``package``, without those of a package below it that exports its own."""
    below = [p + "." for p in PACKAGES if p.startswith(package + ".")]
    root = importlib.import_module(package)
    modules = pkgutil.walk_packages(root.__path__, package + ".")
    return [m.name for m in modules if not m.ispkg and not m.name.startswith(tuple(below))]


@pytest.mark.parametrize("package", PACKAGES)
def test_a_package_exports_the_public_names_of_its_modules(package):
    exported = set(importlib.import_module(package).__all__)
    public = set()
    for name in _modules(package):
        public |= set(importlib.import_module(name).__all__)
    assert exported == public
    module = importlib.import_module(package)
    assert all(getattr(module, name) is not None for name in exported)


def test_importing_the_packages_loads_no_torch():
    code = (
        "import sys, importlib\n"
        f"[importlib.import_module(package) for package in {PACKAGES!r}]\n"
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules\n"
        "from worldcast.engine.inference import ServingOptions\n"
        "assert ServingOptions().decoder == 'wan' and 'torch' not in sys.modules\n"
    )
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
