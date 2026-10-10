"""Package exports that import their module on first use."""

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from worldcast.utils.lazy import lazy_exports

REPO = Path(__file__).resolve().parents[2]
PACKAGES = (
    "worldcast.utils",
    "worldcast.modeling",
    "worldcast.sampling",
    "worldcast.data",
    "worldcast.player_state",
    "worldcast.scene_state",
    "worldcast.distributed",
    "worldcast.engine",
    "worldcast.engine.inference",
    "worldcast.engine.training",
    "worldcast.engine.evaluation",
    "worldcast.engine.checkpoint",
    "worldcast.engine.optim",
)
#: The packages whose exports are their whole public API: what another area or a tool reads from
#: one of their modules, and every class in an exported signature, is exported.
WHOLE_API = ("worldcast.utils", "worldcast.modeling", "worldcast.sampling")


def test_lazy_exports_resolve_a_name_from_its_submodule():
    getattr_, dir_, names = lazy_exports(
        "worldcast.utils", {"seed": ("set_seed",), "precision": ("enable_tf32", "generator_dtype")}
    )
    assert names == dir_() == ["enable_tf32", "generator_dtype", "set_seed"]
    from worldcast.utils.seed import set_seed

    assert getattr_("set_seed") is set_seed
    with pytest.raises(AttributeError, match="'worldcast.utils' has no attribute 'seed_all'"):
        getattr_("seed_all")


def test_a_name_is_exported_from_one_submodule():
    """A name listed under two submodules would resolve to the later one without a word."""
    with pytest.raises(ValueError, match="exports 'set_seed' from both seed and precision"):
        lazy_exports("worldcast.utils", {"seed": ("set_seed",), "precision": ("set_seed",)})


@pytest.mark.parametrize("package", PACKAGES)
def test_every_exported_name_resolves(package):
    module = importlib.import_module(package)
    assert dir(module) == module.__all__  # the exports are listed before they are read
    for name in module.__all__:
        assert getattr(module, name) is not None, name
        # a name that is also a submodule would resolve to either, depending on the import order
        assert importlib.util.find_spec(f"{package}.{name}") is None, name


def _classes_of(annotation) -> set[type]:
    """The classes an annotation names, through unions, containers and callables."""
    if isinstance(annotation, type):
        return {annotation}
    arguments = typing.get_args(annotation)
    return set().union(*(_classes_of(a) for a in arguments)) if arguments else set()


def _signature_classes(exported) -> set[type]:
    """The classes in the signature of an exported function, or of the fields and the public
    methods (class and static methods included) of an exported class."""
    if inspect.isclass(exported):
        members = [m for name, m in vars(exported).items() if not name.startswith("_")]
        members += [vars(exported).get("__init__"), vars(exported).get("__call__")]
        members = [getattr(m, "__func__", m) for m in members]
        functions = [m for m in members if inspect.isfunction(m)]
        functions += [m.fget for m in members if isinstance(m, property)]
        functions += [exported] if dataclasses.is_dataclass(exported) else []
    else:
        functions = [exported] if inspect.isfunction(exported) else []
    classes = set()
    for function in functions:
        for hint in typing.get_type_hints(function).values():
            classes |= _classes_of(hint)
    return classes


def test_the_signature_classes_of_a_class_include_its_classmethods():
    class Options:
        @classmethod
        def from_path(cls, path: Path) -> bytes: ...

        @staticmethod
        def parse(text: str) -> ast.AST: ...

        @property
        def size(self) -> int: ...

        def _private(self, value: float) -> None: ...

    assert _signature_classes(Options) == {Path, bytes, str, ast.AST, int}


@pytest.mark.parametrize("package", WHOLE_API)
def test_the_classes_in_an_exported_signature_are_exported(package):
    """What an exported name takes or returns can be imported from the packages too."""
    exports = {name: importlib.import_module(name).__all__ for name in WHOLE_API}
    module = importlib.import_module(package)
    for name in module.__all__:
        for cls in _signature_classes(getattr(module, name)):
            home = ".".join(cls.__module__.split(".")[:2])
            if home in exports:
                assert cls.__name__ in exports[home], f"{package}.{name} names {cls.__name__}"


def _is_module(name: str) -> bool:
    path = REPO / name.replace(".", "/")
    return path.is_dir() or path.with_suffix(".py").is_file()


def _names_read(tree: ast.AST, package: str) -> set[str]:
    """The names a module reads from the modules of ``package``: imported from one of them by
    name, or read as an attribute of one it imports."""
    names, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [a for a in node.names if a.name.startswith(package + ".")]
            modules |= {alias.asname for alias in imported if alias.asname}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                if not f"{node.module}.".startswith(package + "."):
                    continue
                if _is_module(f"{node.module}.{alias.name}"):
                    modules.add(alias.asname or alias.name)
                elif node.module != package:
                    names.add(alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names |= {node.attr} if node.value.id in modules else set()
    return names


def test_the_names_a_module_reads_from_a_package():
    source = (
        "import worldcast.sampling.unipc as U\n"
        "from worldcast.sampling import Sampler, window as W\n"
        "from worldcast.sampling.schedulers import draw_noise\n"
        "from worldcast.modeling.controls import BUTTONS\n"
        "solver, gather = U.FlowUniPCSolver(20), W.gather_window\n"
    )
    read = _names_read(ast.parse(source), "worldcast.sampling")
    assert read == {"FlowUniPCSolver", "gather_window", "draw_noise"}


@pytest.mark.parametrize("package", WHOLE_API)
def test_what_the_other_areas_and_the_tools_import_is_exported(package):
    """A name another area reads from a module of the package is part of its public API."""
    inside = REPO / package.replace(".", "/")
    read = set()
    for path in [*(REPO / "worldcast").rglob("*.py"), *(REPO / "tools").rglob("*.py")]:
        if inside not in path.parents:
            read |= _names_read(ast.parse(path.read_text()), package)
    assert read and read <= set(importlib.import_module(package).__all__)


def test_importing_the_packages_imports_no_heavy_dependency():
    """The packages import their modules, and torch with them, on first use of a name."""
    code = (
        "import sys, importlib\n"
        f"[importlib.import_module(package) for package in {PACKAGES!r}]\n"
        "heavy = [m for m in ('torch', 'numpy', 'safetensors', 'flash_attn') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "from worldcast.sampling import Sampler\n"
        "assert 'torch' in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
