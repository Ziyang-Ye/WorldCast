"""Package exports that import their module on first use (PEP 562)."""

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["lazy_exports"]


def lazy_exports(
    package: str, exports: Mapping[str, Sequence[str]]
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    """The ``__getattr__``, ``__dir__`` and ``__all__`` of a package whose names live in its
    submodules.

    Args:
        package (str): the package's ``__name__``.
        exports (Mapping[str, Sequence[str]]): submodule (relative to the package) -> the names it
            exports; a name is exported from one submodule.

    Returns:
        tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]: ``__getattr__``, which
        imports the submodule of a name when the name is first read; ``__dir__``, which lists the
        exported names whether they have been read or not; and ``__all__``.
    """
    submodule_of: dict[str, str] = {}
    for module, names in exports.items():
        for name in names:
            if name in submodule_of:
                raise ValueError(
                    f"{package} exports {name!r} from both {submodule_of[name]} and {module}"
                )
            submodule_of[name] = module
    exported = sorted(submodule_of)

    def __getattr__(name: str) -> Any:
        if name not in submodule_of:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        return getattr(importlib.import_module(f"{package}.{submodule_of[name]}"), name)

    def __dir__() -> list[str]:
        return list(exported)

    return __getattr__, __dir__, exported
