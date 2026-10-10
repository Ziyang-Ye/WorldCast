"""The public docs against the code: every link and cited path resolves; every documented tool
flag, config key and cited name exists; clients started by hand get a session's environment; the
weights tables list the files of ``worldcast.hub``; the notice names what the tree holds;
every shipped config loads; and the library example of docs/inference.md runs as written."""

import argparse
import ast
import builtins
import json
import re
import typing
from functools import cache
from pathlib import Path

import pytest

from tests.tools.support import load_tool, synthetic_round, write_config
from worldcast import hub
from worldcast.config import inference, training
from worldcast.config.loader import config_to_dict
from worldcast.engine.inference.session import CLIENT_ENV

REPO = Path(__file__).resolve().parents[2]
#: The guides: what a reader of the release is told.
GUIDES = sorted(
    [
        REPO / "README.md",
        REPO / "examples" / "README.md",
        *(REPO / "docs").glob("*.md"),
    ]
)
NOTICE = REPO / "NOTICE"
CODE = sorted([*(REPO / "worldcast").rglob("*.py"), *(REPO / "tools").glob("*.py")])
#: Every file that documents a command or a setting: the guides, the tools' docstrings, the
#: comments of the shipped configs and of the examples' scripts.
DOCS = [
    *GUIDES,
    *sorted((REPO / "tools").glob("*.py")),
    *sorted((REPO / "configs").rglob("*.yaml")),
    *sorted(path for path in (REPO / "examples").iterdir() if path.suffix in (".yaml", ".sh")),
]
#: The settings of the inference and the training config.
CONFIGS = (
    config_to_dict(inference.InferenceConfig()),
    config_to_dict(training.paper_config("4")),
)
SECTIONS = {section for config in CONFIGS for section in config}
#: Dotted names in the docs that are files or attributes, not settings.
NOT_SETTINGS = re.compile(
    r"\.(json|jsonl|yaml|log|py|js|sh|pt|npy|npz|md|mp4)$|^client\.(records|latents|step|start)"
    r"|^worldcast\.|^model\.(pt|generator)|^torch\."
)
#: A dotted name that is a file's.
FILE_NAME = re.compile(r"\.(md|json|jsonl|yaml|py|js|sh|pt|pth|safetensors|npy|npz|mp4|txt)$")
#: What ``tools/download_examples.py`` writes under ``examples/``: cited, not in a checkout.
DOWNLOADED = re.compile(
    r"examples/(expected|data/\w+/(config\.yaml|opencs2|first_latents|vislabels|obslabels))"
)
#: Cited names that are not the code's: two fields of the visibility labels that no code reads, and
#: a checkpoint directory of a given step.
NOT_CODE = {
    *("_in_frustum", "_binary_offscreen"),
    "checkpoint_model_006000",
}


def name_of(doc: Path) -> str:
    return str(doc.relative_to(REPO))


def text_of(doc: Path) -> str:
    return doc.read_text(encoding="utf-8")


def prose_of(doc: Path) -> str:
    """A Markdown file without its code blocks."""
    return re.sub(r"```.*?```", "", text_of(doc), flags=re.S)


def section_of(doc: Path, heading: str) -> str:
    """The text of a ``##`` section of a Markdown file."""
    return re.split(r"^## ", text_of(doc).split(f"\n## {heading}\n")[1], flags=re.M)[0]


def table_rows(text: str, lead: str = "") -> list[list[str]]:
    """The body rows (their cells) of the first Markdown table after ``lead``; none without it."""
    if lead not in text:
        return []
    blocks = (text.split(lead, 1)[1] if lead else text).split("\n\n")
    table = next(block for block in blocks if block.lstrip().startswith("|")).strip()
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in table.splitlines()[2:]]


def slug(heading: str) -> str:
    """The anchor of a Markdown heading."""
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")


def anchors(doc: Path) -> set[str]:
    return {slug(line.lstrip("#")) for line in prose_of(doc).splitlines() if line.startswith("#")}


@cache
def definitions(file: Path) -> frozenset[str]:
    """Every name a Python file defines (functions, classes, methods, arguments, assigned names and
    attributes) and every word of its string constants (the keys of the files it reads)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(text_of(file))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                for leaf in ast.walk(target):
                    names.update({getattr(leaf, "id", None), getattr(leaf, "attr", None)} - {None})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(re.findall(r"[A-Za-z_]\w*", node.value))
    return frozenset(names)


@cache
def known_names() -> frozenset[str]:
    """What a guide may cite by name: what the code defines, the modules, the shipped configs and
    example cases, the variables of the examples' scripts, and Python's own names."""
    names = set().union(*(definitions(file) for file in CODE)) | {file.stem for file in CODE}
    names |= {path.stem for path in (REPO / "configs").rglob("*.*")}
    names |= set(json.loads(text_of(REPO / hub.EXAMPLES_MANIFEST))["cases"])
    for script in (REPO / "examples").glob("*.sh"):
        names |= set(re.findall(r"\$\{?([A-Z]+)", text_of(script)))
    return frozenset(names | set(dir(builtins)) | set(dir(typing)))


def defined_at(dotted: str) -> bool:
    """Whether ``worldcast.<package>.<module>.<name>`` names a module or what a module defines."""
    here, parts = REPO, dotted.split(".")
    for index, part in enumerate(parts):
        if (here / part).is_dir():
            here = here / part
            continue
        files = [here / f"{part}.py"] if (here / f"{part}.py").is_file() else here.glob("*.py")
        rest = parts[index + (len(files) == 1 if isinstance(files, list) else 0) :]
        return all(any(name in definitions(file) for file in files) for name in rest[:1])
    return True


# ------------------------------------------------------------------------ links, paths and names
@pytest.mark.parametrize("doc", [*GUIDES, NOTICE], ids=name_of)
def test_links_resolve(doc):
    prose = prose_of(doc)
    targets = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", prose) + re.findall(r'src="([^"]+)"', prose)
    for target in targets:
        if target.startswith(("http://", "https://")):
            continue
        file, _, anchor = target.partition("#")
        destination = (doc.parent / file).resolve() if file else doc
        assert destination.exists(), f"{target}: no such file"
        if anchor and destination.suffix == ".md":
            assert anchor in anchors(destination), f"{target}: no such heading"


@pytest.mark.parametrize("doc", [*DOCS, NOTICE], ids=name_of)
def test_cited_paths_exist(doc):
    """A path of the checkout that a doc cites is there."""
    cited = re.findall(
        r"(?<![\w/.-])((?:worldcast|tools|configs|examples|tests)/[\w./-]+)", text_of(doc)
    )
    for path in {path.rstrip(".,:;") for path in cited}:
        assert DOWNLOADED.match(path) or (REPO / path).exists(), f"{path}: no such file"


@pytest.mark.parametrize("doc", GUIDES, ids=name_of)
def test_cited_names_exist(doc):
    """A class, a constant, a call or a dotted ``worldcast.`` name in backticks is the code's."""
    for cited in re.findall(r"`([^`\n]+)`", prose_of(doc)):
        match = re.fullmatch(r"((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)(\(.*\))?", cited)
        if not match or FILE_NAME.search(cited):
            continue
        dotted, call = match.groups()
        if dotted.startswith("worldcast."):
            assert defined_at(dotted), f"{cited}: no such name"
            continue
        parts = dotted.split(".")
        names = {part for part in parts if re.search(r"[A-Z]|_", part)} | (
            {parts[-1]} if call else set()
        )
        assert names <= known_names() | NOT_CODE, f"{cited}: no such name"


def test_the_block_loop_names_the_code():
    """Each numbered step of the block loop of docs/inference.md names a function of the code."""
    section = section_of(REPO / "docs" / "inference.md", "The block loop")
    diagram = re.search(r"```\n(.*?)```", section, flags=re.S).group(1)
    steps = re.findall(r"^ +\d+  \S.*?\s{2,}(\S+)", diagram, flags=re.M)
    assert len(steps) == 10
    for step in steps:
        assert step.split(".")[-1] in known_names(), f"{step}: no such function"


# ------------------------------------------------------------------------- commands and settings
class _Parser(Exception):
    """Carries a tool's argument parser out of its ``main``."""


@cache
def options(tool: str) -> dict[str | None, set[str]]:
    """The flags of a tool: of each of its modes (``"step"``, or ``"step subcommand"`` where a
    mode has subcommands too), or under ``None`` when it has none."""

    def capture(parser, args=None, namespace=None):
        raise _Parser(parser)

    def flags_of(parser, path=()):
        found = {}
        subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if path or not subparsers:
            found[" ".join(path) or None] = {
                flag for action in parser._actions for flag in action.option_strings
            }
        for action in subparsers:
            for name, sub in action.choices.items():
                found.update(flags_of(sub, (*path, name)))
        return found

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        with pytest.raises(_Parser) as captured:
            load_tool(tool).main([])
    return flags_of(captured.value.args[0])


def commands(doc: Path):
    """``(tool, arguments)`` of every ``tools/<tool>.py`` command of a doc."""
    joined = re.sub(r"\\+\n[ #]*", " ", text_of(doc))
    for match in re.finditer(r"tools/(\w+)\.py([^\n`]*)", joined):
        yield match.group(1), match.group(2).split(" #")[0].split()


@pytest.mark.parametrize("doc", DOCS, ids=name_of)
def test_documented_commands_exist(doc):
    for tool, arguments in commands(doc):
        assert (REPO / "tools" / f"{tool}.py").is_file(), f"no tools/{tool}.py"
        flags = {a.split("=")[0] for a in arguments if re.fullmatch(r"--[a-z][\w-]*(=.*)?", a)}
        if not flags:
            continue  # the tool is named, not run
        modes = options(tool)
        mode = None if None in modes else arguments[0]
        if len(arguments) > 1 and f"{mode} {arguments[1]}" in modes:
            mode = f"{mode} {arguments[1]}"  # a subcommand of the mode
        assert mode in modes, f"tools/{tool}.py has no mode {mode}"
        assert flags <= modes[mode], f"tools/{tool}.py {mode or ''} has no {flags - modes[mode]}"


def test_clients_started_by_hand_get_the_environment_of_a_session():
    """The two-machine recipe of examples/README.md exports what ``tools/run_session.py`` gives
    every client."""
    exported = re.findall(r"^export (.+)$", text_of(REPO / "examples" / "README.md"), flags=re.M)
    assert [dict(item.split("=") for item in line.split()) for line in exported] == [CLIENT_ENV]


def is_setting(key: str) -> bool:
    """Whether one of the configs has the dotted ``key``."""

    def has(node, parts) -> bool:
        if not parts:
            return True
        return isinstance(node, dict) and parts[0] in node and has(node[parts[0]], parts[1:])

    return any(has(config, key.split(".")) for config in CONFIGS)


@pytest.mark.parametrize("doc", DOCS, ids=name_of)
def test_documented_settings_exist(doc):
    names = re.findall(r"`([a-z_]+(?:\.[a-z_0-9]+)+)`", text_of(doc))
    names += re.findall(r"--set '?([a-z_.0-9]+)=", text_of(doc))
    for name in names:
        if name.split(".")[0] in SECTIONS and not NOT_SETTINGS.search(name):
            assert is_setting(name), f"{name}: no such setting"


def shipped_configs():
    """Every YAML file of ``configs/`` and the path templates of ``examples/``, each with the
    function that loads it as a tool does."""
    stage4 = REPO / "configs" / "train" / "stage4.yaml"
    for path in sorted((REPO / "configs" / "train").rglob("*.yaml")):
        yield path, lambda path=path: training.load_train_config([path])
    paths = REPO / "examples" / "data_paths.yaml"
    yield paths, lambda: inference.load_config([paths])
    train_paths = REPO / "examples" / "train_paths.yaml"
    yield train_paths, lambda: training.load_train_config([stage4, train_paths])


@pytest.mark.parametrize(
    "path, load", shipped_configs(), ids=lambda value: getattr(value, "name", "")
)
def test_every_shipped_config_loads(path, load):
    load()


def test_no_other_config_is_shipped():
    shipped = {path for path, _ in shipped_configs()}
    found = {*(REPO / "configs").rglob("*.yaml"), *(REPO / "examples").glob("*.yaml")}
    assert found == shipped


# ----------------------------------------------------------------------------- the release files
def test_the_weights_tables_list_the_release_files():
    """The tables of docs/inference.md, "Weights", against ``worldcast.hub``: the files a client
    reads with their ``paths`` keys, the pinned third-party files."""
    section = section_of(REPO / "docs" / "inference.md", "Weights")
    client = table_rows(section, "**What a client reads.**")
    files = {row[1].strip("`"): row[0].strip("`") for row in client}
    assert files == {f"paths.{key}": file for key, file in hub.WEIGHTS.items()}
    third_party = " ".join(
        cell for row in table_rows(section, "**Third-party files**") for cell in row
    )
    pins = (hub.WAN22_REPO, hub.WAN22_REVISION)
    assert all(pin in third_party for pin in pins)


@pytest.mark.parametrize("doc", DOCS, ids=name_of)
def test_a_named_release_file_is_one(doc):
    """A weights file a doc names is a file of ``worldcast.hub``."""
    released = set(hub.WEIGHTS.values())
    for file in set(re.findall(r"\bworldcast_\w+\.safetensors|\b\w+\.safetensors", text_of(doc))):
        assert file in released, f"{file} is not a release file"


# ------------------------------------------------------------------------------- the attribution
def attributed() -> dict[str, dict[str, list[str]]]:
    """Per project of the notice's third-party section, the names it attributes in each file of the
    tree (a name in parentheses is the project's own)."""
    projects = {}
    for section in re.split(r"^## ", text_of(NOTICE).split("## License texts")[0], flags=re.M):
        title, _, body = section.partition("\n")
        items = re.findall(
            r"^  - `(worldcast/[\w/.]+\.py)`: (.*?)(?=^  - |^- |\Z)", body, flags=re.M | re.S
        )
        if items:
            projects[title] = {
                path: re.findall(r"`([A-Za-z_]\w*)`", re.sub(r"\([^()]*\)", "", what))
                for path, what in items
            }
    return projects


def test_attributed_names_are_where_the_file_says():
    projects = attributed()
    assert len(projects) == 7
    for project, files in projects.items():
        for path, names in files.items():
            missing = set(names) - definitions(REPO / path)
            assert names and not missing, f"{project}: {path} has no {missing}"
    for path in projects["CausVid"]:  # "each marked with an attribution comment at its site"
        assert "CausVid" in text_of(REPO / path), f"{path} has no attribution comment"


def test_the_notice_names_the_same_files():
    summary = text_of(NOTICE).split("# Third-party licenses")[0]
    notice = {
        paragraph.split(" (")[0]: set(re.findall(r"worldcast/[\w/.]+\.py", paragraph))
        for paragraph in summary.split("\n\n")
    }
    for project, files in attributed().items():
        assert notice[project] == set(files), project


def test_dependencies_are_listed():
    heading = "Dependencies (installed by pip, not distributed with WorldCast)"
    listed = " ".join(section_of(NOTICE, heading).split())
    project = (
        text_of(REPO / "pyproject.toml").split("[project.urls]")[0].split("dependencies = [", 1)[1]
    )
    packages = set(re.findall(r'"([A-Za-z][\w.-]*)[<>=~!]=', project))
    assert len(packages) > 20
    assert not {package for package in packages if f"{package} (" not in listed}


# ----------------------------------------------------------------------------------- the example
def test_the_library_example_runs(tmp_path, monkeypatch):
    """The "As a library" example of docs/inference.md, on a tiny random model on a synthetic
    round."""
    section = section_of(REPO / "docs" / "inference.md", "As a library")
    code = re.search(r"```python\n(.*?)```", section, flags=re.S).group(1)
    configs = '["weights/paths.yaml", "examples/data/mirage_r16/config.yaml"]'
    assert configs in code
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=1)
    config = write_config(cfg, tmp_path / "config.yaml")
    names: dict = {}
    exec(compile(code.replace(configs, repr([config])), "docs/inference.md", "exec"), names)
    assert tuple(names["field"].shape) == (1, 1, 23, 12, 21) and names["field"].abs().sum() > 0
    assert tuple(names["latents"].shape) == (1, 29, 48, 24, 42)
