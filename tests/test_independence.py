"""The two sweeps that hold the package independent of a GUI platform.

The first parses every module under ``edu_ai_suite`` and reads its absolute imports:
importing this package must not need the platform it was split from. The second reads
every source, prose, script and configuration file in the tree and looks for that
platform's names and its toolkit's, which a standalone library has no reason to write.

The names live in :data:`PLATFORM`, :data:`TOOLKIT` and :data:`UI_UNIT` as fragments,
because this file is inside the tree the second sweep reads: a name written whole would
be found in the test that forbids it.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "edu_ai_suite"

#: The GUI platform the package was split from, in fragments.
PLATFORM = ("Ora", "nge")

#: The toolkit that platform draws with, in fragments.
TOOLKIT = ("Q", "t")

#: The toolkit's unit of user interface, in fragments.
UI_UNIT = ("wid", "get")

#: Top-level modules no module under the package may import: the platform, its
#: namespace, its application module and the binding that wraps its toolkit.
FORBIDDEN_IMPORT_ROOTS = (
    "".join(PLATFORM),
    "".join(("Any",) + TOOLKIT),
    "".join(PLATFORM) + "canvas",
    "".join(PLATFORM) + "contrib",
)

#: Text no source, prose or script file may contain: the platform's name, its
#: distribution name and namespace, the toolkit's name, and its unit of user interface.
FORBIDDEN_NAMES = (
    "".join(PLATFORM),
    "".join(PLATFORM) + "3",
    "".join(PLATFORM) + "contrib",
    "".join(TOOLKIT),
    "".join(UI_UNIT),
)

#: Directories a sweep never descends into: not part of the tree that ships.
SKIPPED_DIRS = {".venv", ".git", "dist", "__pycache__", ".pytest_cache", ".ruff_cache"}

#: The files the text sweep reads beyond the three directories.
SWEPT_FILES = (ROOT / "README.md", ROOT / "GLOSSARY.md")

#: The suffixes the text sweep reads; a binary or an image is not prose.
SWEPT_SUFFIXES = {".py", ".md", ".sh", ".toml"}


def _pruned(directory: pathlib.Path):
    """The files under ``directory``, skipping ``SKIPPED_DIRS`` and the sweep's suffixes."""
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix not in SWEPT_SUFFIXES:
            continue
        if SKIPPED_DIRS & set(path.parts):
            continue
        yield path


def _module_files():
    """Every module under the package."""
    yield from _pruned(PACKAGE)


def _swept_files():
    """Every file the text sweep reads: the package, tests and examples, plus the prose."""
    for directory in (PACKAGE, ROOT / "tests", ROOT / "examples"):
        yield from _pruned(directory)
    yield from SWEPT_FILES


def _absolute_imports(path: pathlib.Path):
    """The ``(top-level module, line)`` of every absolute import in ``path``."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


def test_no_module_imports_the_gui_platform():
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: imports {module}"
        for path in _module_files()
        for module, line in _absolute_imports(path)
        if module in FORBIDDEN_IMPORT_ROOTS
    ]
    assert offenders == []


def test_no_file_names_the_gui_platform():
    offenders = [
        f"{path.relative_to(ROOT)}:{number}: {name}"
        for path in _swept_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        for name in FORBIDDEN_NAMES
        if name in line
    ]
    assert offenders == []


def test_the_sweeps_read_the_files_they_are_meant_to_read():
    # Guard the guards: a sweep that quietly stopped reading files would pass by
    # finding nothing at all.
    modules = list(_module_files())
    swept = list(_swept_files())
    assert len(modules) >= 50, len(modules)
    assert len(swept) >= 100, len(swept)
    assert PACKAGE / "__init__.py" in swept
    assert ROOT / "tests" / "test_independence.py" in swept
    assert all(path.suffix == ".py" for path in modules)
    assert all(not SKIPPED_DIRS & set(path.parts) for path in swept)
