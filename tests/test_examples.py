"""The examples are one marimo notebook and four plain scripts.

The notebooks are the image and LoRA walkthroughs, with screenshots as their rendered
form; the scripts are the language models, one per architecture, and the README names
each of them. This test keeps the directory, the README links and the generated files
in step.
"""

import ast
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCREENSHOTS = ROOT / "docs" / "screenshots"
HTML_PAGES = ROOT / "docs" / "notebooks"
README = ROOT / "README.md"
PAGES_URL = "https://tcorbettclark.github.io/edu-ai-suite/notebooks"

NOTEBOOKS = ("lora",)
SCRIPTS = ("image_features", "train_bigram", "train_mlp", "train_lstm")


def test_examples_hold_the_notebooks_and_the_language_model_scripts():
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == sorted(
        f"{name}.py" for name in NOTEBOOKS + SCRIPTS
    )


def test_every_notebook_is_a_marimo_notebook():
    for name in NOTEBOOKS:
        path = EXAMPLES / f"{name}.py"
        tree = ast.parse(path.read_text())
        assert any(
            isinstance(node, ast.Import) and any(alias.name == "marimo" for alias in node.names)
            for node in tree.body
        ), path.name
        assert any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
            for node in tree.body
        ), path.name
        assert "@app.cell" in path.read_text(), path.name


def _imported_modules(path: pathlib.Path) -> set[str]:
    """The top-level module names a file imports."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add((node.module or "").split(".")[0])
    return names


def test_every_script_takes_no_arguments_and_has_no_notebook_machinery():
    # A reader follows a script top to bottom: the settings are constants, and nothing
    # reads the command line or a cell runtime.
    for name in SCRIPTS:
        path = EXAMPLES / f"{name}.py"
        assert path.is_file(), path
        assert not ({"marimo", "argparse"} & _imported_modules(path)), path.name
        assert "sys.argv" not in path.read_text(), path.name


def test_the_readme_names_every_script():
    readme = README.read_text()
    for name in SCRIPTS:
        assert f"examples/{name}.py" in readme, name


def _png_size(path: pathlib.Path) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", header[16:24])


def test_the_readme_links_every_notebook_screenshot_and_full_page():
    readme = README.read_text()
    for name in NOTEBOOKS:
        assert f"examples/{name}.py" in readme, name
        assert f"docs/screenshots/{name}.png" in readme, name
        assert f"{PAGES_URL}/{name}.html" in readme, name


def test_every_screenshot_exists_and_is_a_hero_crop():
    for name in NOTEBOOKS:
        path = SCREENSHOTS / f"{name}.png"
        assert path.is_file(), path
        assert path.stat().st_size > 0, path
        width, height = _png_size(path)
        assert width <= 1_500, (name, width)
        assert height <= 2_000, (name, height)
        assert height > width, (name, width, height)


def test_every_full_static_page_exists():
    for name in NOTEBOOKS:
        path = HTML_PAGES / f"{name}.html"
        assert path.is_file(), path
        assert path.stat().st_size > 0, path


def test_the_language_model_corpus_is_present():
    path = EXAMPLES / "happy_prince.txt"
    assert path.is_file()
    assert len(path.read_text()) > 1_000


def test_the_maintainer_script_is_present():
    assert (ROOT / "scripts" / "build_britllm_4bit.py").is_file()


def test_the_old_scripts_are_gone():
    assert not (EXAMPLES / "build_britllm_4bit.py").exists()
    assert not (EXAMPLES / "lora_walkthrough.sh").exists()
    for name in ("lm_from_scratch.py", "lora_train.py", "lora_generate.py", "lora_inspect.py"):
        assert not (EXAMPLES / name).exists(), name
