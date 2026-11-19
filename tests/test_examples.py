"""The examples are marimo notebooks, and the README points at each one.

The notebooks are the runnable examples; the screenshots are their rendered form.
This test keeps the directory, the README links and the screenshot files in step.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCREENSHOTS = ROOT / "docs" / "screenshots"
README = ROOT / "README.md"

NOTEBOOKS = ("image_features", "language_models", "lora")


def test_examples_hold_exactly_the_three_notebooks():
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == sorted(
        f"{name}.py" for name in NOTEBOOKS
    )


def test_every_example_is_a_marimo_notebook():
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


def test_the_readme_links_every_notebook_and_screenshot():
    readme = README.read_text()
    for name in NOTEBOOKS:
        assert f"examples/{name}.py" in readme, name
        assert f"docs/screenshots/{name}.png" in readme, name


def test_every_screenshot_exists_and_is_not_empty():
    for name in NOTEBOOKS:
        path = SCREENSHOTS / f"{name}.png"
        assert path.is_file(), path
        assert path.stat().st_size > 0, path


def test_the_language_model_corpus_is_present():
    path = EXAMPLES / "happy_prince.txt"
    assert path.is_file()
    assert len(path.read_text()) > 1_000


def test_the_maintainer_script_and_the_marimo_skill_are_present():
    assert (ROOT / "scripts" / "build_britllm_4bit.py").is_file()
    skill = ROOT / ".agents" / "skills" / "marimo-notebook"
    assert (skill / "SKILL.md").is_file()
    assert (skill / "LICENSE").is_file()
    assert (ROOT / "skills-lock.json").is_file()


def test_the_old_scripts_are_gone():
    assert not (EXAMPLES / "build_britllm_4bit.py").exists()
    assert not (EXAMPLES / "lora_walkthrough.sh").exists()
    for name in ("lm_from_scratch.py", "lora_train.py", "lora_generate.py", "lora_inspect.py"):
        assert not (EXAMPLES / name).exists(), name
