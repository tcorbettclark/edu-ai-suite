"""The examples are five plain scripts, and the README names each of them.

The image features, LoRA and the three language models are standalone scripts: fixed settings,
no arguments, no notebook machinery. This test keeps the directory and the README in
step.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
README = ROOT / "README.md"

SCRIPTS = ("image_features", "lora", "train_bigram", "train_mlp", "train_lstm")


def test_examples_hold_exactly_the_scripts():
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == sorted(
        f"{name}.py" for name in SCRIPTS
    )


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
