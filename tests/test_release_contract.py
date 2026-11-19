"""The version is written once, in ``pyproject.toml``, and it is a release version."""

import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_the_version_is_three_numbers():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _project()["version"])


def test_no_module_keeps_a_second_version():
    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "edu_ai_suite").rglob("*.py")
        if "__version__" in path.read_text()
    ]
    assert offenders == []
