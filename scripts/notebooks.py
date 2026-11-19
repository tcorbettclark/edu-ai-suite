"""Run the example notebooks and write their README screenshots.

The commands are ``check``, ``run`` and ``screenshots``; ``--only`` selects one
notebook. This is a development aid, not part of the package or the release check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCREENSHOTS = ROOT / "docs" / "screenshots"
NOTEBOOKS = ("image_features", "language_models", "lora")
TIMEOUT_SECONDS = 600


def _selected(only: list[str] | None) -> list[Path]:
    """The notebook paths selected by ``--only``, in a stable order."""
    names = NOTEBOOKS if not only else tuple(only)
    unknown = sorted(set(names) - set(NOTEBOOKS))
    if unknown:
        raise SystemExit(f"unknown notebook(s): {', '.join(unknown)}")
    return [EXAMPLES / f"{name}.py" for name in names]


def _run(command: list[str], *, timeout: int = TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    """Run one command from the repository root and capture its output."""
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _check(paths: list[Path]) -> bool:
    """Lint the notebooks with marimo's strict checker."""
    result = _run([sys.executable, "-m", "marimo", "check", "--strict", *map(str, paths)])
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return False
    return True


def _export(path: Path, output: Path) -> bool:
    """Execute one notebook by exporting it to HTML, printing any failure."""
    result = _run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(path),
            "--include-code",
            "-o",
            str(output),
        ]
    )
    if result.returncode:
        print(f"{path.name}: failed", file=sys.stderr)
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        return False
    return output.is_file()


def _screenshot(html: Path, png: Path) -> None:
    """Screenshot one exported notebook, using Playwright's Chromium or system Chrome."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception:
            browser = playwright.chromium.launch(channel="chrome")
        try:
            page = browser.new_page(
                viewport={"width": 1440, "height": 1000},
                device_scale_factor=1,
            )
            page.goto(html.as_uri(), wait_until="load")
            page.wait_for_timeout(1_500)
            content_height = page.evaluate(
                """() => {
                    for (const element of document.querySelectorAll("div, li")) {
                        const text = (element.innerText || "").trim();
                        if (
                            (text.startsWith("Static marimo notebook") && text.length < 200) ||
                            (text.startsWith("Static notebook") && text.length < 400)
                        ) {
                            element.style.display = "none";
                        }
                    }
                    let height = document.documentElement.scrollHeight;
                    for (const element of document.querySelectorAll("*")) {
                        height = Math.max(height, element.scrollHeight);
                    }
                    return height;
                }"""
            )
            page.set_viewport_size(
                {"width": 1440, "height": max(1_000, min(content_height, 12_000))}
            )
            page.wait_for_timeout(500)
            png.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(png), full_page=True)
        finally:
            browser.close()


def _run_command(paths: list[Path]) -> bool:
    """Execute every selected notebook through a temporary HTML export."""
    ok = True
    with tempfile.TemporaryDirectory(prefix="edu-ai-suite-notebooks-") as scratch:
        for path in paths:
            html = Path(scratch) / f"{path.stem}.html"
            if _export(path, html):
                print(f"{path.name}: ran")
            else:
                ok = False
    return ok


def _screenshots_command(paths: list[Path]) -> bool:
    """Execute and screenshot every selected notebook."""
    ok = True
    with tempfile.TemporaryDirectory(prefix="edu-ai-suite-notebooks-") as scratch:
        for path in paths:
            html = Path(scratch) / f"{path.stem}.html"
            png = SCREENSHOTS / f"{path.stem}.png"
            if not _export(path, html):
                ok = False
                continue
            try:
                _screenshot(html, png)
            except ImportError:
                print(
                    "playwright is not installed; run `uv sync` and "
                    "`uv run playwright install chromium`",
                    file=sys.stderr,
                )
                return False
            except Exception as exc:
                print(f"{path.name}: screenshot failed: {exc}", file=sys.stderr)
                ok = False
                continue
            print(f"{path.name}: {png.relative_to(ROOT)}")
    return ok


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "run", "screenshots"),
        help="what to do with the notebooks",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=NOTEBOOKS,
        metavar="NAME",
        help="limit the command to one notebook; repeat for more",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _selected(args.only)
    if args.command == "check":
        return 0 if _check(paths) else 1
    if not _check(paths):
        return 1
    if args.command == "run":
        return 0 if _run_command(paths) else 1
    return 0 if _screenshots_command(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
