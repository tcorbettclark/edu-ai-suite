"""Run the example notebooks, their README screenshots and their full HTML pages.

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
HTML_PAGES = ROOT / "docs" / "notebooks"
NOTEBOOKS = ("lora",)
TIMEOUT_SECONDS = 600
HERO_WIDTH = 1_000
HERO_HEIGHT = 1_500

#: The section a short README screenshot starts at; None keeps the top of the page.
HERO_TARGETS = {
    "lora": "Data",
}


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


def _hide_page_decoration(page) -> None:
    """Hide the static banner, notifications and editor controls in the screenshot."""
    page.evaluate(
        """() => {
            const style = document.createElement("style");
            style.textContent = `
                ::-webkit-scrollbar { display: none; }
                * { scrollbar-width: none; }
            `;
            document.head.appendChild(style);
            for (const element of document.querySelectorAll("div, li, a, button")) {
                const text = (element.innerText || "").trim();
                const classes = (element.className || "").toString();
                if (
                    (text.startsWith("Static marimo notebook") && text.length < 200) ||
                    (text.startsWith("Static notebook") && text.length < 400) ||
                    (text.toLowerCase().includes("made with marimo") && text.length < 100) ||
                    classes.includes("absolute -right-9 top-1") ||
                    classes.includes("absolute top-2 right-2 opacity-0") ||
                    classes.includes("right-0 top-0 z-50") ||
                    classes.includes("fixed top-[25vh]")
                ) {
                    element.style.display = "none";
                }
            }
        }"""
    )


def _scroll_to_heading(page, heading: str) -> bool:
    """Scroll the notebook's content pane so a heading is near the top."""
    return page.evaluate(
        """(heading) => {
            const target = [...document.querySelectorAll("h1, h2, h3, h4")].find(
                (element) => (element.innerText || "").trim() === heading
            );
            if (!target) return false;
            const scroller = [...document.querySelectorAll("div")].find((element) => {
                const style = getComputedStyle(element);
                return (
                    element.scrollHeight > element.clientHeight + 100 &&
                    (style.overflowY === "auto" || style.overflowY === "scroll")
                );
            });
            if (!scroller) return false;
            const scrollerRect = scroller.getBoundingClientRect();
            const targetRect = target.getBoundingClientRect();
            scroller.scrollTop += targetRect.top - scrollerRect.top - 24;
            return true;
        }""",
        heading,
    )


def _screenshot(html: Path, png: Path, name: str) -> None:
    """Screenshot one exported notebook, using Playwright's Chromium or system Chrome."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception:
            browser = playwright.chromium.launch(channel="chrome")
        try:
            page = browser.new_page(
                viewport={"width": HERO_WIDTH, "height": HERO_HEIGHT},
                device_scale_factor=1,
            )
            page.goto(html.as_uri(), wait_until="load")
            page.wait_for_timeout(1_500)
            _hide_page_decoration(page)
            target = HERO_TARGETS.get(name)
            if target and not _scroll_to_heading(page, target):
                print(f"{name}: no {target!r} heading; using the top of the page")
            page.wait_for_timeout(500)
            png.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(png), full_page=False)
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
    """Execute every selected notebook, then write its full HTML page and hero PNG."""
    ok = True
    HTML_PAGES.mkdir(parents=True, exist_ok=True)
    for path in paths:
        html = HTML_PAGES / f"{path.stem}.html"
        png = SCREENSHOTS / f"{path.stem}.png"
        if not _export(path, html):
            ok = False
            continue
        try:
            _screenshot(html, png, path.stem)
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
        print(f"{path.name}: {png.relative_to(ROOT)} and {html.relative_to(ROOT)}")
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
