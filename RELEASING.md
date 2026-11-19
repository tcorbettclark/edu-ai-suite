# Releasing edu-ai-suite

The version is written once, in `pyproject.toml`. The tag is made from it, never
typed.

## Rules

1. `[project].version` is the only place a version is written. No `__version__`.
2. The tag is `v` + `uv version --short`. `scripts/release.py` makes both.
3. A published version is never reused. A broken 0.0.2 becomes 0.0.3; the tag
   stays where it is.
4. Before 1.0, a minor bump may break a caller. That is the event the add-on's
   bracket exists for, and the reason for the downstream gate below.

## What each number means here

- **patch** — a fix, with no API change.
- **minor** — new API. Before 1.0 this may also remove or rename something, so
  it can break `orange3-ai-suite`.
- **major** — 1.0 and after, for a breaking change.

## The release

```bash
uv run python scripts/release.py check
uv run python scripts/release.py release patch --dry-run
uv run python scripts/release.py release patch --publish --gate-confirmed
uv run python scripts/release.py verify
```

`check` refuses when the tree is dirty, the branch is not `main`, HEAD is not
`origin/main`, the tag already exists, the lock is stale, a test fails, or the
built wheel does not install.

`release` bumps the version and the lock, re-runs the suite, rebuilds, commits
`release: X.Y.Z`, tags `vX.Y.Z`, and pushes `main` with the tag. `--publish`
uploads both artifacts, waits for the version to appear on PyPI, and installs it
in a fresh environment.

## The downstream gate

`orange3-ai-suite` depends on this package with a bracket
(`>=X.Y.Z,<X.Y.Z`), so a breaking minor here reaches the add-on. Before
publishing any **minor**, run the add-on's widget suite against the candidate
wheel, in the Orange interpreter, and then restore the published version. The
script prints the three commands after `check`; they are:

```bash
uv pip install --python "$ORANGE_PYTHON" --force-reinstall --no-deps dist/edu_ai_suite-X.Y.Z-py3-none-any.whl
(cd ../orange3-ai-suite && QT_QPA_PLATFORM=offscreen "$ORANGE_PYTHON" -m pytest tests/widgets)
uv pip install --python "$ORANGE_PYTHON" edu-ai-suite==<the published version>
```

`--publish` requires `--gate-confirmed`. Patches do not need the gate: they
change no API.

## Ordering

When an API the add-on uses changes: this package publishes first, the add-on
raises its floor and adapts in the same commit, then the add-on releases. Never
the reverse.

## If a publish fails

If the tag is pushed and the upload failed, fix the cause and run `uv publish`
again for the same version — the version was never published, so it is not being
reused. If the uploaded wheel itself is wrong, do not re-upload it: bump to the
next patch.
