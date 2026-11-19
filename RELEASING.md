# Releasing edu-ai-suite

The version is written once, in `pyproject.toml`. The tag is made from it, never
typed.

## Rules

1. `[project].version` is the only place a version is written. No `__version__`.
2. The tag is `v` + `uv version --short`. `scripts/release.py` makes both.
3. A published version is never reused. A broken 0.0.2 becomes 0.0.3; the tag
   stays where it is.
4. Before 1.0, a minor bump may break a caller. A consumer that needs the 0.0.x
   line brackets its dependency; that bracket, not coordination before release,
   is what keeps a breaking minor out of its environment.

## What each number means here

- **patch** — a fix, with no API change.
- **minor** — new API. Before 1.0 this may also remove or rename something, so
  it can break a caller that does not bracket the 0.0.x line.
- **major** — 1.0 and after, for a breaking change.

## The release

```bash
uv run python scripts/release.py check
uv run python scripts/release.py release patch --dry-run
uv run python scripts/release.py release patch --publish
uv run python scripts/release.py verify
```

`check` refuses when the tree is dirty, the branch is not `main`, HEAD is not
`origin/main`, the tag already exists, the lock is stale, a test fails, or the
built wheel does not install.

`release` bumps the version and the lock, re-runs the suite, rebuilds, commits
`release: X.Y.Z`, tags `vX.Y.Z`, and pushes `main` with the tag. `--publish`
uploads both artifacts, waits for the version to appear on PyPI, and installs it
in a fresh environment.

## If a publish fails

If the tag is pushed and the upload failed, fix the cause and run `uv publish`
again for the same version — the version was never published, so it is not being
reused. If the uploaded wheel itself is wrong, do not re-upload it: bump to the
next patch.
