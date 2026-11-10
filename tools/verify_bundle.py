#!/usr/bin/env python3
"""Verify the selected source snapshot without requiring its original workspace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/SOURCE_MANIFEST.json"
RUNTIME_DIRS = {".git", ".pytest_cache", "__pycache__", "data", "outputs", "logs",
                "runs", "wandb", "checkpoints", "containers", ".venv"}


def main() -> int:
    payload = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in payload["files"]}
    failures = []
    for name, entry in expected.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            failures.append(f"Missing file or unexpected symlink: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Changed since preparation: {name}")
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in RUNTIME_DIRS or part.startswith(".venv") for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"Unexpected symlink: {relative}")
        elif path.is_file() and relative.as_posix() not in expected and relative.as_posix() != MANIFEST:
            failures.append(f"Unlisted file: {relative}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified {len(expected)} files against {MANIFEST}.")
    print("Runtime directories are ignored; this verifies source integrity, not training quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
