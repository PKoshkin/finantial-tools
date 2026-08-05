"""Compare two directories produced by scripts/run_notebooks.py.

    .venv/bin/python scripts/diff_notebook_outputs.py artifacts/baseline artifacts/after

Exits non-zero and prints a unified diff if any notebook's textual output
changed. Passing means the refactoring did not move a single printed number.
"""

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--context", type=int, default=3)
    args = parser.parse_args()

    before_dir, after_dir = resolve(args.before), resolve(args.after)
    names = sorted({p.name for p in before_dir.glob("*.txt")} | {p.name for p in after_dir.glob("*.txt")})
    if not names:
        print(f"no *.txt found in {before_dir} / {after_dir}", file=sys.stderr)
        return 2

    changed = 0
    for name in names:
        before, after = before_dir / name, after_dir / name
        if not before.exists() or not after.exists():
            missing = before if not before.exists() else after
            print(f"MISSING  {name}: {missing} does not exist")
            changed += 1
            continue

        a = before.read_text().splitlines(keepends=True)
        b = after.read_text().splitlines(keepends=True)
        if a == b:
            print(f"same     {name}")
            continue

        changed += 1
        print(f"CHANGED  {name}")
        sys.stdout.writelines(
            difflib.unified_diff(
                a, b, fromfile=str(before), tofile=str(after), n=args.context
            )
        )

    print(f"\n{len(names) - changed}/{len(names)} notebooks identical")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
