"""Execute every notebook and dump its textual output for regression diffing.

    .venv/bin/python scripts/run_notebooks.py --out artifacts/baseline

Writes, per notebook, into the target directory:

    <name>.ipynb    the executed notebook (outputs included)
    <name>.txt      the textual outputs only, one block per cell

Images are deliberately excluded from the .txt: matplotlib PNGs are not stable
enough at the byte level to diff, while every number the notebooks print or
render as a DataFrame is. Memory addresses in reprs are normalised so the .txt
is reproducible across runs.
"""

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebook"

_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


def normalise(text: str) -> str:
    """Strip the only non-reproducible part of a repr: the memory address."""
    return _ADDRESS.sub("0xADDR", text)


def cell_text(cell) -> list[str]:
    blocks: list[str] = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            blocks.append(f"[{output.get('name', 'stdout')}]\n{output.get('text', '')}")
        elif kind in ("execute_result", "display_data"):
            data = output.get("data", {})
            for mime in ("text/plain", "text/html"):
                if mime in data:
                    payload = data[mime]
                    if isinstance(payload, list):
                        payload = "".join(payload)
                    blocks.append(f"[{mime}]\n{payload}")
        elif kind == "error":
            trace = "\n".join(output.get("traceback", []))
            blocks.append(f"[error] {output.get('ename')}: {output.get('evalue')}\n{trace}")
    return blocks


def dump(notebook) -> str:
    parts: list[str] = []
    index = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        index += 1
        blocks = cell_text(cell)
        if not blocks:
            continue
        parts.append(f"===== cell {index} =====")
        parts.extend(blocks)
    return normalise("\n".join(parts).rstrip()) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="target directory")
    parser.add_argument("--only", help="run a single notebook by stem")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if args.only:
        paths = [p for p in paths if p.stem == args.only]
        if not paths:
            print(f"no notebook named {args.only!r}", file=sys.stderr)
            return 2

    failed = False
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
            allow_errors=True,
        )
        client.execute()

        nbformat.write(notebook, out_dir / path.name)
        text_path = out_dir / f"{path.stem}.txt"
        text_path.write_text(dump(notebook))

        errors = sum(
            1
            for cell in notebook.cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        )
        status = f"{errors} ERROR(S)" if errors else "ok"
        failed = failed or bool(errors)
        print(f"{path.name}: {status} -> {text_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
