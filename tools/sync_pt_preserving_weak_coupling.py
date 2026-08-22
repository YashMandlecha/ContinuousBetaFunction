"""Synchronize the reviewed fit4 weak-coupling diagnostics into fit5/fit6.

The order-3 fit4 notebook is the canonical source for the two reviewed plot
families.  This tool deliberately never writes either fit4 reference notebook.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src/betafn/nf4_original_multiplicative_pt3_order3_fit4.ipynb"
TARGETS = (
    ROOT / "src/betafn/nf4_original_multiplicative_pt3_order3_fit5.ipynb",
    ROOT / "src/betafn/nf4_original_multiplicative_pt3_order2_fit6.ipynb",
)
HEADINGS = (
    "## Figure 11 integral matching to the perturbative regime",
    "## Diagnostic: extend intermediate interpolations first, then take the continuum limit",
)


def text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_cells(notebook: dict) -> list[dict]:
    cells = notebook["cells"]
    selected: list[dict] = []
    for heading in HEADINGS:
        index = next(i for i, cell in enumerate(cells) if heading in text(cell))
        selected.extend(copy.deepcopy(cells[index:index + 2]))
    return selected


def adapt(cells: list[dict]) -> list[dict]:
    for cell in cells:
        source = text(cell)
        source = source.replace(
            "'figure11_integral_matching_fit4'",
            "f'figure11_integral_matching_{FIT_ID}'",
        )
        source = source.replace(
            "'extended_intermediate_to_zero_fit4'",
            "f'extended_intermediate_to_zero_{FIT_ID}'",
        )
        source = source.replace(
            "'extended_interpolation_continuum_to_zero_fit4'",
            "f'extended_interpolation_continuum_to_zero_{FIT_ID}'",
        )
        cell["source"] = source.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return cells


def update_target(path: Path, additions: list[dict]) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # Replace an earlier synchronized block, but leave every unrelated cell
    # (including user plotting cells after the manifest) exactly where it is.
    starts = [i for i, cell in enumerate(cells) if HEADINGS[0] in text(cell)]
    if starts:
        start = starts[0]
        end_heading = next(
            i for i, cell in enumerate(cells[start:], start)
            if HEADINGS[1] in text(cell)
        )
        end = min(end_heading + 2, len(cells))
        cells[start:end] = copy.deepcopy(additions)
    else:
        manifest = next(
            i for i, cell in enumerate(cells)
            if "## Scan manifest and validation summary" in text(cell)
        )
        # Keep the manifest markdown and its following code cell together.
        cells[manifest + 2:manifest + 2] = copy.deepcopy(additions)

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    additions = adapt(source_cells(reference))
    for target in TARGETS:
        update_target(target, additions)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
