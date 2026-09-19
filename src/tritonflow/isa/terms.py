"""The terms a schema cost expression may reference.

The reference below is produced from the registry in `isa/cost.py`, so it cannot
disagree with what the evaluator accepts. Regenerate with
`python3 -m tritonflow.isa.terms`.
"""

from __future__ import annotations

import inspect

__all__ = ["render"]


def render() -> str:
    """A Markdown table of every registered term with its provider's summary line."""
    from . import cost

    rows = ["| Term | Meaning |", "|---|---|"]
    providers = getattr(cost, "_registry", {})
    for name in sorted(providers):
        doc = inspect.getdoc(providers[name]) or ""
        summary = doc.splitlines()[0] if doc else "(no description)"
        rows.append(f"| `{name}` | {summary} |")
    rows.append("")
    rows.append("Indexed forms `stride[i]`, `size[i]`, `offset[i]` and `shape[i]` read the descriptor directly.")
    rows.append("`machine.<param>` reads a parameter from the target's machine file.")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    print(render(), end="")
