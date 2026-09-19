#!/usr/bin/env python3
"""Render bench/results.json as the results table.

The table is the artifact; prose is generated from it (methodology §9.7).
Headline rows are marked, upper-bound rows are labelled, and an unavailable row
says why rather than disappearing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(value, unit: str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, float):
        return f"{value:.4g}" + (f" {unit}" if unit else "")
    return f"{value}" + (f" {unit}" if unit else "")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results.json")
    if not path.exists():
        print(f"no results at {path} (bench is non-gating)")
        return 0
    data = json.loads(path.read_text())
    p = data["provenance"]
    print(f"bench schema {data['schema']} · seed {p.get('seed')} · py {p.get('python')} · "
          f"{p.get('platform')} · no GPU")
    print()
    print("| tier | metric | value | status | note |")
    print("|---|---|---|---|---|")
    for r in data["results"]:
        metric = r["metric"] + (" *(headline)*" if r.get("headline") else "")
        if r.get("upper_bound"):
            metric += " *(upper bound)*"
        note = r.get("reason") or r.get("formula", "")
        if "cost" in r.get("metric", ""):
            note += " [modelled cost (uncalibrated) — not comparable across targets]"
        print(f"| `{r['tier']}` | {metric} | {fmt(r['value'], r['unit'])} | {r['status']} | {note} |")
    missing = [r["id"] for r in data["results"] if r["status"] != "ok"]
    if missing:
        print()
        print(f"**{len(missing)} row(s) not measurable yet:** " + ", ".join(f"`{m}`" for m in missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
