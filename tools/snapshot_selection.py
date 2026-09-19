#!/usr/bin/env python3
"""Snapshot instruction selection decisions across schemas and fixtures (Task A3).

Freezes the compiler's instruction selection so that any cost or schema changes
silently affecting emitted instructions surface as visible diffs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tritonflow.emit.assemble import assemble
from tritonflow.emit.ir import Instr, Loop
from tritonflow.idioms.detect import annotate
from tritonflow.isa.schema import load_builtin
from tritonflow.ttir.graph import build_def_use
from tritonflow.ttir.to_ir import parse_module

FIXTURES_DIR = ROOT / "fixtures"
GOLDEN_FILE = FIXTURES_DIR / "selection_golden.json"
LAUNCH = json.loads((FIXTURES_DIR / "launch_env.json").read_text())

SCHEMAS = ["tritonflow1", "tritonflow2", "vortex_rvgpu"]
TIERS = ["t0_vecadd", "t1_matmul", "t2_matmul_relu", "t3_modulo"]


def capture_selections() -> dict:
    records = {}
    for schema_name in sorted(SCHEMAS):
        schema = load_builtin(schema_name)
        schema_records = {}
        for tier in sorted(TIERS):
            ttir_path = FIXTURES_DIR / f"{tier}.ttir"
            res = parse_module(ttir_path.read_text(), source_path=str(ttir_path))
            module = res.module
            graph = build_def_use(module)
            annotations = annotate(module, graph)
            program = assemble(module, graph, annotations, schema, env=dict(LAUNCH[tier]))

            tier_instrs = []

            def record_instr(instr: Instr, loop_id: int | None = None) -> None:
                src = instr.source
                op_name = src.op_name if src else None
                line = src.line if src else None
                col = src.col if src else None
                loc = src.loc_name if src else None
                tier_instrs.append({
                    "instruction": instr.name,
                    "source_op": op_name,
                    "line": line,
                    "col": col,
                    "loc": loc,
                    "defs": list(instr.defs),
                    "loop": loop_id,
                })

            for item in program.execution_order():
                if isinstance(item, Instr):
                    record_instr(item)
                elif isinstance(item, Loop):
                    for sub in item.body:
                        if isinstance(sub, Instr):
                            record_instr(sub, loop_id=item.id)

            schema_records[tier] = {
                "markers": [str(m) for m in program.markers()],
                "instructions": tier_instrs,
            }
        records[schema_name] = schema_records
    return {"schema_version": 1, "selections": records}


def write_snapshot() -> int:
    data = capture_selections()
    GOLDEN_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_FILE.relative_to(ROOT)}: 3 schemas x 4 fixtures")
    return 0


def check_snapshot() -> int:
    if not GOLDEN_FILE.exists():
        print(f"Selection golden missing: {GOLDEN_FILE.relative_to(ROOT)}; run with --write", file=sys.stderr)
        return 1

    golden = json.loads(GOLDEN_FILE.read_text())
    current = capture_selections()

    diffs: list[str] = []
    g_sel = golden.get("selections", {})
    c_sel = current.get("selections", {})

    for schema_name in sorted(set(g_sel.keys()) | set(c_sel.keys())):
        if schema_name not in g_sel:
            diffs.append(f"schema {schema_name} missing from golden")
            continue
        if schema_name not in c_sel:
            diffs.append(f"schema {schema_name} missing from current")
            continue

        for tier in sorted(set(g_sel[schema_name].keys()) | set(c_sel[schema_name].keys())):
            g_tier = g_sel[schema_name].get(tier, {})
            c_tier = c_sel[schema_name].get(tier, {})

            g_markers = g_tier.get("markers", [])
            c_markers = c_tier.get("markers", [])
            if g_markers != c_markers:
                diffs.append(f"[{schema_name} x {tier}] markers differ:\n  golden:  {g_markers}\n  current: {c_markers}")

            g_inst = g_tier.get("instructions", [])
            c_inst = c_tier.get("instructions", [])
            if len(g_inst) != len(c_inst):
                diffs.append(f"[{schema_name} x {tier}] instruction count differ: golden={len(g_inst)}, current={len(c_inst)}")

            limit = min(len(g_inst), len(c_inst))
            for i in range(limit):
                gi = g_inst[i]
                ci = c_inst[i]
                if gi["instruction"] != ci["instruction"]:
                    diffs.append(
                        f"[{schema_name} x {tier}] instr #{i} changed: {gi['instruction']} -> {ci['instruction']} "
                        f"(source: {gi['source_op']} at L{gi['line']}:{gi['col']})"
                    )

    if diffs:
        print(f"SELECTION DRIFT DETECTED ({len(diffs)} difference(s)):", file=sys.stderr)
        for d in diffs:
            print(f"  - {d}", file=sys.stderr)
        return 1

    print("selection snapshot OK: all 3 schemas x 4 fixtures match golden exactly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="Record current instruction selections to golden file")
    g.add_argument("--check", action="store_true", help="Verify instruction selections match golden file")
    args = ap.parse_args()

    if args.write:
        return write_snapshot()
    return check_snapshot()


if __name__ == "__main__":
    sys.exit(main())
