#!/usr/bin/env python3
"""AST-level mutation testing for the TritonFlow compiler modules (A5).

Each mutant changes exactly one operator or boolean constant in one of the target
modules. The relevant tests are then run against the mutant and the outcome is one of:

- killed:   a test that passed on the unmutated code now fails;
- survived: nothing that passed on the unmutated code fails;
- error:    the harness could not decide (timeout, unparseable output). Errors are
            reported separately and never counted as kills.

Safety: the real tree is never written. The needed files are copied into private
temporary trees and every mutant is executed there, so a crash, a Ctrl-C or a
concurrent process cannot leave a mutated source behind.

Determinism: sampling uses a fixed seed and is stratified by operator kind within
each file. The report lists the seed and every mutant id, contains no clock, and the
baseline and after reports use the identical mutant set.

Every survivor of the "after" sweep must be either killed by a test or listed, with a
reason, in tools/mutation_equivalents.json; otherwise this script exits non-zero.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 24173
TIMEOUT_S = 240
EQUIVALENTS_PATH = ROOT / "tools" / "mutation_equivalents.json"

TARGET_FILES = [
    Path("src/tritonflow/emu/exec.py"),
    Path("src/tritonflow/isa/select.py"),
    Path("src/tritonflow/isa/cost.py"),
    Path("src/tritonflow/idioms/detect.py"),
]

MODULE_NAMES: dict[Path, str] = {
    Path("src/tritonflow/emu/exec.py"): "tritonflow.emu.exec",
    Path("src/tritonflow/isa/select.py"): "tritonflow.isa.select",
    Path("src/tritonflow/isa/cost.py"): "tritonflow.isa.cost",
    Path("src/tritonflow/idioms/detect.py"): "tritonflow.idioms.detect",
}

KILL_TESTS = "tests/mutation_kills"

RELEVANT_TESTS: dict[Path, list[str]] = {
    Path("src/tritonflow/emu/exec.py"): [
        "tests/contract/test_emulator.py",
        "tests/contract/test_emulator_differential.py",
        "tests/test_semantics_oracle.py",
        "tests/test_schema_op_matrix.py",
        KILL_TESTS,
    ],
    Path("src/tritonflow/isa/select.py"): [
        "tests/contract/test_selector.py",
        "tests/test_schema_op_matrix.py",
        "tests/contract/test_end_to_end.py",
        KILL_TESTS,
    ],
    Path("src/tritonflow/isa/cost.py"): [
        "tests/test_cost_model.py",
        "tests/property/test_cost_monotonic.py",
        KILL_TESTS,
    ],
    Path("src/tritonflow/idioms/detect.py"): [
        "tests/contract/test_assembler.py",
        "tests/contract/test_gather_scatter.py",
        "tests/contract/test_to_ir.py",
        "tests/test_address_subsumption.py",
        KILL_TESTS,
    ],
}

# The baseline sweep answers "what did the suite catch before the safety-net work":
# these are the checks that did not exist then.
BASELINE_IGNORED = (
    "tests/contract/test_emulator_differential.py",
    "tests/test_semantics_oracle.py",
    "tests/test_schema_op_matrix.py",
    KILL_TESTS,
)

COPY_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "run_logs", "build", ".claude", "research_notes", "scratch",
    "audit_evidence", "repos", ".hypothesis", ".pytest_cache", "*.zip", "*.pdf", "*.pptx",
    ".freebuff", "specs", "diffs",
)


@dataclass
class Mutant:
    mutant_id: str
    file_path: Path
    line_number: int
    op_type: str
    description: str
    original: str
    mutated: str
    mutated_source: str


@dataclass
class Outcome:
    status: str  # killed | survived | error
    new_failures: tuple[str, ...] = ()
    detail: str = ""


@dataclass
class SweepResult:
    outcomes: dict[str, Outcome] = field(default_factory=dict)


class ASTSingleMutator(ast.NodeTransformer):
    """Mutate exactly one target node matching (lineno, col_offset)."""

    _BIN = {"Add->Sub": (ast.Add, ast.Sub), "Sub->Add": (ast.Sub, ast.Add), "Mult->Div": (ast.Mult, ast.Div)}
    _CMP = {
        "Eq->NotEq": (ast.Eq, ast.NotEq), "NotEq->Eq": (ast.NotEq, ast.Eq),
        "Lt->Gt": (ast.Lt, ast.Gt), "Gt->Lt": (ast.Gt, ast.Lt),
        "LtE->GtE": (ast.LtE, ast.GtE), "GtE->LtE": (ast.GtE, ast.LtE),
    }

    def __init__(self, lineno: int, col: int, op_type: str):
        self.lineno, self.col, self.op_type = lineno, col, op_type
        self.applied = False

    def _at(self, node: ast.AST) -> bool:
        return getattr(node, "lineno", None) == self.lineno and getattr(node, "col_offset", None) == self.col

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        pair = self._BIN.get(self.op_type)
        if pair and self._at(node) and isinstance(node.op, pair[0]) and not self.applied:
            self.applied = True
            node = ast.copy_location(ast.BinOp(left=node.left, op=pair[1](), right=node.right), node)
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        pair = self._CMP.get(self.op_type)
        if pair and self._at(node) and not self.applied:
            ops = list(node.ops)
            for i, op in enumerate(ops):
                if isinstance(op, pair[0]):
                    ops[i] = pair[1]()
                    self.applied = True
                    break
            node = ast.copy_location(ast.Compare(left=node.left, ops=ops, comparators=node.comparators), node)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.op_type == "BoolFlip" and self._at(node) and isinstance(node.value, bool) and not self.applied:
            self.applied = True
            return ast.copy_location(ast.Constant(value=not node.value), node)
        return node


class CandidateFinder(ast.NodeVisitor):
    _BIN = {ast.Add: "Add->Sub", ast.Sub: "Sub->Add", ast.Mult: "Mult->Div"}
    _CMP = {ast.Eq: "Eq->NotEq", ast.NotEq: "NotEq->Eq", ast.Lt: "Lt->Gt", ast.Gt: "Gt->Lt",
            ast.LtE: "LtE->GtE", ast.GtE: "GtE->LtE"}

    def __init__(self) -> None:
        self.candidates: list[tuple[int, int, str]] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        kind = self._BIN.get(type(node.op))
        if kind:
            self.candidates.append((node.lineno, node.col_offset, kind))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            kind = self._CMP.get(type(op))
            if kind:
                self.candidates.append((node.lineno, node.col_offset, kind))
                break
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool):
            self.candidates.append((node.lineno, node.col_offset, "BoolFlip"))


def _describe(kind: str) -> str:
    return kind.replace("->", " -> ")


def _stratified_sample(candidates: list[tuple[int, int, str]], limit: int, key: str) -> list[tuple[int, int, str]]:
    """Round-robin over operator kinds, each kind shuffled with a fixed per-file seed."""
    rng = random.Random(f"{SEED}:{key}")
    by_kind: dict[str, list[tuple[int, int, str]]] = {}
    for cand in candidates:
        by_kind.setdefault(cand[2], []).append(cand)
    for kind in sorted(by_kind):
        rng.shuffle(by_kind[kind])
    chosen: list[tuple[int, int, str]] = []
    kinds = sorted(by_kind)
    while len(chosen) < limit and any(by_kind[k] for k in kinds):
        for kind in kinds:
            if by_kind[kind] and len(chosen) < limit:
                chosen.append(by_kind[kind].pop())
    return sorted(chosen)


def _copy_tree(dest: Path) -> None:
    for item in ROOT.iterdir():
        if COPY_IGNORE(str(ROOT), [item.name]):
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=COPY_IGNORE)
        else:
            shutil.copy2(item, target)


def _env(tree: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tree / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _importable(tree: Path, module: str) -> bool:
    res = subprocess.run(
        [sys.executable, "-c", f"import {module}"], cwd=tree, env=_env(tree), capture_output=True, timeout=60
    )
    return res.returncode == 0


_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")


def _run_tests(tree: Path, tests: list[str]) -> tuple[set[str], int, str]:
    """Run pytest in `tree`; return (ids that failed or errored, return code, tail of output)."""
    present = [t for t in tests if (tree / t).exists()]
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rfE", "-p", "no:cacheprovider",
           "-p", "no:randomly", *present]
    proc = subprocess.run(cmd, cwd=tree, env=_env(tree), capture_output=True, text=True, timeout=TIMEOUT_S)
    failed = set()
    for line in proc.stdout.splitlines():
        m = _FAIL_RE.match(line)
        if m:
            failed.add(m.group(1))
    return failed, proc.returncode, proc.stdout[-400:]


def collect_mutants(tree: Path, max_per_file: int) -> list[Mutant]:
    mutants: list[Mutant] = []
    for rel in TARGET_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        base_unparsed = ast.unparse(ast.parse(text)).splitlines()
        finder = CandidateFinder()
        finder.visit(ast.parse(text))
        seen: set[str] = set()
        pool = _stratified_sample(finder.candidates, max_per_file * 3, str(rel))
        for lineno, col, kind in pool:
            if sum(1 for m in mutants if m.file_path == rel) >= max_per_file:
                break
            mutator = ASTSingleMutator(lineno, col, kind)
            mut_tree = mutator.visit(ast.parse(text))
            if not mutator.applied:
                continue
            ast.fix_missing_locations(mut_tree)
            source = ast.unparse(mut_tree)
            mut_lines = source.splitlines()
            diff = [d for d in difflib.ndiff(base_unparsed, mut_lines) if d[:2] in ("- ", "+ ")]
            removed = [d[2:].strip() for d in diff if d[:2] == "- "]
            added = [d[2:].strip() for d in diff if d[:2] == "+ "]
            if not removed or not added:
                continue
            mutant_id = f"{rel.as_posix()}:{lineno}:{col}:{kind}"
            if mutant_id in seen:
                continue
            seen.add(mutant_id)
            (tree / rel).write_text(source, encoding="utf-8")
            try:
                ok = _importable(tree, MODULE_NAMES[rel])
            finally:
                (tree / rel).write_text(text, encoding="utf-8")
            if not ok:
                continue
            mutants.append(Mutant(mutant_id, rel, lineno, kind, _describe(kind),
                                  " ".join(removed), " ".join(added), source))
    return mutants


def _evaluate(mutant: Mutant, tree: Path, baseline_fail: set[str]) -> Outcome:
    target = tree / mutant.file_path
    original = target.read_text(encoding="utf-8")
    try:
        target.write_text(mutant.mutated_source, encoding="utf-8")
        failed, rc, tail = _run_tests(tree, RELEVANT_TESTS[mutant.file_path])
    except subprocess.TimeoutExpired:
        return Outcome("error", detail=f"timeout after {TIMEOUT_S}s")
    except OSError as exc:
        return Outcome("error", detail=f"harness error: {exc}")
    finally:
        target.write_text(original, encoding="utf-8")
    new = tuple(sorted(failed - baseline_fail))
    if new:
        return Outcome("killed", new)
    if rc not in (0, 1):
        return Outcome("error", detail=f"pytest rc={rc}: {tail.strip()[-160:]}")
    return Outcome("survived")


def sweep(mutants: list[Mutant], jobs: int) -> tuple[dict[str, Outcome], dict[Path, set[str]]]:
    """One pytest run per mutant against all relevant tests, in private copies of the tree."""
    trees: queue.Queue[Path] = queue.Queue()
    made: list[Path] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="tf_mut_"))
    try:
        for i in range(jobs):
            d = tmp_root / f"w{i}"
            d.mkdir()
            _copy_tree(d)
            made.append(d)
            trees.put(d)

        baseline_fail: dict[Path, set[str]] = {}
        for rel in TARGET_FILES:
            failed, _rc, _tail = _run_tests(made[0], RELEVANT_TESTS[rel])
            baseline_fail[rel] = failed

        def work(m: Mutant) -> tuple[str, Outcome]:
            tree = trees.get()
            try:
                return m.mutant_id, _evaluate(m, tree, baseline_fail[m.file_path])
            finally:
                trees.put(tree)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = dict(pool.map(work, mutants))
        return results, baseline_fail
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _under(test_id: str, prefixes: tuple[str, ...]) -> bool:
    path = test_id.split("::", 1)[0]
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def _views(outcomes: dict[str, Outcome]) -> tuple[dict[str, str], dict[str, str]]:
    """Per-mutant status for the baseline view (new checks removed) and the after view."""
    baseline: dict[str, str] = {}
    after: dict[str, str] = {}
    for mid, out in outcomes.items():
        after[mid] = out.status
        if out.status == "killed":
            baseline[mid] = "killed" if any(not _under(t, BASELINE_IGNORED) for t in out.new_failures) else "survived"
        else:
            baseline[mid] = out.status
    return baseline, after


def _report(path: Path, title: str, mutants: list[Mutant], statuses: dict[str, str],
            equivalents: dict[str, str], baseline_fail: dict[Path, set[str]], outcomes: dict[str, Outcome]) -> None:
    killed = [m for m in mutants if statuses[m.mutant_id] == "killed"]
    survived = [m for m in mutants if statuses[m.mutant_id] == "survived"]
    errors = [m for m in mutants if statuses[m.mutant_id] == "error"]
    decided = len(killed) + len(survived)
    score = 100.0 * len(killed) / decided if decided else 0.0
    reviewed = [m for m in survived if m.mutant_id in equivalents]
    open_ = [m for m in survived if m.mutant_id not in equivalents]
    adjusted = 100.0 * len(killed) / (decided - len(reviewed)) if decided - len(reviewed) else 0.0
    lines = [
        f"# Mutation report: {title}\n",
        "Each mutant changes one operator or boolean constant in one target module and is run in a private copy of the tree. "
        "A mutant is killed only when a test that passes on the unmutated code fails on it; a timeout or harness fault is an "
        "`error` and is not counted as a kill.\n",
        f"Sampling: seed `{SEED}`, stratified by operator kind within each file. Identical mutant set for the baseline and after reports.\n",
        "## Summary\n", "| Metric | Value |", "|---|---|",
        f"| Mutants | {len(mutants)} |", f"| Killed | {len(killed)} |", f"| Survived | {len(survived)} |",
        f"| Error (excluded from score) | {len(errors)} |",
        f"| **Score, killed / (killed + survived)** | **{score:.1f}%** |",
        f"| Survivors reviewed as behaviourally equivalent | {len(reviewed)} |",
        f"| Score excluding reviewed equivalents | {adjusted:.1f}% |",
        f"| Survivors neither killed nor reviewed | {len(open_)} |\n",
    ]
    pre = {rel.as_posix(): sorted(v) for rel, v in baseline_fail.items() if v}
    lines.append("## Tests already failing on the unmutated code (excluded from kill decisions)\n")
    if pre:
        for rel, ids in sorted(pre.items()):
            lines.append(f"- `{rel}` tests: {len(ids)} failing before mutation")
        lines.append("")
    else:
        lines.append("None.\n")
    lines.append("## Mutant ids\n")
    for m in mutants:
        lines.append(f"- `{m.mutant_id}`: {statuses[m.mutant_id]}")
    lines.append("")
    for heading, group in (("Survivors", survived), ("Errors", errors)):
        lines.append(f"## {heading}\n")
        if not group:
            lines.append("None.\n")
        for m in group:
            lines += [f"- `{m.mutant_id}` ({m.description})",
                      f"  - Original: `{m.original}`", f"  - Mutated:  `{m.mutated}`"]
            if m.mutant_id in equivalents and heading == "Survivors":
                lines.append(f"  - Reviewed equivalent: {equivalents[m.mutant_id]}")
            if heading == "Errors":
                lines.append(f"  - Detail: {outcomes[m.mutant_id].detail}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-per-file", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="tf_mut_collect_") as scratch:
        collect_tree = Path(scratch)
        _copy_tree(collect_tree)
        mutants = collect_mutants(collect_tree, args.max_per_file)
    print(f"{len(mutants)} valid, importable mutants (seed {SEED})")

    outcomes, baseline_fail = sweep(mutants, args.jobs)
    baseline, after = _views(outcomes)
    equivalents = json.loads(EQUIVALENTS_PATH.read_text()) if EQUIVALENTS_PATH.exists() else {}
    out_dir = Path(args.out_dir)
    _report(out_dir / "mutation_baseline.md", "baseline (safety-net checks removed)", mutants, baseline,
            equivalents, baseline_fail, outcomes)
    _report(out_dir / "mutation_after.md", "after (all checks active)", mutants, after,
            equivalents, baseline_fail, outcomes)

    ids = {m.mutant_id for m in mutants}
    survivors = [m.mutant_id for m in mutants if after[m.mutant_id] == "survived"]
    unreviewed = [s for s in survivors if s not in equivalents]
    stale = [k for k in equivalents if k not in ids or after.get(k) != "survived"]
    errors = [k for k, v in after.items() if v == "error"]
    for label, group in (("survivor neither killed nor reviewed", unreviewed),
                         ("equivalents entry that is stale or not a survivor", stale),
                         ("mutant with harness error", errors)):
        for item in group:
            print(f"FAIL: {label}: {item}")
    killed = sum(1 for v in after.values() if v == "killed")
    print(f"after: killed={killed} survived={len(survivors)} error={len(errors)}")
    return 1 if (unreviewed or stale or errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
