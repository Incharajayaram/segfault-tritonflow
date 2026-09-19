#!/usr/bin/env python3
"""run_kernel_corpus.py — Evaluates the 22-kernel Triton corpus through the full pipeline (Task E1).

Emits `reports/kernel_corpus_status.md` recording the named outcome for each kernel at every stage:
Extract -> Parse -> Recognise -> Select -> Assemble -> Execute -> Reference Parity.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bench.corpus.kernels import get_corpus

from tritonflow.extract.aot import compile_aot


def main() -> int:
    corpus = get_corpus()
    print(f"Evaluating {len(corpus)} kernels through the compiler pipeline...\n")
    results = []

    for i, kernel in enumerate(corpus, 1):
        print(f"[{i:02d}/{len(corpus):02d}] {kernel.name} ({kernel.category})...")
        inputs = kernel.make_inputs()
        res = compile_aot(
            fn=kernel.fn,
            signature=kernel.signature,
            constexprs=kernel.constexprs,
            name=kernel.name,
            isa_name="vortex_rvgpu",
            env=kernel.env,
            inputs=inputs,
            reference_fn=kernel.reference,
        )
        results.append((kernel, res))
        print(f"       extract={res.outcomes['extract'].status} "
              f"parse={res.outcomes['parse'].status} "
              f"recognise={res.outcomes['recognise'].status} "
              f"select={res.outcomes['select'].status} "
              f"assemble={res.outcomes['assemble'].status} "
              f"execute={res.outcomes['execute'].status} "
              f"parity={res.outcomes['parity'].status}")

    # Generate Markdown Table Report
    report_path = ROOT / "reports/kernel_corpus_status.md"
    lines = [
        "# Kernel Corpus Status Report (Task E1)\n",
        "Evaluates 23 diverse, hand-written `@triton.jit` kernels compiled ahead-of-time without a GPU ",
        "through the full compiler pipeline on target `vortex_rvgpu`.\n",
        "This replaces fixture-bound evaluation (`fixtures/t0..t3`) with an open-ended corpus covering ",
        "reductions, broadcasts, transposes, indirect indexing, deep elementwise chains, control flow, ",
        "strided 2D tiles, and mixed dtypes.\n",
        "## Pipeline Stages\n",
        "- **Extract**: Ahead-of-time Triton compilation (`GPUTarget('cuda', 80, 32)`) producing TTIR.",
        "- **Parse**: TTIR syntax parsing (`parse_module`).",
        "- **Recognise**: Affine def-use walk and access descriptor recovery (`annotate`).",
        "- **Select**: Target ISA instruction matching against `vortex_rvgpu.yaml` (`assemble`).",
        "- **Assemble**: Instruction sequencing, placement, and cost assignment.",
        "- **Execute**: Hardware state emulation (`emulate`).",
        "- **Parity**: Numerical equivalence against NumPy reference.\n",
        "## Results Table\n",
        "| Kernel | Category | Description | Extract | Parse | Recognise | Select | Assemble | Execute | Parity |",
        "|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    counts = {"extract": 0, "parse": 0, "recognise": 0, "select": 0, "assemble": 0, "execute": 0, "parity": 0}

    for kernel, res in results:
        ext = res.outcomes["extract"].status
        prs = res.outcomes["parse"].status
        rec = res.outcomes["recognise"].status
        sel = res.outcomes["select"].status
        asm = res.outcomes["assemble"].status
        exe = res.outcomes["execute"].status
        par = res.outcomes["parity"].status

        if ext == "OK":
            counts["extract"] += 1
        if prs == "OK":
            counts["parse"] += 1
        if rec == "OK":
            counts["recognise"] += 1
        if sel == "OK":
            counts["select"] += 1
        if asm == "OK":
            counts["assemble"] += 1
        if exe == "OK":
            counts["execute"] += 1
        if par == "PASS":
            counts["parity"] += 1

        row = (
            f"| `{kernel.name}` | {kernel.category} | {kernel.description} | "
            f"{ext} | {prs} | {rec} | {sel} | {asm} | {exe} | {par} |"
        )
        lines.append(row)

    total = len(corpus)
    lines.extend([
        "",
        "## Summary & Findings\n",
        "| Stage | Passed | Failed / Refused | Pass Rate |",
        "|---|:---:|:---:|:---:|",
        f"| Extract (Triton AOT) | {counts['extract']}/{total} | {total - counts['extract']}/{total} | {counts['extract']/total*100:.1f}% |",
        f"| Parse (`parse_module`) | {counts['parse']}/{total} | {total - counts['parse']}/{total} | {counts['parse']/total*100:.1f}% |",
        f"| Recognise (`annotate`) | {counts['recognise']}/{total} | {total - counts['recognise']}/{total} | {counts['recognise']/total*100:.1f}% |",
        f"| Select (`vortex_rvgpu`) | {counts['select']}/{total} | {total - counts['select']}/{total} | {counts['select']/total*100:.1f}% |",
        f"| Assemble | {counts['assemble']}/{total} | {total - counts['assemble']}/{total} | {counts['assemble']/total*100:.1f}% |",
        f"| Execute (`emulate`) | {counts['execute']}/{total} | {total - counts['execute']}/{total} | {counts['execute']/total*100:.1f}% |",
        f"| Reference Parity | {counts['parity']}/{total} | {total - counts['parity']}/{total} | {counts['parity']/total*100:.1f}% |\n",
        "### Root Causes for Non-Passing Kernels\n",
    ])

    # Detail failures
    for kernel, res in results:
        for stage_name in ["extract", "parse", "recognise", "select", "assemble", "execute", "parity"]:
            st = res.outcomes[stage_name]
            if st.status not in ("OK", "PASS", "SKIPPED"):
                lines.append(f"- **`{kernel.name}` ({stage_name})**: `{st.status}` — {st.detail}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
