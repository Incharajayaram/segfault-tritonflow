#!/usr/bin/env python3
"""Verify vortex_rvgpu.yaml against upstream Vortex hardware constants (Task E5).

Checks schema constants against upstream VX_config.toml and VX_types.toml:
1. TCU performance-counter CSR addresses (0xB03, 0xB04, 0xB05)
2. TCU latency values (DPI: 4, DSP: 42, BHF: 13, TFR: 4)
3. Number of TCU blocks (1 for single-issue core)
4. Scratchpad bank count (4 for 4-lane LSU)
5. Cache line size (64 bytes matching L1 / MEM_BLOCK_SIZE)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

UPSTREAM_CONFIG_PATH = ROOT / "third_party" / "vortex" / "VX_config.toml"
UPSTREAM_TYPES_PATH = ROOT / "third_party" / "vortex" / "VX_types.toml"
SCHEMA_PATH = ROOT / "src" / "tritonflow" / "isa" / "schemas" / "vortex_rvgpu.yaml"
MACHINE_PATH = ROOT / "src" / "tritonflow" / "isa" / "machines" / "vortex_rvgpu.json"
SNAPSHOT_PATH = ROOT / "third_party" / "vortex" / "SNAPSHOT.md"


def clog2(x: int) -> int:
    return math.ceil(math.log2(x)) if x > 1 else 0


def verify_constants() -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not UPSTREAM_CONFIG_PATH.exists():
        errors.append(f"Missing upstream config: {UPSTREAM_CONFIG_PATH}")
    if not UPSTREAM_TYPES_PATH.exists():
        errors.append(f"Missing upstream types: {UPSTREAM_TYPES_PATH}")
    if not SCHEMA_PATH.exists():
        errors.append(f"Missing schema: {SCHEMA_PATH}")

    if errors:
        return False, errors

    upstream_types_txt = UPSTREAM_TYPES_PATH.read_text(encoding="utf-8")
    upstream_cfg_txt = UPSTREAM_CONFIG_PATH.read_text(encoding="utf-8")
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))

    cfg = schema.get("config", {})
    csrs = schema.get("csr_registers", {})
    data_model = schema.get("data_model", {})

    # 1. TCU CSR Addresses
    # Upstream: VX_CSR_MPM_TCU_TBUF_STALLS = 0xB03, CACHE_HITS = 0xB04, LMEM_READS = 0xB05
    expected_csrs = {
        "TCU_TBUF_STALLS": 0xB03,
        "TCU_TBUF_CACHE_HITS": 0xB04,
        "TCU_LMEM_READS": 0xB05,
    }
    for name, expected_val in expected_csrs.items():
        val = csrs.get(name)
        if val != expected_val:
            errors.append(
                f"CSR {name} mismatch: schema has {hex(val) if val else None}, "
                f"upstream defines {hex(expected_val)}"
            )

    # Verify upstream types file actually declares these
    assert "VX_CSR_MPM_TCU_TBUF_STALLS    = 0xB03" in upstream_types_txt
    assert "VX_CSR_MPM_TCU_TBUF_CACHE_HITS= 0xB04" in upstream_types_txt
    assert "VX_CSR_MPM_TCU_LMEM_READS     = 0xB05" in upstream_types_txt

    # 2. TCU Latency Values
    # Upstream formula:
    # num_threads = 4 (default in pipeline.num_threads)
    # tcu_tc_k = 1 << (clog2(num_threads) // 2) = 2
    # dsp: 1 + 8 + clog2(2 * tcu_tc_k + 1) * 11 = 1 + 8 + 3 * 11 = 42
    # bhf: 4 + clog2(2 * tcu_tc_k + 1) * 3 = 4 + 3 * 3 = 13
    # dpi: 4
    # tfr: 4 (sim) / 5 (synth)
    num_threads = cfg.get("pipeline", {}).get("num_threads", 4)
    tcu_tc_k = 1 << (clog2(num_threads) // 2)
    expected_dsp = 1 + 8 + clog2(2 * tcu_tc_k + 1) * 11
    expected_bhf = 4 + clog2(2 * tcu_tc_k + 1) * 3
    expected_dpi = 4
    expected_tfr_sim = 4

    latencies = cfg.get("tcu", {}).get("latencies", {})
    if latencies.get("dsp") != expected_dsp:
        errors.append(f"TCU latency 'dsp' mismatch: schema={latencies.get('dsp')}, upstream={expected_dsp}")
    if latencies.get("bhf") != expected_bhf:
        errors.append(f"TCU latency 'bhf' mismatch: schema={latencies.get('bhf')}, upstream={expected_bhf}")
    if latencies.get("dpi") != expected_dpi:
        errors.append(f"TCU latency 'dpi' mismatch: schema={latencies.get('dpi')}, upstream={expected_dpi}")
    if latencies.get("tfr") not in (expected_tfr_sim, 5):
        errors.append(f"TCU latency 'tfr' mismatch: schema={latencies.get('tfr')}, upstream={expected_tfr_sim}")

    # 3. TCU Block Count
    # Upstream: VX_CFG_NUM_TCU_BLOCKS = "expr: $VX_CFG_ISSUE_WIDTH" (issue_width=1)
    issue_width = cfg.get("pipeline", {}).get("issue_width", 1)
    num_blocks = cfg.get("tcu", {}).get("num_blocks")
    if num_blocks != issue_width:
        errors.append(f"TCU num_blocks mismatch: schema={num_blocks}, upstream (issue_width)={issue_width}")

    # 4. Scratchpad Bank Count
    # Upstream: VX_CFG_LMEM_NUM_BANKS = "expr: $VX_CFG_NUM_LSU_LANES" (= simd_width = 4)
    simd_width = cfg.get("pipeline", {}).get("simd_width", 4)
    scratch_banks = cfg.get("memory", {}).get("scratchpad_banks")
    if scratch_banks != simd_width:
        errors.append(f"scratchpad_banks mismatch: schema={scratch_banks}, upstream (simd_width)={simd_width}")

    spaces = {s["name"]: s for s in data_model.get("memory_spaces", [])}
    if "scratch" in spaces:
        dm_scratch_banks = spaces["scratch"].get("banks")
        if dm_scratch_banks != simd_width:
            errors.append(
                f"data_model scratch banks mismatch: schema={dm_scratch_banks}, upstream={simd_width}"
            )

    # 5. Cache Line Size
    # Upstream: VX_CFG_L1_LINE_SIZE = "expr: $VX_CFG_MEM_BLOCK_SIZE" = 64
    assert "VX_CFG_MEM_BLOCK_SIZE = 64" in upstream_cfg_txt
    cache_line_bytes = cfg.get("memory", {}).get("cache_line_bytes")
    if cache_line_bytes != 64:
        errors.append(f"cache_line_bytes mismatch: schema={cache_line_bytes}, upstream=64")

    # 6. The vendored snapshot must be the one SNAPSHOT.md records.
    import hashlib
    import json
    import re

    snap = SNAPSHOT_PATH.read_text(encoding="utf-8") if SNAPSHOT_PATH.exists() else ""
    for label, path in (("VX_config.toml", UPSTREAM_CONFIG_PATH), ("VX_types.toml", UPSTREAM_TYPES_PATH)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in snap:
            errors.append(f"{label} sha256 {digest[:16]}... is not recorded in third_party/vortex/SNAPSHOT.md")

    # 7. The machine parameters the cost model reads must match upstream too.
    machine = json.loads(MACHINE_PATH.read_text(encoding="utf-8"))
    mp = {k: v["value"] for k, v in machine.get("params", {}).items()}
    for name, want in (("cache_line_bytes", 64), ("lmem_banks", simd_width)):
        if mp.get(name) != want:
            errors.append(f"machines/vortex_rvgpu.json {name}={mp.get(name)} but upstream implies {want}")
    if machine.get("source", {}).get("config_sha256") != hashlib.sha256(UPSTREAM_CONFIG_PATH.read_bytes()).hexdigest():
        errors.append("machines/vortex_rvgpu.json config_sha256 does not match the vendored VX_config.toml")

    # 8. Prose in the schema description must not restate values the config contradicts.
    description = str(schema.get("description", ""))
    for stale in (r"16 banks", r"0xB08", r"0xB09", r"0xB0A"):
        if re.search(stale, description):
            errors.append(f"schema description still states a superseded value: {stale!r}")

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if any constant differs from upstream")
    parser.parse_args()

    ok, errors = verify_constants()
    if not ok:
        print("VORTEX HARDWARE CONSTANTS CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("Vortex hardware constants OK: all categories (CSRs, latencies, blocks, banks, line size, snapshot, machine file, prose) match upstream VX_config.toml and VX_types.toml exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
