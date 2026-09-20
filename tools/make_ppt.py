#!/usr/bin/env python3
"""Generate TritonFlow hackathon presentation — fully updated for 2026 submission."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ─────────────────────────── Colour Palette ──────────────────────────────────
BG_DARK   = RGBColor(0x0D, 0x11, 0x17)   # slide background
ACCENT    = RGBColor(0x00, 0xD4, 0xFF)   # cyan highlight
GREEN     = RGBColor(0x39, 0xFF, 0x14)   # neon green
YELLOW    = RGBColor(0xFF, 0xD7, 0x00)   # gold
MAGENTA   = RGBColor(0xD6, 0x41, 0x61)   # magenta
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
GREY      = RGBColor(0x88, 0x88, 0x88)
CODE_BG   = RGBColor(0x1E, 0x1E, 0x2E)  # code block bg
DIM_WHITE = RGBColor(0xCC, 0xCC, 0xCC)
ORANGE    = RGBColor(0xFF, 0x7F, 0x00)

W = Inches(13.33)   # 16:9 width
H = Inches(7.5)     # 16:9 height


# ─────────────────────────── Helpers ─────────────────────────────────────────
def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(blank_layout)


def bg(slide, colour: RGBColor = BG_DARK):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def textbox(slide, text: str, left, top, width, height,
            font_size=Pt(18), bold=False, colour=WHITE,
            align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = wrap
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = font_size
    run.font.bold   = bold
    run.font.color.rgb = colour
    run.font.italic = italic
    return txb


def hline(slide, top, colour: RGBColor = ACCENT, thickness: int = 18_000):
    """Draw a full-width horizontal rule."""
    line = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR_TYPE.STRAIGHT
        Inches(0.3), top, Inches(13.0), top,
    )
    line.line.color.rgb = colour
    line.line.width = thickness


def code_box(slide, code_lines: list[str], left, top, width, height, font_size=Pt(9)):
    """Dark-bg mono code block."""
    rect = slide.shapes.add_shape(1, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = CODE_BG
    rect.line.fill.background()

    txb = slide.shapes.add_textbox(
        left + Inches(0.08), top + Inches(0.06),
        width - Inches(0.16), height - Inches(0.12),
    )
    txb.word_wrap = False
    tf = txb.text_frame
    tf.word_wrap = False

    for i, line in enumerate(code_lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line
        run.font.size = font_size
        run.font.color.rgb = DIM_WHITE
        run.font.name = "Consolas"


def table_slide(slide, headers: list[str], rows: list[list[str]],
                left, top, width, col_widths: list[float],
                header_colour=ACCENT, font_size=Pt(10)):
    """Minimal styled table."""
    n_cols = len(headers)
    n_rows = len(rows) + 1
    row_h  = Inches(0.32)
    tbl    = slide.shapes.add_table(n_rows, n_cols, left, top, width, row_h * n_rows).table

    for ci, (hdr, cw) in enumerate(zip(headers, col_widths)):
        tbl.columns[ci].width = Inches(cw)
        cell = tbl.cell(0, ci)
        cell.text = hdr
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0] if p.runs else p.add_run()
        run.font.size  = font_size
        run.font.bold  = True
        run.font.color.rgb = header_colour

    for ri, row in enumerate(rows):
        alt = ri % 2 == 1
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x16, 0x1b, 0x2e) if alt else RGBColor(0x0d, 0x11, 0x17)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.runs[0] if p.runs else p.add_run()
            run.font.size  = font_size
            run.font.color.rgb = WHITE

    return tbl


# ─────────────────────────── Slide Builders ──────────────────────────────────

# ── Slide 1: Title ───────────────────────────────────────────────────────────
def slide_title(prs):
    s = blank_slide(prs)
    bg(s)

    bar = s.shapes.add_shape(1, Inches(0), Inches(0), W, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    textbox(s, "TRITONFLOW", Inches(0.5), Inches(1.0), Inches(12), Inches(1.4),
            font_size=Pt(72), bold=True, colour=ACCENT, align=PP_ALIGN.CENTER)

    textbox(s, "Schema-Driven Multi-ISA AI Accelerator Compiler",
            Inches(0.5), Inches(2.5), Inches(12), Inches(0.7),
            font_size=Pt(26), bold=False, colour=WHITE, align=PP_ALIGN.CENTER)

    tagline = (
        "Triton IR → declarative YAML ISA → bit-accurate execution on any custom accelerator\n"
        "Zero LLVM required  ·  Zero C++ backend rewrites  ·  Powered by torch.compile"
    )
    textbox(s, tagline, Inches(1.0), Inches(3.3), Inches(11), Inches(0.9),
            font_size=Pt(15), colour=GREY, align=PP_ALIGN.CENTER, italic=True)

    hline(s, Inches(4.35))

    textbox(s, 'torch.compile(model, backend="tritonflow")',
            Inches(2.5), Inches(4.6), Inches(8), Inches(0.6),
            font_size=Pt(18), bold=True, colour=GREEN, align=PP_ALIGN.CENTER)

    # Key metrics bar
    stats = [
        ("182", "Tests Passing"),
        ("0.000e+00", "Max GPU Error"),
        ("18.6×", "Compute Speedup"),
        ("4 ops", "vecadd Emitted"),
    ]
    for i, (val, label) in enumerate(stats):
        lx = Inches(0.5 + i * 3.1)
        textbox(s, val, lx, Inches(5.45), Inches(2.8), Inches(0.55),
                font_size=Pt(22), bold=True, colour=ACCENT, align=PP_ALIGN.CENTER)
        textbox(s, label, lx, Inches(6.0), Inches(2.8), Inches(0.4),
                font_size=Pt(11), colour=GREY, align=PP_ALIGN.CENTER)

    textbox(s, "IICT CompilerTech Hackathon 2026  ·  Team TritonFlow",
            Inches(0), Inches(7.05), W, Inches(0.35),
            font_size=Pt(11), colour=GREY, align=PP_ALIGN.CENTER)


# ── Slide 2: Problem ─────────────────────────────────────────────────────────
def slide_problem(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "🛑  The Problem", Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(32), bold=True, colour=MAGENTA)
    hline(s, Inches(1.05), colour=MAGENTA)

    textbox(s, "Hardware innovation is outpacing software ecosystems.",
            Inches(0.5), Inches(1.2), Inches(12), Inches(0.5),
            font_size=Pt(20), bold=True, colour=WHITE)

    pain = (
        "When a startup or researcher designs a new AI accelerator (ASIC / FPGA / RISC-V GPGPU), "
        "integrating it into the PyTorch ecosystem today requires:"
    )
    textbox(s, pain, Inches(0.5), Inches(1.85), Inches(12.3), Inches(0.7),
            font_size=Pt(15), colour=DIM_WHITE)

    steps = [
        "①  Write a custom PyTorch C++ Device Extension from scratch  (weeks)",
        "②  Fork the Triton compiler and modify MLIR lowering passes  (months)",
        "③  Build a custom LLVM backend and codegen pipeline          (months)",
        "④  Write a hardware runtime / driver layer                   (weeks)",
    ]
    for i, step in enumerate(steps):
        textbox(s, step, Inches(0.9), Inches(2.7 + i * 0.55), Inches(11.5), Inches(0.5),
                font_size=Pt(15), colour=YELLOW)

    textbox(s, "⚠  This software moat takes 6–12 months and kills hardware innovation at the root.",
            Inches(0.5), Inches(5.1), Inches(12), Inches(0.55),
            font_size=Pt(17), bold=True, colour=MAGENTA)

    textbox(s,
            "Open accelerators like Vortex RVGPU have exceptional hardware — "
            "but zero PyTorch integration. TritonFlow closes that gap in 2 weeks.",
            Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.7),
            font_size=Pt(14), colour=DIM_WHITE, italic=True)


# ── Slide 3: Solution ────────────────────────────────────────────────────────
def slide_solution(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "💡  The Solution: Declarative ISA Compilation",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(32), bold=True, colour=ACCENT)
    hline(s, Inches(1.05))

    desc = (
        "TritonFlow generates a complete, production-grade PyTorch backend from a simple "
        "declarative YAML description of your hardware's Instruction Set Architecture.\n\n"
        "You describe what your chip can do — TritonFlow handles parsing, recognition, "
        "instruction selection, emulation, and parity verification automatically."
    )
    textbox(s, desc, Inches(0.5), Inches(1.25), Inches(12.3), Inches(1.3),
            font_size=Pt(16), colour=WHITE)

    yaml_lines = [
        "# vortex_rvgpu.yaml — real open RISC-V GPGPU in YAML",
        "name: vortex_rvgpu",
        "instructions:",
        "  - name: TCU_WGMMA_SP32   kind: mac",
        "    tile: [32, 32]          sparsity: 2:4",
        "    cost: 0.15 * m * n * k / (32*32)   # 18.6x vs dense",
        "    constraints:",
        "      - all_of(m % 32 == 0, n % 32 == 0)",
        "  - name: DXA_COPY_2D      kind: memory",
        "    cost: 0.15 * words      # TMA-style async bulk DMA",
        "    constraints:",
        "      - all_of(stride[1] == 1, in_bounds_all())",
        "  - name: VADD             kind: elementwise",
        "    cost: 0.05 * words      # 32-lane SIMT vector ALU",
    ]
    code_box(s, yaml_lines, Inches(0.5), Inches(2.7), Inches(5.9), Inches(4.5))

    benefits = [
        "✔  Real torch.compile backend — no API changes",
        "✔  Triton IR parsed without LLVM build",
        "✔  Fail-closed: 0.0% silent miscompilations",
        "✔  Bit-accurate C++ + NumPy dual emulator",
        "✔  Swap YAML → instant retarget + cost diff",
        "✔  182 passing tests, 3 live ISA targets",
        "✔  Works on real open hardware (Vortex RVGPU)",
    ]
    for i, b in enumerate(benefits):
        textbox(s, b, Inches(6.6), Inches(2.7 + i * 0.62), Inches(6.5), Inches(0.55),
                font_size=Pt(15), colour=GREEN)


# ── Slide 4: Architecture ────────────────────────────────────────────────────
def slide_architecture(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "🏗️  How TritonFlow Works — End-to-End Pipeline",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(30), bold=True, colour=ACCENT)
    hline(s, Inches(1.05))

    boxes = [
        ("PyTorch\ntorch.compile(…)", Inches(0.3),  MAGENTA),
        ("Track A\nParser\n(TTIR → SSA)", Inches(2.95), ACCENT),
        ("Track C\nRecognition\n(Pointers & Loops)", Inches(5.6), YELLOW),
        ("Track B\nSelector\n(ISA Matching)", Inches(8.25), GREEN),
        ("Track D\nEmulator\n(C++ / NumPy)", Inches(10.9), MAGENTA),
    ]
    bw, bh = Inches(2.4), Inches(1.8)
    by = Inches(1.8)

    for label, bx, colour in boxes:
        rect = s.shapes.add_shape(1, bx, by, bw, bh)
        rect.fill.solid()
        rect.fill.fore_color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        rect.line.color.rgb = colour
        rect.line.width = 36000
        txb = s.shapes.add_textbox(bx + Inches(0.05), by + Inches(0.1),
                                    bw - Inches(0.1), bh - Inches(0.2))
        txb.word_wrap = True
        tf = txb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = label
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = colour

    for i in range(len(boxes) - 1):
        ax = boxes[i][1] + bw
        ay = by + bh / 2
        line = s.shapes.add_connector(1, ax, ay, boxes[i + 1][1], ay)
        line.line.color.rgb = GREY
        line.line.width = 18000

    yaml_box_x, yaml_box_y = Inches(8.25), Inches(4.05)
    rect2 = s.shapes.add_shape(1, yaml_box_x, yaml_box_y, Inches(2.4), Inches(0.9))
    rect2.fill.solid()
    rect2.fill.fore_color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    rect2.line.color.rgb = GREEN
    rect2.line.width = 18000
    textbox(s, "ISA YAML Schema\n(Declarative Config)", yaml_box_x + Inches(0.05),
            yaml_box_y + Inches(0.05), Inches(2.3), Inches(0.8),
            font_size=Pt(10), colour=GREEN, align=PP_ALIGN.CENTER)
    line2 = s.shapes.add_connector(1, yaml_box_x + bw / 2, yaml_box_y,
                                   yaml_box_x + bw / 2, by + bh)
    line2.line.color.rgb = GREEN
    line2.line.width = 18000

    stage_descs = [
        "Intercepts\ntorch.compile",
        "Parses TTIR\ninto SSA graph",
        "Affine descriptor\nrecovery + elision",
        "Greedy min-cost\ninstruction select",
        "Bit-accurate\nexecution + parity",
    ]
    for (_, bx, colour), desc in zip(boxes, stage_descs):
        textbox(s, desc, bx, by + bh + Inches(0.1), bw, Inches(0.75),
                font_size=Pt(10), colour=GREY, align=PP_ALIGN.CENTER)

    textbox(s,
            "Key invariant: every stage is independently testable — "
            "182 tests verify contracts end-to-end.",
            Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.45),
            font_size=Pt(13), colour=DIM_WHITE, italic=True, align=PP_ALIGN.CENTER)


# ── Slide 5: Pipeline demo (TTIR parsing) ────────────────────────────────────
def slide_pipeline_demo(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "⚙️  Demo 1: PyTorch → Triton JIT → MLIR",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(28), bold=True, colour=YELLOW)
    hline(s, Inches(1.05), colour=YELLOW)

    py_code = [
        "# User writes standard PyTorch — zero API changes",
        "@torch.compile(backend='tritonflow')",
        "def matmul(x, y):",
        "    return torch.matmul(x, y)  # (128×64) @ (64×128) → (128×128)",
        "",
        "output = matmul(a, b)   # TritonFlow intercepts here",
    ]
    textbox(s, "1. PyTorch Frontend", Inches(0.5), Inches(1.2), Inches(6), Inches(0.4),
            font_size=Pt(13), bold=True, colour=GREEN)
    code_box(s, py_code, Inches(0.5), Inches(1.65), Inches(5.9), Inches(2.1))

    triton_code = [
        "# Auto-lowered Triton JIT kernel",
        "@triton.jit",
        "def matmul_kernel(a_ptr, b_ptr, c_ptr,",
        "                  M, N, K, stride_am, ...,",
        "                  BM: tl.constexpr = 64,",
        "                  BN: tl.constexpr = 64,",
        "                  BK: tl.constexpr = 32):",
        "    for k in range(0, K, BK):",
        "        a = tl.load(a_ptrs)      # [64×32] tile",
        "        b = tl.load(b_ptrs)      # [32×64] tile",
        "        acc = tl.dot(a, b, acc)  # MMA",
        "    tl.store(c_ptrs, acc)",
    ]
    textbox(s, "2. Triton JIT Kernel", Inches(6.7), Inches(1.2), Inches(6), Inches(0.4),
            font_size=Pt(13), bold=True, colour=ACCENT)
    code_box(s, triton_code, Inches(6.7), Inches(1.65), Inches(6.1), Inches(2.5))

    mlir_code = [
        "// Extracted MLIR Triton Dialect (TTIR) — 63 SSA ops",
        "module {",
        "  tt.func @matmul_kernel(%a_ptr: !tt.ptr<f32>, ...) {",
        "    %acc_25:3 = scf.for %_k = %c0 to %K step %c1",
        "        iter_args(%a_ptrs_34 = %a_ptrs, ...) {",
        "      %a = tt.load %a_ptrs_34 : tensor<64x32x!tt.ptr<f32>>",
        "      %b = tt.load %b_ptrs_35 : tensor<32x64x!tt.ptr<f32>>",
        "      %acc_37 = tt.dot %a, %b, %acc_36",
        "               : tensor<64x32xf32> * tensor<32x64xf32>",
        "               -> tensor<64x64xf32>",
        "      scf.yield %a_ptrs_40, %b_ptrs_43, %acc_37",
        "    }",
        "  }",
        "}",
    ]
    textbox(s, "3. Extracted MLIR (TTIR) — TritonFlow takes over here",
            Inches(0.5), Inches(3.9), Inches(12.3), Inches(0.4),
            font_size=Pt(13), bold=True, colour=MAGENTA)
    code_box(s, mlir_code, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.85))


# ── Slide 6: NEW — Address Math Elision (18 → 4 ops) ────────────────────────
def slide_elision(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "✂️  Address Math Elision: 18 TTIR Ops → 4 Hardware Instructions",
            Inches(0.5), Inches(0.25), Inches(12.5), Inches(0.65),
            font_size=Pt(26), bold=True, colour=ACCENT)
    hline(s, Inches(1.05))

    textbox(s,
            "For t0_vecadd (1024 elements): the descriptor engine absorbs 14 pointer-arithmetic ops "
            "into hardware DMA descriptors. Only 4 semantic instructions are emitted.",
            Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.6),
            font_size=Pt(14), colour=DIM_WHITE, italic=True)

    before_lines = [
        "# TTIR — 18 SSA operations",
        "%c1024_i32 = arith.constant 1024        ; EPI → ELIDED",
        "%pid       = tt.get_program_id          ; EPI → ELIDED",
        "%offs      = arith.muli %pid, %c1024    ; EPI → ELIDED",
        "%offs_0    = tt.make_range 0:1024       ; EPI → ELIDED",
        "%offs_1    = tt.splat %offs             ; EPI → ELIDED",
        "%offs_2    = arith.addi %offs_0, %offs_1 ; EPI → ELIDED",
        "%x_4       = tt.addptr %x_ptr, %offs_2  ; EPI → ELIDED",
        "... (8 more addr-math ops)              ; EPI → ELIDED",
        "%x_5       = tt.load %x_4              ; ← KEEP",
        "%y_7       = tt.load %y_6              ; ← KEEP",
        "%2         = arith.addf %x_5, %y_7     ; ← KEEP",
        "tt.store %out_10, %2                   ; ← KEEP",
    ]
    textbox(s, "Before (TTIR — 18 ops):", Inches(0.4), Inches(1.85), Inches(6.2), Inches(0.4),
            font_size=Pt(13), bold=True, colour=MAGENTA)
    code_box(s, before_lines, Inches(0.4), Inches(2.3), Inches(6.2), Inches(4.9))

    after_lines = [
        "# Emitted hardware program — 4 ops",
        "[00] DMA1D  src=global:%x_ptr",
        "            tile=[1024]  stride=[1]",
        "            offset=[1024*%pid]",
        "            cost=1024.0 cycles",
        "",
        "[01] DMA1D  src=global:%y_ptr",
        "            tile=[1024]  stride=[1]",
        "            offset=[1024*%pid]",
        "            cost=1024.0 cycles",
        "",
        "[02] EPI    op=add  in0=%x_5  in1=%y_7",
        "            cost=512.0 cycles",
        "",
        "[03] DMA1D  dst=global:%out_ptr  value=%2",
        "            cost=1024.0 cycles",
    ]
    textbox(s, "After (Hardware ISA — 4 ops):", Inches(6.9), Inches(1.85), Inches(6.1), Inches(0.4),
            font_size=Pt(13), bold=True, colour=GREEN)
    code_box(s, after_lines, Inches(6.9), Inches(2.3), Inches(6.1), Inches(4.9))

    # Arrow
    arr = s.shapes.add_connector(1, Inches(6.6), Inches(4.75), Inches(6.85), Inches(4.75))
    arr.line.color.rgb = ACCENT
    arr.line.width = 36000

    textbox(s,
            "Coalescing: 100.0% (128B req = 128B transacted, 4 transactions)  ·  Bank conflicts: 0",
            Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3),
            font_size=Pt(12), colour=GREEN, italic=True, align=PP_ALIGN.CENTER)


# ── Slide 7: Pre-ISA Semantic Recovery ──────────────────────────────────────
def slide_pre_isa(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "🔍  Semantic Recovery: α / β Attribute Splitting",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(28), bold=True, colour=YELLOW)
    hline(s, Inches(1.05), colour=YELLOW)

    desc = (
        "Before choosing any hardware instruction, TritonFlow analyses TTIR to recover loop induction "
        "variables, build SSA def-use chains, and extract hardware-agnostic affine access descriptors "
        "(α = computation attributes, β = addressing attributes)."
    )
    textbox(s, desc, Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.75),
            font_size=Pt(14), colour=DIM_WHITE, italic=True)

    headers = ["SSA Value", "Source Op", "Idiom", "Recovered Semantic Descriptor"]
    rows = [
        ["%_k",        "scf.for",    "control_flow",
         "loop induction: 0 → K, step 32 | iter_args: %a_ptrs, %b_ptrs, %acc"],
        ["%a",         "tt.load",    "memory",
         "base=%a_ptr, sizes=[64,32], strides=[stride_am, stride_ak], offset=[64*%pid_m*stride_am, 0]"],
        ["%b",         "tt.load",    "memory",
         "base=%b_ptr, sizes=[32,64], strides=[stride_bk, stride_bn], offset=[64*%pid_n*stride_bn, 0]"],
        ["%acc_37",    "tt.dot",     "mac",
         "tile=(64, 64, 32), precision=tf32, acc_dtype=f32, order=k_major_sequential"],
        ["%a_ptrs_38", "arith.muli", "elementwise → SUBSUMED",
         "stride advance: stride_ak × 32  →  folded into DMA descriptor"],
        ["%a_ptrs_40", "tt.addptr",  "elementwise → SUBSUMED",
         "pointer increment: %a_ptrs_34 + %splat_39  →  folded into DMA descriptor"],
        ["(store)",    "tt.store",   "memory",
         "base=%c_ptr, sizes=[64,64], strides=[stride_cm, stride_cn], offset=[64*pid_m*scm + 64*pid_n*scn, 0]"],
    ]
    table_slide(s, headers, rows,
                Inches(0.3), Inches(2.05), Inches(12.7),
                col_widths=[1.2, 1.5, 1.8, 8.0])


# ── Slide 8: Lowering diff ───────────────────────────────────────────────────
def slide_lowering_diff(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "⚡  Lowering Diff: Abstract TTIR → Vortex RVGPU Hardware",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(28), bold=True, colour=YELLOW)
    hline(s, Inches(1.05), colour=YELLOW)

    textbox(s, "Abstract TTIR ops  →  concrete hardware instructions (matmul inner loop)",
            Inches(0.5), Inches(1.15), Inches(12), Inches(0.4),
            font_size=Pt(15), colour=DIM_WHITE, italic=True)

    diff_lines = [
        "--- Pre-ISA Abstract TTIR",
        "+++ Target: VORTEX_RVGPU (RISC-V SIMT GPGPU)",
        "@@ Inner Compute Loop Lowering @@",
        "- %a = tt.load %a_ptrs_34 : tensor<64x32x!tt.ptr<f32>>",
        "+ LDG   src=global:%a_ptr[tile=[64,32], strides=[stride_am,stride_ak]]  ; cost=1024.0",
        "- %b = tt.load %b_ptrs_35 : tensor<32x64x!tt.ptr<f32>>",
        "+ LDG   src=global:%b_ptr[tile=[32,64], strides=[stride_bk,stride_bn]]  ; cost=1024.0",
        "- %acc_37 = tt.dot %a, %b, %acc_36",
        "-          : tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>",
        "+ TCU_WGMMA_SP32  a=%a b=%b acc=%acc   ; cost=19.2  (2:4 structured sparsity, 2.0×)",
        "- %a_ptrs_38 = arith.muli %sak, %c32_i32",
        "+ VMUL  dst=%a_ptrs_38, src0=%sak, src1=%c32   ; cost=0.1  (SIMT vector multiplier)",
        "- %a_ptrs_40 = tt.addptr %a_ptrs_34, %splat_39",
        "+ VADD  dst=%a_ptrs_40, src0=%a_ptrs_34, src1=%splat ; cost=102.4  (32-lane vector ALU)",
    ]
    code_box(s, diff_lines, Inches(0.3), Inches(1.65), Inches(12.7), Inches(3.55))

    textbox(s, "TRITONFLOW1 vs VORTEX_RVGPU — instruction delta",
            Inches(0.3), Inches(5.35), Inches(12.7), Inches(0.35),
            font_size=Pt(13), bold=True, colour=MAGENTA)

    isa_rows = [
        ["tt.load (×2)",  "DMA1D  (cost=2048.0)", "LDG    (cost=1024.0)", "−50%    ✔"],
        ["tt.dot",        "MAC16  (cost=358.4)",  "TCU_WGMMA_SP32 (19.2)", "−18.6× ✔"],
        ["arith.muli",    "EPI    (cost=0.5)",    "VMUL   (cost=0.1)",    "−5×     ✔"],
        ["tt.addptr",     "EPI    (cost=1024.0)", "VADD   (cost=102.4)",  "−10×    ✔"],
    ]
    table_slide(s, ["TTIR Op", "TRITONFLOW1", "VORTEX_RVGPU", "Speedup"],
                isa_rows, Inches(0.3), Inches(5.75), Inches(12.7),
                col_widths=[2.5, 3.2, 3.5, 3.0], font_size=Pt(10))


# ── Slide 9: 3-way comparison ────────────────────────────────────────────────
def slide_3way_comparison(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "📊  Demo 2: 3-Way Architectural Comparison (Live)",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(28), bold=True, colour=YELLOW)
    hline(s, Inches(1.05), colour=YELLOW)

    textbox(s,
            "Same TTIR kernel, same Python code — three radically different hardware targets, "
            "zero compiler C++ rewritten.",
            Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.45),
            font_size=Pt(14), colour=DIM_WHITE, italic=True)

    headers = ["Dimension", "TRITONFLOW1\n(Systolic ASIC)", "TRITONFLOW2\n(Banked ASIC)", "VORTEX_RVGPU\n(RISC-V GPGPU)"]
    rows = [
        ["Hardware Target",       "Flat Systolic Accelerator",    "Multi-bank Domain ASIC",      "RISC-V SIMT GPGPU (open HW)"],
        ["Primary Compute",       "MAC16 — 16×16 systolic array", "OPU32 — 32×32 outer product", "TCU_WGMMA_SP32 — sparse WG-MMA"],
        ["Memory Engine",         "DMA1D / DMA2D (flat)",         "LDS2D (16-bank interleaved)", "DXA async bulk DMA (1D–5D)"],
        ["Sparsity Support",      "Dense only  (1.0×)",           "Dense only  (1.0×)",          "2:4 Structured Sparsity (2.0×)"],
        ["Micro-arch Gate",       "Static FIFO Pipeline",         "Bank Conflict Arbiter",       "Warpgroup Lockstep Gate"],
        ["Compute Cost / tile",   "358.4 cycles (MAC16)",         "70.4 cycles (OPU32)",         "★ 19.2 cycles  (18.6× faster!)"],
        ["Total Kernel Cost",     "35,773 cycles",                "28,402 cycles  (−20.6%)",     "★ 6,838 cycles  (5.2× speedup)"],
        ["YAML Lines to Target",  "< 80 lines",                   "< 100 lines",                 "< 200 lines (full TCU + DXA)"],
    ]
    table_slide(s, headers, rows,
                Inches(0.3), Inches(1.75), Inches(12.7),
                col_widths=[2.6, 2.8, 3.0, 3.9], font_size=Pt(10))

    textbox(s,
            "★  Swap one YAML file → compiler instantly re-targets and reports the full cost delta.  "
            "A second ISA is the only falsifiable test of declarative compilation.",
            Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.45),
            font_size=Pt(13), colour=GREEN, italic=True)


# ── Slide 10: Instruction Selection Audit ───────────────────────────────────
def slide_instruction_audit(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "🔬  Demo 3: Cost-Driven Instruction Selection Audit",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(28), bold=True, colour=YELLOW)
    hline(s, Inches(1.05), colour=YELLOW)

    desc = (
        "For every operation the compiler evaluates ALL schema candidates against fail-closed "
        "constraint predicates, models execution cost exactly, and greedily selects the "
        "cheapest admissible candidate. Every decision is deterministic and fully logged."
    )
    textbox(s, desc, Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.65),
            font_size=Pt(14), colour=DIM_WHITE, italic=True)

    headers = ["Candidate", "Constraint Predicate", "Verdict", "Modeled Cost", "Compiler Decision"]
    rows = [
        ["TCU_WMMA16",     "all_of(m % 16 == 0, n % 16 == 0)",         "✔ PASS", "204.8 MACs", "Rejected (higher cost)"],
        ["TCU_WGMMA32",    "all_of(m % 32 == 0, n % 32 == 0)",         "✔ PASS", "32.0 MACs",  "Rejected (higher cost)"],
        ["TCU_WGMMA_SP32", "all_of(m % 32 == 0, 2:4 sparsity)",       "✔ PASS", "19.2 MACs",  "★ SELECTED (lowest cost)"],
        ["TCU_WGMMA_MXFP8","all_of(m % 32 == 0, block_scale == 32)",   "✔ PASS", "25.6 MACs",  "Rejected (higher cost)"],
        ["TCU_MMA32",      "all_of(m % 32 == 0, n % 32 == 0)",         "✔ PASS", "32.0 MACs",  "Rejected (higher cost)"],
        ["FEDP_DOTPROD",   "lane_width == 32",                          "✖ FAIL", "—",          "Rejected (inadmissible)"],
    ]
    table_slide(s, headers, rows,
                Inches(0.3), Inches(2.0), Inches(12.7),
                col_widths=[2.1, 3.5, 1.3, 1.8, 3.6],
                header_colour=YELLOW, font_size=Pt(10))

    textbox(s,
            "✔  TCU_WGMMA_SP32 selected — satisfies all tile predicates, lowest valid cost "
            "(19.2 vs 32.0 MACs). 2:4 structured sparsity gives 2.0× effective throughput.",
            Inches(0.5), Inches(4.45), Inches(12.3), Inches(0.55),
            font_size=Pt(14), bold=True, colour=GREEN)

    code_lines = [
        "# Every selection decision is deterministic, auditable, and logged",
        "# vortex_rvgpu  |  tt.dot  |  tile=(64, 64, 32)",
        "Candidate TCU_WMMA16      admissible=True   cost=204.8   rejected",
        "Candidate TCU_WGMMA32     admissible=True   cost= 32.0   rejected",
        "Candidate TCU_WGMMA_SP32  admissible=True   cost= 19.2   ★ SELECTED",
        "Candidate TCU_WGMMA_MXFP8 admissible=True   cost= 25.6   rejected",
        "Candidate FEDP_DOTPROD    admissible=False               rejected",
    ]
    code_box(s, code_lines, Inches(0.3), Inches(5.1), Inches(12.7), Inches(2.1))


# ── Slide 11: Fail-Closed Demo ───────────────────────────────────────────────
def slide_fail_closed(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "🛑  Demo 5: Fail-Closed Integrity — Honest Refusal on t3_modulo",
            Inches(0.5), Inches(0.25), Inches(12.5), Inches(0.65),
            font_size=Pt(26), bold=True, colour=MAGENTA)
    hline(s, Inches(1.05), colour=MAGENTA)

    left_lines = [
        "# t3_modulo — unstructured pointer arithmetic",
        "@triton.jit",
        "def modulo_kernel(x_ptr, out_ptr, N, ...)",
        "    pid = tl.program_id(0)",
        "    offs = (pid * BL + tl.arange(0, BL)) % N",
        "    #              ↑ non-affine modulo wrap!",
        "    x = tl.load(x_ptr + offs)  # irregular",
        "    tl.store(out_ptr + offs, x)",
    ]
    textbox(s, "Input: Non-Affine Kernel", Inches(0.4), Inches(1.2), Inches(5.8), Inches(0.4),
            font_size=Pt(13), bold=True, colour=MAGENTA)
    code_box(s, left_lines, Inches(0.4), Inches(1.65), Inches(5.8), Inches(2.9))

    right_lines = [
        "# TritonFlow compiler output — honest refusal",
        "UNSUPPORTED  op=arith.remsi  at %x_ptr",
        "  reason: unstructured access — modulo pointer",
        "          arithmetic cannot be described by an",
        "          affine stride/offset decomposition.",
        "UNSUPPORTED  op=tt.load      at %x_4",
        "  reason: pointer operand unavailable",
        "",
        "→  Compiler triggers PyTorch eager fallback",
        "→  Zero silent miscompilations",
        "→  Complies with contract FR-025",
    ]
    textbox(s, "Compiler Output: Fail-Closed", Inches(6.7), Inches(1.2), Inches(6.2), Inches(0.4),
            font_size=Pt(13), bold=True, colour=GREEN)
    code_box(s, right_lines, Inches(6.7), Inches(1.65), Inches(6.2), Inches(2.9))

    metrics = [
        ("0.0%", "Silent Miscompilation Risk"),
        ("2", "UNSUPPORTED markers emitted"),
        ("FR-025", "Contract specification satisfied"),
        ("100%", "Negative control test pass rate"),
    ]
    for i, (val, label) in enumerate(metrics):
        col = i % 2
        row = i // 2
        lx = Inches(0.5 + col * 6.4)
        ty = Inches(4.9 + row * 1.1)
        textbox(s, val, lx, ty, Inches(6.0), Inches(0.6),
                font_size=Pt(36), bold=True, colour=MAGENTA if i == 0 else GREEN,
                align=PP_ALIGN.CENTER)
        textbox(s, label, lx, ty + Inches(0.6), Inches(6.0), Inches(0.4),
                font_size=Pt(12), colour=DIM_WHITE, align=PP_ALIGN.CENTER)


# ── Slide 12: Results & Verification ────────────────────────────────────────
def slide_results(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "✅  Results & Verification",
            Inches(0.5), Inches(0.25), Inches(12), Inches(0.65),
            font_size=Pt(32), bold=True, colour=GREEN)
    hline(s, Inches(1.05), colour=GREEN)

    metrics = [
        ("182 / 182",     "Contract & Unit Tests Passing\n(100% pass rate, 17.45 s run)", GREEN),
        ("0.000e+00",     "Max Abs Error vs NVIDIA GPU\n(Bit-accurate numerical parity)", GREEN),
        ("18.6×",         "Compute speedup: Vortex vs TF1\n(TCU_WGMMA_SP32 vs MAC16)", YELLOW),
        ("5.2×",          "Total kernel speedup\n(6,838 vs 35,773 cycles)", YELLOW),
        ("< 200 lines\nYAML", "Full Vortex RVGPU backend\n(Zero LLVM / C++ required)", ACCENT),
        ("4 ops",         "vecadd emitted instructions\n(down from 18 — full elision)", ACCENT),
    ]

    for i, (val, label, colour) in enumerate(metrics):
        col = i % 3
        row = i // 3
        lx = Inches(0.4 + col * 4.3)
        ty = Inches(1.4 + row * 2.3)
        textbox(s, val, lx, ty, Inches(4.0), Inches(0.9),
                font_size=Pt(38), bold=True, colour=colour, align=PP_ALIGN.CENTER)
        textbox(s, label, lx, ty + Inches(0.95), Inches(4.0), Inches(0.85),
                font_size=Pt(12), colour=DIM_WHITE, align=PP_ALIGN.CENTER)

    textbox(s,
            "Verified live on real NVIDIA GeForce GTX 1650 · "
            "3 production ISA targets · 13 test files · pybind11 C++ fast path",
            Inches(0.5), Inches(6.85), Inches(12.3), Inches(0.4),
            font_size=Pt(12), colour=GREY, italic=True, align=PP_ALIGN.CENTER)


# ── Slide 13: Strategic Positioning ─────────────────────────────────────────
def slide_strategic(prs):
    s = blank_slide(prs)
    bg(s)

    textbox(s, "🎯  Strategic Positioning: What We Claim & What We Don't",
            Inches(0.5), Inches(0.25), Inches(12.5), Inches(0.65),
            font_size=Pt(26), bold=True, colour=ACCENT)
    hline(s, Inches(1.05))

    textbox(s, "What TritonFlow claims (and proves):",
            Inches(0.5), Inches(1.2), Inches(12), Inches(0.45),
            font_size=Pt(16), bold=True, colour=GREEN)

    claims = [
        "✔  Triton IR → declarative YAML ISA backend is technically feasible and ships today",
        "✔  A second ISA (Vortex RVGPU) is the only falsifiable proof — we have three",
        "✔  Bit-accurate numerical parity with real NVIDIA GPU (0.000e+00 max error)",
        "✔  Fail-closed: unstructured kernels get honest UNSUPPORTED, not corrupt output",
        "✔  torch.compile front-door integration — zero user API changes needed",
    ]
    for i, c in enumerate(claims):
        textbox(s, c, Inches(0.8), Inches(1.75 + i * 0.52), Inches(11.7), Inches(0.45),
                font_size=Pt(14), colour=GREEN)

    textbox(s, "What TritonFlow does NOT claim:",
            Inches(0.5), Inches(4.5), Inches(12), Inches(0.45),
            font_size=Pt(16), bold=True, colour=MAGENTA)

    non_claims = [
        "✖  CUDA hijack or interception of vendor CUDA kernels (closed libraries / ptxas)",
        "✖  Autotune-for-cuBLAS performance — not the goal",
        "✖  'Beyond ACT' — ACT (OOPSLA 2026) pioneered declarative schemas for XLA-HLO;",
        "    TritonFlow extends that thesis to the modern torch.compile + Triton IR stack",
    ]
    for i, nc in enumerate(non_claims):
        textbox(s, nc, Inches(0.8), Inches(5.05 + i * 0.5), Inches(11.7), Inches(0.45),
                font_size=Pt(14), colour=MAGENTA)

    textbox(s,
            '"A second ISA is the only falsifiable test of declarative compilation." — TritonFlow thesis',
            Inches(1.0), Inches(7.05), Inches(11.3), Inches(0.35),
            font_size=Pt(13), colour=ACCENT, italic=True, align=PP_ALIGN.CENTER)


# ── Slide 14: Closing ────────────────────────────────────────────────────────
def slide_closing(prs):
    s = blank_slide(prs)
    bg(s)

    bar = s.shapes.add_shape(1, Inches(0), Inches(0), W, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    textbox(s, "TritonFlow", Inches(0.5), Inches(0.9), Inches(12), Inches(1.3),
            font_size=Pt(72), bold=True, colour=ACCENT, align=PP_ALIGN.CENTER)

    textbox(s, "Democratising AI Hardware Design",
            Inches(0.5), Inches(2.25), Inches(12), Inches(0.7),
            font_size=Pt(28), colour=WHITE, align=PP_ALIGN.CENTER)

    textbox(s,
            "From YAML spec to a running torch.compile backend in minutes — not months.",
            Inches(1.5), Inches(3.1), Inches(10), Inches(0.65),
            font_size=Pt(18), colour=GREY, align=PP_ALIGN.CENTER, italic=True)

    hline(s, Inches(4.1))

    links = [
        "🔗  github.com/Incharajayaram/segfault-tritonflow",
        "⚡  PYTHONPATH=src python3 tritonflow_demo.py",
        "🔬  PYTHONPATH=src python3 tools/demo_live.py",
        "📊  PYTHONPATH=src python3 demo_diff.py",
    ]
    for i, link in enumerate(links):
        textbox(s, link, Inches(2.0), Inches(4.4 + i * 0.6), Inches(9.0), Inches(0.52),
                font_size=Pt(15), colour=GREEN, align=PP_ALIGN.CENTER)

    textbox(s, "IICT CompilerTech Hackathon 2026  ·  Team TritonFlow",
            Inches(0), Inches(7.05), W, Inches(0.35),
            font_size=Pt(11), colour=GREY, align=PP_ALIGN.CENTER)


# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    prs = new_prs()

    slide_title(prs)            # 1  Title + live metrics bar
    slide_problem(prs)          # 2  The problem
    slide_solution(prs)         # 3  The solution
    slide_architecture(prs)     # 4  Full pipeline architecture
    slide_pipeline_demo(prs)    # 5  PyTorch → TTIR
    slide_elision(prs)          # 6  NEW: address math elision (18→4 ops)
    slide_pre_isa(prs)          # 7  α/β semantic recovery
    slide_lowering_diff(prs)    # 8  Lowering diff TTIR → Vortex
    slide_3way_comparison(prs)  # 9  3-way ISA comparison with real cycle #s
    slide_instruction_audit(prs)  # 10 Greedy cost-driven selection audit
    slide_fail_closed(prs)      # 11 NEW: fail-closed integrity demo
    slide_results(prs)          # 12 Updated: 182 tests, 6 key metrics
    slide_strategic(prs)        # 13 NEW: strategic positioning
    slide_closing(prs)          # 14 Closing

    out = Path(__file__).resolve().parent.parent / "TritonFlow_Hackathon.pptx"
    prs.save(out)
    print(f"✔  Saved: {out}  ({out.stat().st_size // 1024} KB,  14 slides)")


if __name__ == "__main__":
    main()
