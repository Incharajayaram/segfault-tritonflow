# TritonFlow: Schema-Driven Multi-ISA AI Accelerator Compiler

TritonFlow is a declarative, schema-driven compiler backend that enables rapid retargeting of emerging AI accelerators directly from PyTorch and Triton IR (TTIR). Instead of rewriting tens of thousands of lines of C++ LLVM backend passes for every new tape-out or chip architecture, TritonFlow separates **compiler mechanics** from **hardware specifications**: simply provide a declarative YAML schema defining instruction semantics, constraints, and cost models.

---

## Key Highlights

- **Declarative ISA Schemas**: Target architectures are expressed entirely in YAML descriptions specifying instruction opcodes, memory bank properties, constraint predicates, and cost functions.
- **Fail-Closed Verification**: The compiler mathematically refuses to lower code it cannot prove admissible, eliminating silent numerical corruption and falling back cleanly to eager PyTorch execution.
- **Multi-ISA Retargetability**: Lowers the exact same high-level Triton IR into drastically different chip designs:
  - **TRITONFLOW1**: 16×16 Flat Systolic Array (DMA1D / DMA2D).
  - **TRITONFLOW2**: 32×32 Outer Product Unit (OPU) with 16-bank conflict arbitration.
  - **VORTEX_RVGPU**: Open-source RISC-V SIMT GPGPU with 2:4 structured sparsity and coalesced bulk async transfers (grounded in official Vortex hardware RTL configs).
  - **EDGE_NPU**: Microcontroller-class Neural Processing Unit (e.g. ARM Ethos-U). Integer-only SIMD, no FP32, unified global SRAM (no scratchpad).
- **Native PyTorch Integration**: Registers seamlessly as a  backend, intercepting computation graphs and lowering supported subgraphs while preserving eager semantics.
- **Auditable Intermediate Progression**: Every compilation transformation is physically inspectable through concrete stage dumps in `demo_slides/` and `runtime_artifacts/`.

---

## Architecture Pipeline Deep-Dive

TritonFlow lowers high-level deep learning computations down to silicon microcode through 6 explicit, auditable stages:

```
  PyTorch Model (torch.compile)
               │
               ▼
   [Stage 0] PyTorch Source & FX Graph Partitioning
               │
               ▼
   [Stage 1] High-Level Hardware-Neutral Triton IR (TTIR)
               │
               ▼
   [Stage 2] Def-Use SSA & Memory Descriptor Recognition
             (Affine Stride Analysis, Gather/Scatter Bifurcation)
               │
               ▼
   [Stage 3] Declarative Schema-Driven Instruction Selection
             (Greedy Min-Cost & Equality Saturation E-Graphs)
               │
               ▼
   [Stage 4] Program Assembly & Cycle Cost Pricing
               │
               ▼
   [Stage 5] Target Execution & Differential Emulation
             (Bit-Accurate Python & Accelerated C++ Machine)
```

### Stage 0: PyTorch Model Definition & FX Graph Partitioning
PyTorch nn.Modules and functional tensor programs are ingested via PyTorch 2.0 Dynamo. TritonFlow intercepts the computation graph using `torch.fx`, tracing tensor shapes, intermediate buffers, and operator topologies into an explicit DAG. Operations eligible for acceleration (e.g. Matrix Multiplications, Pointwise Activations, Bias Epilogues) are grouped into lowerable subgraphs, while dynamic control flows are preserved.

### Stage 1: Hardware-Neutral Triton Intermediate Representation (TTIR)
The intercepted compute graph is lowered into canonical Triton-IR (TTIR). TTIR expresses tiled loop bounds, multi-dimensional pointer arithmetic, tile loads (`tt.load`), and tile dot-products (`tt.dot`). TritonFlow's custom MLIR lexer and parser construct a clean, typed AST with basic block terminators without relying on LLVM C++ runtime bindings.

### Stage 2: Def-Use Graph & Semantic Memory Descriptors
The compiler traverses SSA definition-use chains to analyze tensor memory access semantics. Using affine stride recognition, it bifurcates:
- **Contiguous & Strided 2D Tiles**: Recognized as affine tile descriptors with base, stride, shape, and element size for DMA block transfer acceleration.
- **Indirect Pointer Offsets**: Recognized as non-affine gather/scatter indexing operations.
- **Redundant Pointer Increments**: Coalesced and hoisted to eliminate pointer thrashing in nested loops.

### Stage 3: Declarative Schema-Driven Instruction Selection
Candidate instructions from the target ISA's YAML schema are evaluated against fail-closed constraint predicates. The selector verifies:
- Tile dimensionality constraints (e.g.,  \pmod{16} == 0, n \pmod{16} == 0$).
- Memory alignment and bank conflict conditions.
- Supported floating-point precisions (FP32, FP16, BF16, TF32).
- Cost modeling: Compares cycle latencies and resource usage across candidates (e.g., standard MMA vs Warpgroup WGMMA vs 2:4 Sparse TCU) and selects the mathematically optimal instruction.

### Stage 4: Assembly Emission & Microarchitectural Pricing
The compiler formats selected instructions into concrete machine assembly:
- **Preamble**: Configures base DMA registers, tile dimension strides, and scratchpad offsets.
- **Loop Body**: Emits pipelined tile loads and fused compute instructions (e.g., `TCU_WGMMA_SP32` or `OPU32`).
- **Epilogue**: Emits bias additions, pointwise activations (GELU, ReLU), and asynchronous stores.
- **Cost Engine**: Computes exact cycle times based on memory bus widths, bank conflicts, and compute pipeline latencies.

### Stage 5: Silicon Emulation & Verification
The generated machine assembly is executed on TritonFlow's cycle-accurate machine emulator:
- **Python Reference**: High-precision reference implementation for correctness assertions.
- **Accelerated C++ Machine Emulator (`_emu_cpp.so`)**: Bit-accurate hardware simulation with IEEE FP16/TF32 round-to-nearest-even conversion, bank conflict simulation, and tile cache simulation.
- **Output Validation**: Element-by-element numerical verification against eager PyTorch references with absolute difference bounds.

---

## Target Architectures & Benchmarks

Performance is modeled deterministically during instruction selection based on memory hierarchy, bus widths, and compute unit geometry.

For a representative `128×64×64` tiled matrix multiply:

| Architecture | Compute Unit | Memory Hierarchy | Sparsity | Model Cost (Cycles) | Speedup vs Baseline |
|---|---|---|:---:|:---:|:---:|
| **TRITONFLOW1** | MAC16 (16×16 Systolic) | Flat DMA1D / DMA2D | Dense | 25,405 | 1.00× (Baseline) |
| **TRITONFLOW2** | OPU32 (32×32 Outer Product) | 16-Bank Interleaved LDG | Dense | 20,108 | 1.26× |
| **VORTEX_RVGPU** | TCU_WGMMA (Warpgroup MMA) | Coalesced DXA Bulk Async | 2:4 Structured | 5,814 | **4.37×** |
| **EDGE_NPU** | INT_MAC16 / INT_MAC8 | Unified Global SRAM | Dense | N/A | Integer-Only Edge Target |

*The Vortex RISC-V GPU architecture achieves significant speedup primarily due to its 2:4 structured sparsity acceleration (19.2 cycles/tile) and wide coalesced bulk memory paths.*

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- C++17 compiler and CMake (for building the accelerated emulator)

### Installation

```bash
git clone https://github.com/Incharajayaram/segfault-tritonflow.git
cd segfault-tritonflow

# Install in editable mode
pip install -e .
```

### PyTorch Usage

```python
import torch
import tritonflow

# Define standard PyTorch model
class MatMulBiasReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 64))
        self.bias = torch.nn.Parameter(torch.randn(64))

    def forward(self, x):
        return torch.relu(x @ self.weight + self.bias)

model = MatMulBiasReLU()
compiled_model = torch.compile(model, backend="tritonflow")

x = torch.randn(8, 64)
output = compiled_model(x)
```

---

## Interactive Demos & Showcases

TritonFlow includes standalone, end-to-end demonstrations that showcase multi-ISA code generation, PyTorch `torch.compile` lowering, differential execution, and bit-accurate hardware emulation.

### 1. Master Compiler & PyTorch Showcase (`demo.py`)

The primary showcase demonstrates full stage-by-stage lowering across two deep learning workloads and an integrity audit:
- **Workload 1: Dense Tiled GEMM (128 x 128 x 64, tile 4 x 64 x 32$)**: Traces high-level matrix multiplication through TTIR, memory descriptors, instruction selection, assembly generation, and bit-accurate numerical verification.
- **Workload 2: Advanced Multi-Tile Neural Network (MTP / Fused MLP)**: Lowers an end-to-end multi-layer neural network (8 x 32 -> 8 x 64 -> 8 x 16) featuring Linear $->$ ReLU $->$ Linear with **100% execution on the custom ISA emulator (0 eager fallbacks)** and compares all 128 output numbers against eager PyTorch.
- **Workload 3: Fail-Closed Negative Control (Contract FR-025)**: Proves that non-affine pointer modulo operations are safely refused rather than silently generating invalid silicon code.
- **Runtime Artifacts**: Automatically generates 11 inspection files in `runtime_artifacts/`.

```bash
# Run automated non-interactive showcase (quick mode)
python3 demo.py --quick

# Step through interactively stage-by-stage
python3 demo.py

# Run specific workloads
python3 demo.py --workload gemm    # Dense Tiled GEMM
python3 demo.py --workload mlp     # Fused Multi-Tile Neural Network
python3 demo.py --workload audit   # Fail-closed refusal audit
```

### 2. Multi-ISA Diff & Selection Audit (`demo_diff.py`)

A standalone comparison showcase that does not require runtime JIT compilation, parsing canonical MLIR directly:
- **Stage 1**: TTIR AST and instruction structure.
- **Stage 2**: Def-Use Graph and Affine Memory Descriptors.
- **Stage 3**: Parallel 3-Way Lowering Mapping across TRITONFLOW1, TRITONFLOW2, and VORTEX_RVGPU.
- **Stage 4**: Target Accelerator Architecture Comparison Matrix (Hardware classes, compute units, memory pipelines, sparsity support, and cycle speedups).
- **Stage 5**: Compiler Instruction Selection Audit Trail (Displays exact mathematical constraint predicates evaluated for candidate instructions, modeled costs, and selection verdicts).

```bash
# Run full automated diff showcase
python3 demo_diff.py --auto

# Inspect specific canonical fixtures
python3 demo_diff.py --fixture t1_matmul --auto
python3 demo_diff.py --fixture t2_matmul_relu --auto
python3 demo_diff.py --fixture t0_vecadd --auto

# Step through interactively
python3 demo_diff.py
```

### 3. Open-Source Differential Test Suite (`tools/test_multi_isa_matrix.py`)

Curled directly from official Triton tutorials (`triton-lang/triton`) on GitHub (`01_vector_add.py`, `02_fused_softmax.py`, `03_matrix_multiplication.py`), this suite runs an 8-workload comparative matrix across all 3 target ISAs:
- Demonstrates how systolic, banked ASIC, and SIMT architectures perform on real-world Triton workloads.
- Displays detailed cycle costs and relative speedups.
- Demonstrates mathematical fail-closed refusal on kernels with unsupported reduction or pointer wrap semantics.

```bash
python3 tools/test_multi_isa_matrix.py
```

### 4. End-to-End PyTorch 2.0 Integration (`tools/demo_torch_compile.py`)

Demonstrates how TritonFlow acts as a first-class PyTorch compiler backend:
- Intercepts PyTorch subgraphs via Dynamo and FX graph lowering.
- Lowers operations directly to custom silicon ISA assembly.
- Dispatches execution to the compiled C++ machine emulator.
- Compares computed tensors against PyTorch eager execution.

```bash
PYTHONPATH=src python3 tools/demo_torch_compile.py
```

### 5. Live Physical GPU vs Silicon Emulator Parity (`tools/demo_live.py`)

Compiles Triton IR, executes the generated program on the bit-accurate machine emulator, runs the exact same kernel on a physical NVIDIA GPU (RTX 4060), and displays a numerical parity comparison:

```bash
# Vector addition (1D elementwise)
python3 tools/demo_live.py --tier t0_vecadd

# Matrix multiplication (2D tiled GEMM)
python3 tools/demo_live.py --tier t1_matmul

# Fail-closed refusal on unmodeled modulo wrap
python3 tools/demo_live.py --tier t3_modulo
```

---

## Runtime Artifacts Reference

When running `python3 demo.py`, concrete intermediate stage dumps are written to the `runtime_artifacts/` directory for inspection:

| Artifact File | Compiler Stage | Description |
|---|:---:|---|
| `gemm_stage0_source.py` | Stage 0 | PyTorch source definition of the tiled GEMM kernel (128 x 128 x 64). |
| `gemm_stage1_ttir.mlir` | Stage 1 | Complete Triton-IR MLIR module with tile loops and `tt.dot` operations. |
| `gemm_stage2_descriptors.txt` | Stage 2 | Extracted 2D affine memory descriptors (tile shapes, strides, offsets). |
| `gemm_stage3_selection.txt` | Stage 3 | Instruction selection decisions and cycle cost evaluations across ISAs. |
| `gemm_stage4_assembly.asm` | Stage 4 | Emitted target machine assembly instructions with preamble, loop, and epilogue. |
| `gemm_stage5_results.txt` | Stage 5 | Actual numerical output tensor computed on the custom silicon emulator. |
| `mlp_stage0_source.py` | Stage 0 | Multi-layer PyTorch Fused MLP model (8 x 32 -> 8 x 64 -> 8 x 16). |
| `mlp_stage1_fx_graph.txt` | Stage 1 | Captured PyTorch FX computation graph nodes and operand relationships. |
| `mlp_stage4_assembly.asm` | Stage 4 | Lowered silicon assembly for both dense linear layers and ReLU epilogue. |
| `mlp_stage5_results.txt` | Stage 5 | Comparison table of 128 computed numbers vs PyTorch eager reference. |
| `refusal_audit.txt` | Audit | Full mathematical log proving fail-closed refusal of non-affine pointer ops. |

To inspect any artifact:
```bash
cat runtime_artifacts/gemm_stage4_assembly.asm
cat runtime_artifacts/mlp_stage5_results.txt
cat runtime_artifacts/refusal_audit.txt
```

---

## Testing & Formal Verification

The TritonFlow verification suite enforces mathematical correctness, type safety, and contract compliance across the entire compiler stack:

```bash
# Run fast test suite (unit tests and core validation)
pytest tests/unit/ -q

# Run full pytest suite
pytest -q

# Run contract tests verifying architecture and functional requirements (FR-001 - FR-025)
pytest tests/contract/ -v

# Run differential emulator oracle (Python reference vs C++ emulator vs NumPy)
python3 verify/verify_differential_oracle.py

# Run formal AST mutation test harness
python3 tools/mutation_harness.py
```

- **Combinatorial Schema Matrix**: Evaluates every declared opcode across all 3 ISAs against an independent IEEE FP64 semantics oracle.
- **Differential Emulation**: Validates numerical equivalence between the Python emulator, C++ machine emulator, and NumPy reference outputs.
- **Fail-Closed Contracts (FR-001 - FR-025)**: Asserts that unrepresentable memory access patterns or unsupported reduction semantics refuse cleanly with typed errors rather than silently generating invalid code.
- **AST Mutation Score**: Synthesizes AST code mutants across instruction selection, cost calculations, and emulator routines to mathematically prove test sensitivity against compiler bugs.

---

## Project Structure

```
.
├── bench/                # Benchmark suites & open-source Triton tutorial kernels
│   └── open_source/      # Curled real-world Triton workloads from triton-lang/triton
├── fixtures/             # Canonical TTIR kernel fixtures and launch environments
├── runtime_artifacts/    # Concrete stage-by-stage compiler dumps generated during demo execution
├── src/
│   └── tritonflow/
│       ├── canon/        # IR canonicalization passes
│       ├── emit/         # Target assembly and binary formatting
│       ├── emu/          # Python & C++ bit-accurate machine emulators
│       ├── extract/      # Ahead-of-time (AOT) and dynamic PyTorch/Triton ingestion
│       ├── idioms/       # Hardware idiom detection
│       ├── isa/          # Declarative YAML schemas, cost models, and selection engines
│       ├── pipeline.py   # Unified 5-stage compilation driver
│       ├── recognize/    # Affine memory descriptors and gather/scatter recognition
│       ├── torch_backend/# torch.compile Dynamo and FX graph partitioning
│       └── ttir/         # TTIR MLIR lexer, parser, and def-use SSA builder
├── tests/                # Unit, contract, property, and mutation verification suites
├── third_party/          # Vendored Vortex RISC-V hardware configs
└── tools/                # Multi-ISA matrix, diff showcases, live parity, and AOT runners
```

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).
