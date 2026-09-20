# TritonFlow: Schema-Driven Multi-ISA AI Accelerator Compiler

TritonFlow is a declarative, schema-driven compiler backend that enables rapid retargeting of emerging AI accelerators directly from PyTorch and Triton IR (TTIR). Instead of rewriting thousands of lines of C++ LLVM backend passes for every new tape-out or chip architecture, TritonFlow separates **compiler mechanics** from **hardware specifications**: simply provide a declarative YAML schema defining instruction semantics, constraints, and cost models.

---

## Key Highlights

- **Declarative ISA Schemas**: Target architectures are expressed entirely in YAML descriptions specifying instruction opcodes, memory bank properties, constraint predicates, and cost functions.
- **Fail-Closed Verification**: The compiler mathematically refuses to lower code it cannot prove admissible, eliminating silent numerical corruption and falling back cleanly to eager PyTorch execution.
- **Multi-ISA Retargetability**: Lowers the exact same high-level Triton IR into drastically different chip designs:
  - **TRITONFLOW1**: 16×16 Flat Systolic Array (DMA1D / DMA2D).
  - **TRITONFLOW2**: 32×32 Outer Product Unit (OPU) with 16-bank conflict arbitration.
  - **VORTEX_RVGPU**: Open-source RISC-V SIMT GPGPU with 2:4 structured sparsity and coalesced bulk async transfers (grounded in official Vortex hardware RTL configs).
- **Native PyTorch Integration**: Registers seamlessly as a `torch.compile(backend="tritonflow")` backend, intercepting computation graphs and lowering supported subgraphs while preserving eager semantics.
- **Auditable Intermediate Progression**: Every compilation transformation is physically inspectable through concrete stage dumps in `demo_slides/`.

---

## Architecture Pipeline

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

1. **Extraction & Ingestion**: Intercepts `torch.compile` or accepts pre-compiled `.ttir` fixtures, parsing them into an SSA module with explicit def-use chains.
2. **Semantic Descriptor Recognition**: Recovers multidimensional tensor memory access patterns, separating contiguous 2D strided tiles from indexed gather/scatter loads, and eliding redundant pointer increments.
3. **Instruction Selection**: Evaluates candidate instructions against declarative YAML constraint predicates (alignment, bounds, data types). Supports both sub-millisecond **Greedy Min-Cost Selection** and formal **E-Graph Equality Saturation** rewrites.
4. **Assembly & Cost Modeling**: Emits structured preamble, loop-body, and epilogue instructions, computing deterministic, shape-aware cycle counts based on memory transfer shapes and compute pipelines.
5. **Emulation & Verification**: Validates execution against bit-accurate Python references and a compiled C++ machine emulator (`_emu_cpp.so`) featuring round-half-to-even IEEE float16 conversion.

---

## Target Architectures & Benchmarks

Performance is modeled deterministically during instruction selection based on memory hierarchy, bus widths, and compute unit geometry.

For a representative `128×64×64` tiled matrix multiply:

| Architecture | Compute Unit | Memory Hierarchy | Sparsity | Model Cost (Cycles) | Speedup vs Baseline |
|---|---|---|:---:|:---:|:---:|
| **TRITONFLOW1** | MAC16 (16×16 Systolic) | Flat DMA1D / DMA2D | Dense | 25,405 | 1.00× (Baseline) |
| **TRITONFLOW2** | OPU32 (32×32 Outer Product) | 16-Bank Interleaved LDG | Dense | 20,108 | 1.26× |
| **VORTEX_RVGPU** | TCU_WGMMA (Warpgroup MMA) | Coalesced DXA Bulk Async | 2:4 Structured | 5,814 | **4.37×** |

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

### Running Interactive Demos

```bash
# Run interactive multi-ISA demo
python3 tritonflow_demo.py

# Run non-interactive automated showcase
python3 tritonflow_demo.py --auto

# View intermediate compiler stages
cat demo_slides/STAGE_0_pytorch_source.py
cat demo_slides/STAGE_1_fx_graph.txt
cat demo_slides/STAGE_2_ttir_raw.mlir
cat demo_slides/STAGE_3_memory_descriptors.txt
cat demo_slides/STAGE_4_instruction_selection.txt
cat demo_slides/STAGE_5_assembled_programs.txt
```

---

## Testing & Verification

The compiler verification suite enforces mathematical correctness across the entire stack:

```bash
# Run test suite
pytest -q
```

- **Combinatorial Schema Matrix**: Evaluates every declared opcode across all 3 ISAs against an independent IEEE FP64 semantics oracle.
- **Differential Emulation**: Validates numerical equivalence between the Python emulator, C++ machine emulator, and NumPy reference outputs.
- **Fail-Closed Contracts**: Asserts that unrepresentable memory access patterns or unsupported reduction semantics refuse cleanly with typed errors rather than silently generating invalid code.
- **Mutation Testing**: Synthesizes AST code mutants across instruction selection, cost calculations, and emulator routines to mathematically prove test sensitivity against compiler bugs.

---

## Project Structure

```
.
├── demo_slides/          # Stage-by-stage compiler IR dumps (PyTorch -> TTIR -> Descriptors -> ASM)
├── fixtures/             # Canonical TTIR kernel fixtures and launch environments
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
└── tools/                # Benchmark suites, AOT runners, and selector comparison harnesses
```

---

## License

This project is open-source research code developed for the IICT CompilerTech Hackathon 2026.
