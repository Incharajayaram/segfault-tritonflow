# TritonFlow: Schema-Driven Multi-ISA AI Accelerator Compiler

TritonFlow is a research compiler that enables rapid retargeting of AI accelerators via declarative ISA schemas. It transforms PyTorch/Triton code into hardware-specific binaries without rewriting compiler backends—simply provide a new YAML ISA description.

## Key Features

- **Schema-Driven Design**: ISAs are defined in YAML files specifying instructions, constraints, and costs.
- **Fail-Closed Guarantee**: The compiler silently refuses to generate code it cannot verify, eliminating silent miscompilations.
- **Multi-ISA Code Generation**: The same Triton intermediate representation (TTIR) lowers to diverse architectures (e.g., systolic arrays, banked memory ASICs, RISC-V SIMT) by swapping the ISA schema.
- **PyTorch Integration**: Registers as a `torch.compile` backend, requiring no user code changes.
- **Honest Fallback**: When a kernel cannot be lowered, execution falls back to eager PyTorch with a documented reason.
- **Auditability**: Every selection decision is traceable to exact predicate evaluations and cost models.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.9+ (with CUDA support for GPU demos)
- Triton (for dynamic extraction)
- Rich (for terminal demos)

Install dependencies:
```bash
pip install torch triton rich
```

### Running the Demo

From the project root:
```bash
# Interactive menu
PYTHONPATH=src python3 tritonflow_demo.py

# Run all demos sequentially (ideal for recording)
PYTHONPATH=src python3 tritonflow_demo.py --auto

# Jump to a specific demo (1-6)
PYTHONPATH=src python3 tritonflow_demo.py --demo 3
```

See `tritonflow_demo.py` for detailed usage.

## Project Structure

```
src/
├── tritonflow/
│   ├── extract/          # Dynamic TTIR extraction from PyTorch/Triton
│   ├── recognize/        # ISA-aware pattern matching
│   ├── isa/              # Schema loading and instruction definitions
│   ├── emit/             # Assembly and code generation
│   ├── emu/              # Emulator for validation
│   ├── torch_backend/    # torch.compile integration
│   └── ...               # Other components (idioms, lower, etc.)
tests/                    # Unit, contract, and integration tests
fixtures/                 # Reference TTIR kernels and launch environments
```

## Test Status

As of the latest run, all tests pass:
- **175 passed**, 1 skipped (C++ backend not installed), 3 subtests passed
- Test suite includes contract tests, unit tests, and integration tests covering extraction, lowering, selection, emulation, and torch.compile integration.

See `ARCHITECTURE_AND_STATUS.md` for detailed benchmark numbers and test breakdowns.

## Architecture Overview

TritonFlow operates in stages:
1. **Extract**: Obtain TTIR from PyTorch (via attached, dynamic, flaggems, or recorded sources).
2. **Parse**: Convert TTIR text to an MLIR-like IR.
3. **Recognize**: Bind operations to ISA-specific patterns using rewrite rules.
4. **Select**: Choose the cheapest admissible instruction per operation using fail-closed predicate evaluation.
5. **Emit**: Assemble a binary program (or emulator-ready representation) for the target ISA.
6. **Validate**: Run in emulator or on hardware to verify correctness.

The core innovation is that stages 2–5 are driven entirely by the ISA schema—adding a new hardware target requires only a new YAML file.

## Extending to New Hardware

To add support for a new accelerator:
1. Create a YAML file in `src/tritonflow/isa/schemas/` (or any path) describing:
   - Instruction set (name, kind, rule, cost, constraints)
   - Data model (memory spaces, alignment, banking)
   - Allocator promises (alignment guarantees)
2. Reference the schema by name (or path) in the compiler or demo.
3. No code changes are required to the pipeline.

Example schema snippets are available in `src/tritonflow/isa/schemas/` for TRITONFLOW1, TRITONFLOW2, and VORTEX_RVGPU.

## Design Philosophy

TritonFlow adheres to three hard-won lessons from compiler engineering:
1. **Decidable Predicates**: Constraints are limited to integer comparisons and `aligned`/`in_bounds` checks—no arbitrary code execution—ensuring the solver terminates and decisions are auditable.
2. **Fail-Closed by Construction**: Every term resolves to a known value or `unknown`; arithmetic/propagation of `unknown` makes inadmissibility explicit, preventing speculative choices.
3. **Symbolic Descriptors**: Launch environment variables (e.g., strides, offsets) ground symbolic addresses; the same TTIR may yield different code based on runtime parameters, preserving correctness.

These principles ensure that TritonFlow never guesses, never silently sacrifices correctness, and always explains *why* a kernel is unsupported.

## License

This project is research code. See individual files for licensing information.

## Acknowledgments

Developed for the IICT CompilerTech Hackathon 2026. Contributions from the open-source Triton, PyTorch, and MLIR communities.