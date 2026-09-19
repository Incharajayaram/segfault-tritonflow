import argparse
import sys
from pathlib import Path

from tritonflow.pipeline import compile_fixture


def main():
    parser = argparse.ArgumentParser(prog="tritonflow")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--out", default=None, help="Output directory for extracted fixtures")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--check-gate", action="store_true")

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("fixture", help="Path to .ttir fixture")
    compile_parser.add_argument("--schema", default="vortex_rvgpu", help="ISA schema name or path")
    compile_parser.add_argument("--out", default=None, help="Output program path")

    args = parser.parse_args()

    if args.command == "extract":
        if args.check_gate:
            from tritonflow.extract.dynamic_extract import capability
            cap = capability(refresh=True)
            if not cap.available:
                print(f"GPU-free extraction unavailable: {cap.reason}", file=sys.stderr)
                sys.exit(1)
            print(f"GPU-free extraction OK: {cap.reason}")
            return
        if not args.out:
            print("tritonflow extract: error: --out is required when --check-gate is not specified", file=sys.stderr)
            sys.exit(2)
        from tritonflow.harness.extract_fixtures import extract
        extract(args.out, args.force)
    elif args.command == "compile":
        schema = Path(args.schema).stem if args.schema else "vortex_rvgpu"
        result = compile_fixture(args.fixture, isa_name=schema)
        if not result.fully_lowered:
            print(f"Compilation refused {len(result.unsupported)} operation(s):", file=sys.stderr)
            for reason in result.unsupported:
                print(f"  {reason}", file=sys.stderr)
            sys.exit(1)
        if args.out:
            from tritonflow.emit.disasm import serialize
            Path(args.out).write_text(serialize(result.program), encoding="utf-8")
            print(f"Compiled to {args.out} (cost={result.total_cost})")
        else:
            print(f"Compiled {args.fixture} (cost={result.total_cost}, instrs={result.emitted_instructions})")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
