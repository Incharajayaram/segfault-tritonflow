"""User-friendly diagnostic reporting engine.

Replaces opaque Python exception stack traces with structured, actionable compiler diagnostics
featuring source locations, visual caret pointers, contextual error explanations, and hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "Diagnostic",
    "DiagnosticSeverity",
    "diagnose_descriptor_failure",
    "diagnose_refusal",
    "diagnose_unsupported_op",
    "format_diagnostic",
]


try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility
    class StrEnum(str, Enum):
        pass


class DiagnosticSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


@dataclass(frozen=True)
class Diagnostic:
    """A single structured compiler diagnostic."""

    severity: DiagnosticSeverity
    message: str
    op_name: str | None = None
    loc: str | None = None
    line: int | None = None
    col: int | None = None
    source_line: str | None = None
    hint: str | None = None

    def render(self, use_color: bool = True) -> str:
        """Render diagnostic message with visual formatting and caret pointer."""
        c_red = "\033[1;31m" if use_color else ""
        c_yellow = "\033[1;33m" if use_color else ""
        c_blue = "\033[1;34m" if use_color else ""
        c_bold = "\033[1m" if use_color else ""
        c_reset = "\033[0m" if use_color else ""

        sev_color = c_red if self.severity == DiagnosticSeverity.ERROR else (
            c_yellow if self.severity == DiagnosticSeverity.WARNING else c_blue
        )

        header = f"{sev_color}{self.severity.value.upper()}{c_reset}: {c_bold}{self.message}{c_reset}"
        if self.loc or self.line:
            loc_parts = []
            if self.loc:
                loc_parts.append(f"loc({self.loc})")
            if self.line:
                loc_parts.append(f"line {self.line}" + (f":{self.col}" if self.col else ""))
            header += f" [{', '.join(loc_parts)}]"

        lines = [header]

        # Source context snippet with pointer
        if self.source_line:
            lines.append(f"    {self.source_line}")
            if self.col and self.col > 0:
                indent = " " * (4 + self.col - 1)
                lines.append(f"{c_red}{indent}^{c_reset}")

        # Actionable suggestion / hint
        if self.hint:
            lines.append(f"  {c_blue}hint{c_reset}: {self.hint}")

        return "\n".join(lines)


def format_diagnostic(
    message: str,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    op_name: str | None = None,
    loc: str | None = None,
    line: int | None = None,
    col: int | None = None,
    source_line: str | None = None,
    hint: str | None = None,
    use_color: bool = True,
) -> str:
    """Convenience helper to format a diagnostic into a displayable string."""
    d = Diagnostic(
        severity=severity,
        message=message,
        op_name=op_name,
        loc=loc,
        line=line,
        col=col,
        source_line=source_line,
        hint=hint,
    )
    return d.render(use_color=use_color)


def diagnose_unsupported_op(
    op_name: str,
    loc: str | None = None,
    line: int | None = None,
    reason: str | None = None,
    target_isa: str = "tritonflow1",
) -> Diagnostic:
    """Build an actionable diagnostic for an unsupported operation."""
    msg = f"Cannot lower operation '{op_name}' to target ISA '{target_isa}'"
    hint = None

    if op_name in ("arith.remsi", "arith.remui"):
        hint = "Non-affine modulo address calculation is intentionally unmodeled (FR-025). Kernel will be delegated to PyTorch eager fallback."
    elif op_name == "scf.for":
        hint = "Ensure loop iteration variables are bound in iter_args with explicit SSA def-use tracking."
    elif "dot" in op_name:
        hint = f"Target ISA '{target_isa}' requires square matrix tile sizes matching hardware MAC engine (e.g. 16x16)."
    else:
        hint = f"Check if a custom lowering rule or instruction can be added to '{target_isa}.yaml'."

    if reason:
        msg += f": {reason}"

    return Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        message=msg,
        op_name=op_name,
        loc=loc,
        line=line,
        hint=hint,
    )


def diagnose_descriptor_failure(
    base: str,
    strides: Any,
    offsets: Any,
    loc: str | None = None,
) -> Diagnostic:
    """Build an actionable diagnostic when access descriptor cannot resolve affine strides."""
    return Diagnostic(
        severity=DiagnosticSeverity.ERROR,
        message=f"Memory access on base '{base}' does not resolve to an affine layout",
        loc=loc,
        hint=(
            f"Observed strides={strides}, offsets={offsets}. "
            "DMA engines require constant or symbol-affine strides. "
            "Indirect or pointer-chasing patterns trigger eager fallback."
        ),
    )


def diagnose_refusal(program: Any, kernel_name: str = "kernel") -> list[Diagnostic]:
    """Scan an assembled Program for UnsupportedMarker markers and generate diagnostics."""
    diagnostics = []
    if hasattr(program, "markers"):
        for m in program.markers():
            src = getattr(m, "source", None)
            op_name = getattr(src, "op_name", "unknown") if src else "unknown"
            loc = getattr(src, "loc_name", None) if src else None
            line = getattr(src, "line", None) if src else None
            reason = getattr(m, "reason", "unsupported operation")

            diagnostics.append(
                diagnose_unsupported_op(
                    op_name=op_name,
                    loc=loc,
                    line=line,
                    reason=reason,
                    target_isa=getattr(program, "isa_name", "tritonflow1"),
                )
            )
    return diagnostics
