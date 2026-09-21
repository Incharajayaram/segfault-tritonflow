"""Path 1 of the extraction pipeline: dynamic TTIR from real Triton compilation.

Importing this package does *not* need Triton. Every entry point degrades to a
typed refusal (:class:`~tritonflow.extract.dynamic_extract.ExtractionUnavailable`)
carrying the reason, which is what lets the seam run with Triton absent and say
so rather than pretend it extracted something.
"""

from __future__ import annotations

from .aot import AotResult, StageOutcome, compile_aot
from .dynamic_extract import (
    DEFAULT_TILE,
    ELEMENTWISE_OPS,
    SUPPORTED_OPS,
    Capability,
    CompileRecord,
    CompileSpy,
    Extracted,
    ExtractionError,
    ExtractionUnavailable,
    capability,
    compile_triton_kernel,
    extract_elementwise,
    extract_for_op,
    extract_linear,
    extract_matmul,
    is_supported,
    is_triton_available,
    record_compilations,
    require_triton,
)
from .inductor_bridge import (
    INDUCTOR_OPS,
    CaptureResult,
    InductorUnavailable,
    capture_inductor_ttir,
    extract_via_inductor,
    is_inductor_op,
    records_to_extracted,
)

__all__ = [
    "DEFAULT_TILE",
    "ELEMENTWISE_OPS",
    "INDUCTOR_OPS",
    "SUPPORTED_OPS",
    "AotResult",
    "CaptureResult",
    "Capability",
    "CompileRecord",
    "CompileSpy",
    "Extracted",
    "ExtractionError",
    "ExtractionUnavailable",
    "InductorUnavailable",
    "StageOutcome",
    "capability",
    "capture_inductor_ttir",
    "compile_aot",
    "compile_triton_kernel",
    "extract_elementwise",
    "extract_for_op",
    "extract_linear",
    "extract_matmul",
    "extract_via_inductor",
    "is_inductor_op",
    "is_supported",
    "is_triton_available",
    "record_compilations",
    "records_to_extracted",
    "require_triton",
]
