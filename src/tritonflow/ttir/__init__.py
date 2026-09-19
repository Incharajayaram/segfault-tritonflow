"""ttir: the front end.

`lexer` and `parser` are the syntax layer (text-level, `RawModule`); `ssa`, `to_ir`
and `graph` are the semantic layer (`Module`). `RawModule` is the seam between
them.

No module in here may import Triton or torch (enforced by
`tests/unit/test_no_triton_runtime.py`).
"""
