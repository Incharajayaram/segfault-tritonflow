"""Canonicalisation that makes **no reduction claim**.

v1 of the methodology justified this pass with a premise its own dump disproves:
that `ttir` carries "redundant `tt.splat`/`tt.broadcast` chains" to collapse.
Measured on the frozen corpus, Tier 1 has 13 `tt.splat`, 6 `tt.broadcast` and
**0** sign-extension or truncate operations — the splat/broadcast sequence *is*
the required broadcast expansion of `rm[:,None]*sam + rk[None,:]*sak`, and
TensorLift's A1/A2 passes (which fold sign-extension) have no referent here.
There is nothing to reduce, so this module reduces nothing and says so.

What it does instead is the part that is real: **normalisation**. Attribute
dictionaries are rebuilt in sorted key order. Attribute *order* is not semantics
— `{noinline = false, sym_name = "k"}` and its reverse are the same operation —
but it is visible in any textual form and it varies with how the printer
happened to emit the dict, so two runs over the same module could differ in
their canonical text for no reason. Sorting it here means the determinism suite
compares something that cannot drift.

Two things are deliberately **not** done, and both for the same reason — they
would change what a reader or a downstream stage sees:

* **no value renaming.** SSA names are what `loc` names and every report
  reference; renaming `%acc_25#2` to `%v7` would make the emitted program
  unreconcilable with the Triton dump.
* **no operation removal, no re-association, no folding.** Not implemented, and
  :func:`assert_no_reduction_claim` fails if anyone adds one.

One thing *is* normalised that looks like content, and the case for it is
narrow: the **`loc` file**. A `loc` binding is `file:line:col`, and the file is
the Python source the kernel was compiled from — provenance, not IR. The frozen
corpus already records a decision about this: the harness rewrites every source
path to the token `LOCFILE`, because a corpus fixture cannot bake in one
machine's checkout path. `canonical_form` had not been told, so it rendered the
raw path, and `canonical_form(live) != canonical_form(frozen)` for every tier —
not from any nondeterminism (it is stable across `PYTHONHASHSEED`s and across
processes; measured) but because the two captures named the same file
differently. That made the digest useless for the one thing the frozen-fixture
workflow needs it for: deciding whether a regeneration changed the IR or merely
reproduced it from a different path. The line and column are *kept* — those move
when the kernel changes, which is exactly what the digest is for.

**Idempotence is the whole guarantee**: `canonicalize(canonicalize(m)) ==
canonicalize(m)`, asserted by tests, and `canonical_form(m)` is unchanged by
canonicalisation (that equality *is* the no-reduction guard).
"""

from __future__ import annotations

from dataclasses import replace

from ..ttir.ssa import Block, Module, Operation, Region


class CanonError(ValueError):
    """A canonicalisation that broke one of its own two promises.

    Raised by :func:`assert_no_reduction_claim` — either the pass stopped being
    idempotent, or it started changing what the module means.
    """


def canonicalize(module: Module) -> Module:
    """Rebuild `module` with attribute dictionaries in sorted key order.

    Total (every module maps to a module), pure (the input is not mutated — the
    `Operation`/`Region` tree is frozen and is rebuilt rather than edited), and
    idempotent.
    """
    return Module(
        body=_region(module.body),
        loc_table=dict(module.loc_table),
        source_path=module.source_path,
        triton_version=module.triton_version,
    )


#: The token the harness substitutes for a source path, and the one
#: `canonical_form` renders. See the module docstring for why.
LOC_FILE = "LOCFILE"


def canonical_form(module: Module) -> str:
    """A stable textual form of the whole module, for determinism tests.

    Deterministic by construction: no `set` is iterated, no `id()` is printed,
    no dict order is relied on (attribute keys are sorted here too), and the
    result depends only on values that survive a round trip through the parser.

    The `loc` file is rendered as :data:`LOC_FILE` rather than as the path it was
    read from, so two captures of the same IR compare equal. Position is kept.
    """
    lines: list[str] = []
    _render_region(module.body, lines, depth=0)
    for name in sorted(module.loc_table):
        loc = module.loc_table[name]
        lines.append(f"#loc {name} = {LOC_FILE}:{loc.line}:{loc.col}")
    return "\n".join(lines) + "\n"


def assert_no_reduction_claim(before: Module, after: Module) -> None:
    """Fail if canonicalisation changed the module's meaning or its size.

    Both checks are cheap and both are tripwires rather than proofs: with the
    implementation above they pass by construction. That is the point — the guard
    exists so that the *next* change, the one that adds a "harmless" folding
    pass to "improve" the numbers, fails here instead of quietly inflating a
    coverage percentage in `report/coverage.py`.
    """
    before_ops = _count_ops(before)
    after_ops = _count_ops(after)
    if before_ops != after_ops:
        raise CanonError(
            f"canonicalisation changed the operation count ({before_ops} -> {after_ops}); "
            "this pass is not allowed to remove or add operations — the corpus has no "
            "redundancy to remove"
        )
    if canonical_form(before) != canonical_form(after):
        raise CanonError(
            "canonicalisation changed the module's canonical form; normalisation must "
            "change representation only (attribute order), never content"
        )
    if canonical_form(canonicalize(after)) != canonical_form(after):
        raise CanonError("canonicalize is not idempotent; the second pass changed the module")


# --------------------------------------------------------------------------- #
# Rebuilding, parent links included
# --------------------------------------------------------------------------- #


def _region(region: Region) -> Region:
    """A region with rebuilt blocks. The caller sets `parent` on the new owner."""
    return Region(blocks=tuple(_block(block) for block in region.blocks))


def _block(block: Block) -> Block:
    operations = tuple(_operation(op) for op in block.operations)
    index = -1
    if block.terminator is not None:
        for position, op in enumerate(block.operations):
            if op is block.terminator:
                index = position
                break
    terminator = operations[index] if 0 <= index < len(operations) else None
    return Block(args=block.args, operations=operations, terminator=terminator)


def _operation(op: Operation) -> Operation:
    regions = tuple(_region(region) for region in op.regions)
    rebuilt = replace(
        op,
        attributes={name: op.attributes[name] for name in sorted(op.attributes)},
        regions=regions,
    )
    for region in rebuilt.regions:
        # The one sanctioned escape hatch for a back-reference (ssa.py explains
        # why): a region knows its owning operation, and that link is a cycle, so
        # it is filled in after construction rather than passed through it.
        object.__setattr__(region, "parent", rebuilt)
    return rebuilt


def _count_ops(module: Module) -> int:
    total = 0
    for block in module.body.blocks:
        total += _count_block(block)
    return total


def _count_block(block: Block) -> int:
    total = 0
    for op in block.operations:
        total += 1
        for region in op.regions:
            for nested in region.blocks:
                total += _count_block(nested)
    return total


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_region(region: Region, lines: list[str], *, depth: int) -> None:
    for block in region.blocks:
        for arg in block.args:
            lines.append(f"{'  ' * depth}arg {arg.name} : {arg.type.raw}")
        for op in block.operations:
            lines.append(_render_op(op, depth=depth))
            for nested in op.regions:
                _render_region(nested, lines, depth=depth + 1)


def _render_op(op: Operation, *, depth: int) -> str:
    indent = "  " * depth
    results = ", ".join(value.name for value in op.results)
    operands = ", ".join(value.name for value in op.operands)
    attrs = " ".join(f"{name}={op.attributes[name].value}" for name in sorted(op.attributes))
    loc = f" loc({LOC_FILE})" if op.loc is not None else ""
    head = f"{results} = " if results else ""
    return f"{indent}{head}{op.name}({operands}) [{attrs}]{loc} @{op.line}:{op.col}"


__all__ = [
    "LOC_FILE",
    "CanonError",
    "assert_no_reduction_claim",
    "canonical_form",
    "canonicalize",
]
