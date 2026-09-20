import sys
import re

with open("src/tritonflow/emu/exec.py", "r") as f:
    content = f.read()

# Fix 1: _declared_shape
old_shape = """def _declared_shape(instr: Instr) -> tuple[int, ...]:
    for operand in instr.operands.values():
        if isinstance(operand, MemRef):
            return _sizes(operand.access_key)
    return ()"""

new_shape = """def _declared_shape(instr: Instr) -> tuple[int, ...]:
    for operand in instr.operands.values():
        if isinstance(operand, MemRef) and operand.access_key is not None:
            sizes = _sizes(operand.access_key)
            if sizes:
                return sizes
    if instr.constrained_on:
        import re
        match = re.search(r"sizes=\\[([0-9, ]+)\\]", instr.constrained_on)
        if match:
            return tuple(int(s) for s in match.group(1).split(",") if s.strip())
    return ()"""

content = content.replace(old_shape, new_shape)

# Fix 2: _apply_memory
old_mem = """    if not instr.defs:
        raise UnsupportedInstruction(f"{instr.name} load defines no value to bind")
    loaded = state.gather(state.resolve(src), _align_mask(mask, state.resolve(src)))
    state.bind(instr.defs[0], loaded.reshape(_declared_shape(instr) or loaded.shape))"""

new_mem = """    if not instr.defs:
        raise UnsupportedInstruction(f"{instr.name} load defines no value to bind")

    tile_shape = _declared_shape(instr)
    
    # If access_key is None, resolve(src) only returned [base]. Generate the full contiguous range.
    if getattr(src, "access_key", None) is None and src.base in state.storages:
        storage = state.storages[src.base]
        # Fused-K case needs the full buffer extent, nominal case needs tile_shape
        count = int(np.prod(storage.shape)) if storage.shape else int(np.prod(tile_shape)) if tile_shape else 1
        indices = np.arange(storage.base, storage.base + count, dtype=np.int64)
    else:
        indices = state.resolve(src)

    loaded = state.gather(indices, _align_mask(mask, indices))

    expected = int(np.prod(tile_shape)) if tile_shape else 0
    if tile_shape and loaded.size != expected:
        # Fused-K: loaded full buffer extent. Reshape to actual storage shape.
        storage = state.storages.get(src.base) if isinstance(src, MemRef) else None
        actual_shape = storage.shape if storage is not None else (loaded.size,)
        state.bind(instr.defs[0], loaded.reshape(actual_shape))
    else:
        state.bind(instr.defs[0], loaded.reshape(tile_shape or loaded.shape))"""

content = content.replace(old_mem, new_mem)

with open("src/tritonflow/emu/exec.py", "w") as f:
    f.write(content)
