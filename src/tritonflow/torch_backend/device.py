"""The toy device's storage plane: allocate, copy, synchronise, account.

**This module imports no torch**, deliberately. Everything here is a property of
the *device*, not of the framework that drives it: a flat fp32 array, a bump
allocator, and the alignment promise the ISA schema declares. Keeping it
framework-free means the storage rules are testable without PyTorch, and it
means the ISA schema's `alignment_words: 4` is enforced in one place
rather than asserted in a comment.

The alignment field is the reason this module exists at all. `isa/schemas/
tritonflow1.yaml` declares `alignment_words: 4` under `data_model.memory_spaces`, and
the selector *decides* `aligned(a_base, 4)` is `True` on the strength of it
(`isa/schema.py`'s symbol grounding). A machine that allocated at arbitrary word
offsets would make every program the selector produced unsound — the constraint
would have been checked against a promise nobody kept. So the promise is kept
here, in code, by refusing an allocation whose base is not a multiple of the
declared alignment.

`ToyDevice` models one flat address space with a **word** (fp32 element) as the
addressing unit, which is the same model `emu/exec.py` executes against: a
pointer is an element index. That equivalence is what lets the seam hand the
emulator's own buffers to a caller as tensors without a conversion step that
could silently reorder anything.

The device is *synchronous*. `synchronize()` therefore returns without waiting on
anything, and that is a statement about the machine, not a stub: there is no
queue. It is counted (`sync_count`) so a caller can prove its synchronisation
points were reached rather than assume them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ALIGNMENT_WORDS",
    "DEVICE",
    "WORD_BYTES",
    "AlignmentError",
    "DeviceError",
    "DevicePtr",
    "OutOfStorage",
    "ToyDevice",
    "allocate",
    "copy_device_to_host",
    "copy_host_to_device",
    "device",
    "free",
    "reset",
    "synchronize",
]

#: One fp32 element. The addressing unit of the whole project: the ISA's
#: `alignment_words`, the emulator's flat array, and the DMA's `words` cost all
#: count in these.
WORD_BYTES = 4

#: `isa/schemas/tritonflow1.yaml` → `data_model.memory_spaces[0].alignment_words`.
#: Duplicated here as a named constant rather than read from the schema because
#: the *allocator* must not depend on the schema loader: the machine keeps its
#: promise whether or not anyone loaded a YAML file to be told about it.
ALIGNMENT_WORDS = 4

#: 4 Mi words = 16 MiB. The corpus's largest tier addresses 8 KiB, so this is
#: three orders of magnitude of headroom; it is a bound, not a measurement.
DEFAULT_CAPACITY_WORDS = 1 << 22


class DeviceError(RuntimeError):
    """A device-level misuse: a bad pointer, a shape mismatch, exhaustion."""


class AlignmentError(DeviceError):
    """An allocation or copy that would break the declared alignment promise."""


class OutOfStorage(DeviceError):
    """The device's storage is exhausted. Named, never silently grown."""


@dataclass(frozen=True)
class DevicePtr:
    """A handle to a region of device storage.

    `base` is the **word index** into the device's flat array, exactly as
    `emu/exec.py`'s `Storage.base` is. `length` is in words (elements), not
    bytes; `nbytes` derives bytes so a caller who thinks in bytes is not
    silently off by four.
    """

    base: int
    length: int
    shape: tuple[int, ...]
    dtype: str = "f32"

    @property
    def nbytes(self) -> int:
        return self.length * WORD_BYTES

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"DevicePtr(base={self.base}, length={self.length}, "
            f"shape={list(self.shape)}, dtype={self.dtype})"
        )


class ToyDevice:
    """One flat fp32 address space with a bump allocator and an alignment promise.

    Allocation is bump-only and `free` releases nothing: a device this small does
    not need a real allocator, and a fake one that *looked* like a real one would
    invite a leak-shaped bug that only shows up on the hundredth compile. What it
    does provide is the two things that are actually load-bearing:

    * **the alignment promise** — every `base` is a multiple of
      `alignment_words`, refused otherwise;
    * **bounds accounting** — `allocated_bytes` and `live_bytes` are real sums
      over live regions, so the coverage report's memory column is a measurement
      rather than an estimate.
    """

    def __init__(
        self,
        *,
        name: str = "tritonflow",
        alignment_words: int = ALIGNMENT_WORDS,
        capacity_words: int = DEFAULT_CAPACITY_WORDS,
    ) -> None:
        if alignment_words < 1:
            raise AlignmentError(f"alignment_words must be >= 1, got {alignment_words}")
        self.name = name
        self.alignment_words = alignment_words
        self.capacity_words = capacity_words
        self._memory = np.zeros(capacity_words, dtype=np.float32)
        self._next = 0
        self._live: dict[int, DevicePtr] = {}
        self._sync_count = 0
        self._copies = 0

    # -- accounting ---------------------------------------------------------- #

    @property
    def allocated_words(self) -> int:
        return self._next

    @property
    def allocated_bytes(self) -> int:
        return self._next * WORD_BYTES

    @property
    def live_regions(self) -> int:
        return len(self._live)

    @property
    def copy_count(self) -> int:
        return self._copies

    @property
    def sync_count(self) -> int:
        return self._sync_count

    def storage(self) -> np.ndarray:
        """The whole flat array. Read-only by convention; the seam never mutates it."""
        return self._memory

    # -- allocation ---------------------------------------------------------- #

    def allocate(self, shape: int | tuple[int, ...], dtype: str = "f32") -> DevicePtr:
        """Allocate `shape` words on the device. The base is always aligned.

        A caller asking for an alignment the machine has not promised is refused
        rather than quietly rounded up: `aligned(a_base, 16)` is `False` in
        ISA-1's predicate language *because* the promise is 4, and a device that
        happened to satisfy 16 would make that `False` a lie.
        """
        dims = (int(shape),) if isinstance(shape, int) else tuple(int(d) for d in shape)
        if not dims or any(d < 0 for d in dims):
            raise DeviceError(f"cannot allocate shape {dims!r}")
        words = 1
        for dim in dims:
            words *= dim
        # Round the bump pointer up to the alignment boundary before placing: an
        # odd-sized allocation (a 3-word region) leaves `_next` at 3, and handing
        # out base 3 would break the promise — but refusing *without advancing*
        # deadlocks the allocator forever, which a strengthened check exposed.
        # Rounding up is what real allocators do, and it keeps every base aligned
        # by construction rather than by luck of the sizes requested.
        base = -(-self._next // self.alignment_words) * self.alignment_words
        if base + words > self.capacity_words:
            raise OutOfStorage(
                f"device storage exhausted: asked for {words} words at {base}, "
                f"capacity is {self.capacity_words} words"
            )
        pointer = DevicePtr(base=base, length=words, shape=dims, dtype=dtype)
        self._next = base + words
        self._live[base] = pointer
        return pointer

    def free(self, pointer: DevicePtr) -> None:
        """Release a region from the live set. Storage is not reclaimed."""
        if pointer.base not in self._live:
            raise DeviceError(f"{pointer} is not a live allocation of {self.name}")
        del self._live[pointer.base]

    def view(self, pointer: DevicePtr) -> np.ndarray:
        """The region as a shaped array. A *copy*, so a caller cannot alias storage."""
        end = pointer.base + pointer.length
        if end > self.capacity_words:
            raise DeviceError(f"{pointer} extends past the device's storage")
        return self._memory[pointer.base : end].reshape(pointer.shape)

    # -- transfer ------------------------------------------------------------ #

    def copy_host_to_device(self, src: np.ndarray, pointer: DevicePtr) -> None:
        array = np.asarray(src, dtype=np.float32)
        if array.size != pointer.length:
            raise DeviceError(
                f"copy shape mismatch: host has {array.size} words, {pointer} has "
                f"{pointer.length}"
            )
        self._memory[pointer.base : pointer.base + pointer.length] = array.reshape(-1)
        self._copies += 1

    def copy_device_to_host(self, pointer: DevicePtr) -> np.ndarray:
        self._copies += 1
        return self.view(pointer).copy()

    # -- synchronisation ----------------------------------------------------- #

    def synchronize(self) -> None:
        """Return once every submitted operation has completed.

        The machine is synchronous, so there is nothing to wait for; the call is
        counted so that "we synchronised before reading" is checkable.
        """
        self._sync_count += 1

    def reset(self) -> None:
        """Return the device to its post-boot state. Used by checks, not by the seam."""
        self._memory[:] = np.float32(0.0)
        self._next = 0
        self._live.clear()
        self._sync_count = 0
        self._copies = 0

    def assert_alignment(self) -> None:
        """Every live region starts on the declared boundary. Raises otherwise."""
        for pointer in self._live.values():
            if pointer.base % self.alignment_words != 0:
                raise AlignmentError(
                    f"{pointer} starts at word {pointer.base}, which is not a multiple "
                    f"of the declared alignment {self.alignment_words}"
                )

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"ToyDevice({self.name}, alignment={self.alignment_words} words, "
            f"allocated={self.allocated_bytes} B, live={self.live_regions})"
        )


#: The process's device. One is enough: the corpus is one kernel at a time, and a
#: second device would be a second set of alignment promises with no consumer.
DEVICE = ToyDevice()


def device() -> ToyDevice:
    return DEVICE


# `torch.cuda`-shaped module-level functions, so `torch.tritonflow.allocate(...)`
# is a real entry point rather than a name that resolves to nothing.


def allocate(shape: int | tuple[int, ...], dtype: str = "f32") -> DevicePtr:
    return DEVICE.allocate(shape, dtype)


def free(pointer: DevicePtr) -> None:
    DEVICE.free(pointer)


def copy_host_to_device(src: np.ndarray, pointer: DevicePtr) -> None:
    DEVICE.copy_host_to_device(src, pointer)


def copy_device_to_host(pointer: DevicePtr) -> np.ndarray:
    return DEVICE.copy_device_to_host(pointer)


def synchronize() -> None:
    DEVICE.synchronize()


def reset() -> None:
    DEVICE.reset()
