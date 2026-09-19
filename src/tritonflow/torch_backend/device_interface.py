"""The `DeviceInterface` seat: what makes `tritonflow` a device PyTorch can name.

PyTorch's device machinery is reached through two Python-only doors
(`torch/_dynamo/device_interface.py`), and this module walks through the one that
leads somewhere:

    register_interface_for_device("tritonflow", TritonFlowInterface)

`register_interface_for_device` is a plain dictionary write — it takes a *string*
and does no device-string parsing — so after this call
`get_interface_for_device("tritonflow")` returns this class, and every part of
PyTorch that asks a device what it can do (`is_available`, `device_count`, device
properties, dtype support) gets an answer about *this* machine. This is the
registration that matters for the deliverable, and it needs no C++.

**The other door is closed in C++, and opening the obvious way makes things
worse.** `torch.device("tritonflow")`, `torch.tritonflow.<anything>` and
`tensor.to("tritonflow")` all require `torch.utils.rename_privateuse1_backend`, which
re-registers the *PrivateUse1* dispatcher key. Measured on this checkout
(torch 2.12.1+cpu):

    >>> torch.utils.rename_privateuse1_backend("tritonflow")
    >>> torch.device("tritonflow")            # works
    >>> torch.tensor([1.0]).to("tritonflow")
    RuntimeError: PyTorch is not linked with support for tritonflow devices

and — the finding that decided this design — the rename also makes
`torch.accelerator` treat `tritonflow` as *the* accelerator, which breaks Dynamo for
**every** graph, toy device or not:

    torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: PyTorch is not
    linked with support for tritonflow devices
      at torch/_dynamo/variables/streams.py:273 in SymbolicStreamState.__init__
        stream = torch.accelerator.current_stream()

So the rename would cost the entire backend to buy a device string that still
cannot allocate a tensor. It is therefore not performed, `RENAME_BLOCKED` records
it, and `UNAVAILABLE` lists each C++-gated capability with the error it produces.
`torch.utils.cpp_extension` could open all of them — that is exactly what
`torch_openreg`, the project's reference pattern, does — but that would make the
artifact depend on a C++ toolchain to be *installed*, and the project's whole
framing is that the reachable surface (Triton's IR, PyTorch's backend hook) is
Python. The data plane stays out of scope, by measurement rather than by choice.

**The slot inventory is asserted, not assumed.** This module keeps a test that
fails when PyTorch adds a slot upstream without an explicit
decision here, because a slot that silently falls back to a base-class default is
a slot whose behaviour nobody chose. `SLOT_INVENTORY` is measured from the
installed torch (see `measure_slot_inventory`) and `UNIMPLEMENTED` lists every
slot this interface refuses **with the reason it refuses**, which is what the
limitations document quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device

from . import device as device_module

__all__ = [
    "DATA_PLANE_GAP",
    "DEVICE_NAME",
    "SLOT_INVENTORY",
    "UNIMPLEMENTED",
    "NotSupportedError",
    "ToyDeviceProperties",
    "TritonFlowInterface",
    "device_module_object",
    "install",
    "measure_slot_inventory",
]

#: The device type string, in one place. `torch.device(DEVICE_NAME)` is what a
#: caller writes; nothing hard-codes the literal twice.
DEVICE_NAME = "tritonflow"

#: Why the data plane is not implemented, quoted into the limitations document.
DATA_PLANE_GAP = (
    "`tensor.to('tritonflow')` is not available: allocating a torch.Tensor on a "
    "PrivateUse1 device requires an allocator and dispatch keys registered from a "
    "compiled C++ extension (the reference is torch_openreg, which ships one). "
    "Measured here: RuntimeError('PyTorch is not linked with support for tritonflow "
    "devices'). The seam therefore runs programs supplied by Dynamo and returns "
    "tensors, and never accepts a tensor whose storage lives on the toy device."
)

#: Why the PrivateUse1 rename is not performed, measured rather than assumed.
RENAME_BLOCKED = (
    "`rename_privateuse1_backend('tritonflow')` is deliberately not called. It would "
    "buy `torch.device('tritonflow')` (and still not `tensor.to('tritonflow')`, which needs "
    "a C++ allocator), but it makes torch.accelerator treat tritonflow as the "
    "accelerator, so torch.accelerator.current_stream() raises "
    "'PyTorch is not linked with support for tritonflow devices' inside "
    "torch/_dynamo/variables/streams.py and EVERY graph fails to trace. "
    "Measured on torch 2.12.1+cpu."
)

#: Capabilities gated behind C++, with the error each produces when asked for.
#: A reader of the limitations document gets the exact failure, not a category.
UNAVAILABLE: dict[str, str] = {
    "tensor.to('tritonflow')": (
        "RuntimeError: PyTorch is not linked with support for tritonflow devices"
    ),
    "torch.device('tritonflow')": (
        "RuntimeError: Expected one of cpu, cuda, ... device type at start of device "
        "string: tritonflow"
    ),
    "torch.tritonflow.<module>": (
        "RuntimeError: Expected one of cpu, cuda, ... device type at start of device "
        "string: tritonflow"
    ),
    "rename_privateuse1_backend": RENAME_BLOCKED,
}


class NotSupportedError(NotImplementedError):
    """A device slot this machine deliberately does not implement.

    A subclass of `NotImplementedError` so a caller that checks the standard
    exception still works, and a distinct type so a check can assert *which*
    slots refuse rather than merely that something raised.
    """

    def __init__(self, slot: str, reason: str) -> None:
        self.slot = slot
        self.reason = reason
        super().__init__(f"{DEVICE_NAME}.{slot} is not implemented: {reason}")


@dataclass(frozen=True)
class ToyDeviceProperties:
    """What `get_device_properties` returns.

    Field names follow the attributes PyTorch code reads off a device properties
    object (`type`, `index`, `name`, `major`, `minor`, `total_memory`,
    `multi_processor_count`), because a caller that duck-types this against
    `torch.cuda.get_device_properties` should not have to special-case us.
    """

    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int
    type: str = DEVICE_NAME
    index: int = 0

    @property
    def uuid(self) -> str:
        return f"{DEVICE_NAME}-0"

    @property
    def L2_cache_size(self) -> int:
        # The toy machine has no cache hierarchy; the ISA-1 model declares a
        # flat global space and nothing else. 0 states that, rather than inventing
        # a plausible number a scheduler might then trust.
        return 0

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.name} ({DEVICE_NAME}:{self.index})"


#: Slots that refuse, and why. Every entry is quoted by the limitations document;
#: none is a silent stub, and none pretends to work.
UNIMPLEMENTED: dict[str, str] = {
    "current_stream": (
        "the toy machine is synchronous: there is no queue, so there is no current "
        "stream to return"
    ),
    "set_stream": "no device queue exists to assign work to",
    "stream": "no device stream type exists; the machine's only ordering is program order",
    "get_raw_stream": "there is no raw stream handle; the machine exposes no stream object",
    "Event": "no asynchronous work exists for an event to record or wait on",
    "Stream": "no asynchronous work exists to schedule on a stream",
}


def measure_slot_inventory() -> tuple[str, ...]:
    """The device slots the *installed* torch declares, sorted.

    Measured rather than written down, for the same reason the fixtures are
    frozen dumps: a list typed by hand records what the author believed, and this
    list is used to notice that PyTorch changed.
    """
    return tuple(
        sorted(
            name
            for name in vars(DeviceInterface)
            if not name.startswith("__") and not name.startswith("_")
        )
    )


#: The inventory at the time this interface was written. Asserted against the
#: installed torch by the contract check, so an upstream addition is a decision
#: to make rather than a base-class default to inherit.
SLOT_INVENTORY: tuple[str, ...] = (
    "Event",
    "Stream",
    "Worker",
    "current_device",
    "current_stream",
    "device",
    "device_count",
    "exchange_device",
    "get_compute_capability",
    "get_device_properties",
    "get_raw_stream",
    "is_available",
    "is_bf16_supported",
    "is_dtype_supported",
    "is_triton_capable",
    "maybe_exchange_device",
    "memory_allocated",
    "raise_if_triton_unavailable",
    "set_device",
    "set_stream",
    "stream",
    "synchronize",
)


class TritonFlowInterface(DeviceInterface):
    """The toy device as PyTorch sees it.

    `device_count` is 1 and `current_device` is a real piece of state that
    `set_device` moves and validates: a device count of one with no state would
    make `set_device(0)` unobservable, and the contract asks for a round trip.
    """

    _current_device = 0

    # -- identity ------------------------------------------------------------ #

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return TritonFlowInterface._current_device

    @staticmethod
    def set_device(device: int | torch.device | str) -> None:
        index = _index_of(device)
        if index != 0:
            raise ValueError(
                f"{DEVICE_NAME} has exactly one device (index 0); refusing to select {index}"
            )
        TritonFlowInterface._current_device = index

    @staticmethod
    def get_device_properties(device: int | torch.device | str | None = None) -> Any:
        index = 0 if device is None else _index_of(device)
        if index != 0:
            raise ValueError(f"{DEVICE_NAME} has no device {index}")
        return ToyDeviceProperties(
            name="tritonflow emulator (CPU numpy backend)",
            major=1,
            minor=0,
            total_memory=device_module.DEVICE.capacity_words * device_module.WORD_BYTES,
            multi_processor_count=1,
        )

    # -- device contexts ----------------------------------------------------- #

    class device:
        """`with <interface>.device(0):` — sets the current device for the block.

        A nested class, not a method, because that is the shape torch's own
        `DeviceInterface` declares the slot in — and `SLOT_INVENTORY` names it, so
        the check that compares our slots against torch's measured ones would
        otherwise be comparing a class against a function. Torch's base raises;
        implementing it is cheap, and the alternative is a caller wrapping every
        call in try/except for no reason.
        """

        def __init__(self, device: Any) -> None:
            self.device = device
            self.previous = TritonFlowInterface._current_device

        def __enter__(self) -> TritonFlowInterface.device:
            TritonFlowInterface.set_device(self.device)
            return self

        def __exit__(self, *exc: object) -> bool:
            TritonFlowInterface._current_device = self.previous
            return False

    # -- capabilities -------------------------------------------------------- #

    @staticmethod
    def is_bf16_supported(including_emulation: bool = True) -> bool:
        # ISA-1's data model declares one dtype (`f32`) and the emulator's storage
        # is fp32. bf16 is not supported, emulated or otherwise.
        return False

    @staticmethod
    def is_dtype_supported(dtype: torch.dtype, device: Any = None) -> bool:
        return dtype == torch.float32

    @staticmethod
    def is_triton_capable(device: Any = None) -> bool:
        # The toy ISA is the *target* of a Triton lowering; it does not run Triton.
        return False

    @staticmethod
    def raise_if_triton_unavailable(*args: Any, **kwargs: Any) -> None:
        raise NotSupportedError(
            "raise_if_triton_unavailable",
            "the toy ISA is a Triton lowering target, not a Triton backend",
        )

    @staticmethod
    def get_compute_capability(device: Any = None) -> int:
        raise NotSupportedError(
            "get_compute_capability",
            "a compute capability is an NVIDIA-specific property; the toy ISA "
            "describes its abilities in its schema, not in a version number",
        )

    # -- execution ----------------------------------------------------------- #

    @staticmethod
    def synchronize(device: Any = None) -> None:
        device_module.synchronize()

    @staticmethod
    def memory_allocated(device: Any = None) -> int:
        return device_module.DEVICE.allocated_bytes

    @staticmethod
    def exchange_device(device: Any) -> Any:
        return _exchange(device)

    @staticmethod
    def maybe_exchange_device(device: Any) -> Any:
        return _exchange(device)

    # -- refusals ------------------------------------------------------------ #
    # Each raises NotSupportedError carrying the reason recorded in UNIMPLEMENTED;
    # nothing here returns a plausible-looking default.

    @staticmethod
    def current_stream(device: Any = None) -> Any:
        raise NotSupportedError("current_stream", UNIMPLEMENTED["current_stream"])

    @staticmethod
    def set_stream(stream: Any) -> None:
        raise NotSupportedError("set_stream", UNIMPLEMENTED["set_stream"])

    @staticmethod
    def stream(stream: Any) -> Any:
        raise NotSupportedError("stream", UNIMPLEMENTED["stream"])

    @staticmethod
    def get_raw_stream(device: Any = None) -> Any:
        raise NotSupportedError("get_raw_stream", UNIMPLEMENTED["get_raw_stream"])

    class Stream:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotSupportedError("Stream", UNIMPLEMENTED["Stream"])

    class Event:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotSupportedError("Event", UNIMPLEMENTED["Event"])

    class Worker(DeviceInterface.Worker):
        """The worker view of the device, delegating to the interface.

        PyTorch's `Worker` is the subset a worker process may touch; all three
        slots have real answers here, so none of them is a refusal.
        """

        @staticmethod
        def current_device() -> int:
            return TritonFlowInterface.current_device()

        @staticmethod
        def get_device_properties(device: Any = None) -> Any:
            return TritonFlowInterface.get_device_properties(device)

        @staticmethod
        def set_device(device: Any) -> None:
            TritonFlowInterface.set_device(device)


def _index_of(device: Any) -> int:
    """The integer index of a device-like value, refusing anything not ours."""
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        if device.type != DEVICE_NAME:
            raise ValueError(f"{device} is not a {DEVICE_NAME} device")
        return device.index if device.index is not None else 0
    if isinstance(device, str):
        parsed = torch.device(device)
        return _index_of(parsed)
    raise ValueError(f"cannot read a device index out of {device!r}")


class _Exchange:
    """`with torch.tritonflow.device(0):` — the device-context protocol."""

    def __init__(self, device: Any) -> None:
        self.target = device
        self.previous = device_module.DEVICE
        self.previous_index = TritonFlowInterface._current_device

    def __enter__(self) -> int:
        TritonFlowInterface.set_device(self.target)
        return TritonFlowInterface._current_device

    def __exit__(self, *exc: object) -> bool:
        TritonFlowInterface._current_device = self.previous_index
        del self.previous
        return False


def _exchange(device: Any) -> _Exchange:
    return _Exchange(device)


def device_module_object() -> ModuleType:
    """The module `torch.tritonflow` resolves to.

    Registered under the device name so `torch.tritonflow.allocate(...)` works the way
    `torch.cuda.allocate` would, which is the honest entry point for storage on a
    device whose tensors cannot be allocated through `torch.empty`.
    """
    module = ModuleType(f"torch.{DEVICE_NAME}")
    module.__doc__ = (
        "The toy device's storage plane: allocate, copy, synchronize. "
        "See tritonflow.torch_backend.device for the machine this drives."
    )
    for name in (
        "ALIGNMENT_WORDS",
        "DEVICE",
        "WORD_BYTES",
        "allocate",
        "copy_device_to_host",
        "copy_host_to_device",
        "device",
        "free",
        "reset",
        "synchronize",
    ):
        setattr(module, name, getattr(device_module, name))
    module.is_available = TritonFlowInterface.is_available
    module.device_count = TritonFlowInterface.device_count
    module.current_device = TritonFlowInterface.current_device
    module.set_device = TritonFlowInterface.set_device
    module.get_device_properties = TritonFlowInterface.get_device_properties
    module._is_compiled = False
    return module


_installed = False


def install() -> dict[str, Any]:
    """Make `tritonflow` a device PyTorch can name. Idempotent, and C++-free.

    One dictionary write: `register_interface_for_device(DEVICE_NAME, cls)`.
    Nothing here reaches into PrivateUse1, so installing the seam cannot break
    anything else in the process — which is the property that makes it safe to do
    at import time, and the reason the rename path is refused (see
    `RENAME_BLOCKED`).
    """
    global _installed
    if _installed:
        return {"interface_registered": True, "renamed": False, "cxx_required": list(UNAVAILABLE)}
    register_interface_for_device(DEVICE_NAME, TritonFlowInterface)
    _installed = True
    return {"interface_registered": True, "renamed": False, "cxx_required": list(UNAVAILABLE)}
