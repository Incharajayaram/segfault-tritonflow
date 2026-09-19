"""emu: NumPy execution of the emitted stream, with a derived tolerance policy.

No NumPy dtype magic is hidden here: the precision policy is explicit and
recorded per dtype.
"""

try:
    from tritonflow.emu import _emu_cpp  # noqa: F401
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from .dxa import DxaDescriptor, DxaEmulator, DxaPerformanceCounters
from .hardware import BankConflictModel, BankConflictUnit, CoalescingUnit, HardwarePerformanceStats
from .tcu import TcuEmulator, TcuPerformanceCounters

__all__ = [
    "HAS_CPP",
    "BankConflictModel",
    "BankConflictUnit",
    "CoalescingUnit",
    "DxaDescriptor",
    "DxaEmulator",
    "DxaPerformanceCounters",
    "HardwarePerformanceStats",
    "TcuEmulator",
    "TcuPerformanceCounters",
]
