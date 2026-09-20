import sys; sys.path.insert(0, "src")
from tritonflow.emu.exec import _sizes
print("_sizes(None):", _sizes(None))  # Should be empty tuple
