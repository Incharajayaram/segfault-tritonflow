import sys; sys.path.insert(0, "src")
import torch
from tritonflow.torch_backend import compiler as seam
from tritonflow.extract import dynamic_extract as de

extracted = de.extract_matmul(8, 32, 64)
print("padded:", extracted.padded, "problem:", extracted.problem)
print("grid:", seam.prepare(extracted.name, extracted.ttir, extracted.env, provenance="dynamic").grid)
print("extents:", seam.prepare(extracted.name, extracted.ttir, extracted.env, provenance="dynamic").extents)
print("=== TTIR ===")
print(extracted.ttir)
