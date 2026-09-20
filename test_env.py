import sys; sys.path.insert(0, "src")
import numpy as np
import torch
from tritonflow.torch_backend import compiler as seam
from tritonflow.extract import dynamic_extract as de

extracted = de.extract_matmul(128, 128, 64)
prep = seam.prepare(extracted.name, extracted.ttir, extracted.env, provenance="dynamic")
print("env:", extracted.env)
print("extents:", prep.extents)
print("grid:", prep.grid)
