import sys
sys.path.insert(0, "src")
import torch
from tests.integration.test_dynamic_extraction import run_backend

def mock_plan_graph(graph, example_inputs):
    print("GRAPH NODES:")
    for node in graph.graph.nodes:
        print(f"  op={node.op} target={node.target} target_name={getattr(node.target, '__name__', 'N/A')}")
    import tritonflow.torch_backend.compiler as c
    return orig_plan(graph, example_inputs)

import tritonflow.torch_backend.compiler as c
orig_plan = c.plan_graph
c.plan_graph = mock_plan_graph

model = torch.nn.Sequential(
    torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
).eval()
x = torch.randn(8, 64)
try:
    call, result = run_backend(lambda t: model(t), x)
except Exception as e:
    pass
