import sys
sys.path.insert(0, "src")
from tritonflow.extract.dynamic_extract import extract_elementwise
from tritonflow.ttir.to_ir import parse_module

ex = extract_elementwise("add", 100)
parsed = parse_module(ex.ttir).module
for block in parsed.body.blocks:
    for op in block.operations:
        if op.name == "tt.func":
            for b in op.regions[0].blocks:
                for o in b.operations:
                    if o.name == "tt.get_program_id":
                        print("get_program_id op properties:", vars(o))
