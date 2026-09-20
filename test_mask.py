from tritonflow.extract.dynamic_extract import extract_elementwise
ex = extract_elementwise("add", 100)
print(ex.ttir)
