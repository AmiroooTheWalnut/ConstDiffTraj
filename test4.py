import onnx

# Load the ONNX model
model = onnx.load("modelV3.onnx")

# Print output names
print("ONNX Model Output Names:")
for output in model.graph.output:
    print(output.name)