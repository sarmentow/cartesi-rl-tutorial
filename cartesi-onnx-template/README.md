# ONNX DApp Template

This is a template for ONNX Cartesi DApps in python using the onnxruntime package. It uses python3 to execute the backend application.
The application entrypoint is the `dapp.py` file.

## How to use it
1. Replace the simple_nn.onnx file by the model you've trained
2. Change the MODEL_INPUT_SHAPE variable, and the dtype to conform to your input shape specification
3. Send your inputs as base64 encoded strings as such:
```python
import numpy as np
import base64

np.random.seed(42)
# Create a (1, 10) random float32 numpy array
arr = np.random.rand(1, 10).astype(np.float32)

# Serialize to bytes and encode to a base64 string
arr_bytes = arr.tobytes()
encoded_str = base64.b64encode(arr_bytes).decode('utf-8')

# Output the encoded string
print(encoded_str)
```
