# Using ONNX Runtime's built-in quantizer instead of bitsandbytes because the
# latter is designed for CUDA GPU inference.

from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import shutil

def main():

    print(" Quantizing model with ONNX runtime...")

    input_model = Path("model_onnx/model.onnx")
    output_model = Path("model_onnx_quantized/model.onnx")
    output_model.parent.mkdir(exist_ok=True)

    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8   # INT8 is the sweet spot for CPU
    )

    # Creating a copy of the tokenizer files to the quantized folder
    for f in Path("model_onnx").glob("*.json"):
        shutil.copy(f, "model_onnx_quantized")

    print("Quantization done!")

if __name__ == "__main__":
    main()