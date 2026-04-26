## NOTE: Same to export_to_onnx.py with minor mods
## ------------------------------------------------

# This stack works on Python 3.11 + transformers 4.51.3. This stack is compatible with ONNX,
# at the time of implementation the lastest version of transformers is not compatible with ONNX.

# from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path

def main():

    model_id = "gpt2"  # or the local folder path where the model is saved
    output_dir = Path("model_repository/gpt2_onnx/1")

    print(f'Exporting {model_id} to ONNX...')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    model = ORTModelForCausalLM.from_pretrained(
        "gpt2",
        export=True,
        use_cache=False,
    )

    model.save_pretrained(output_dir)
    
    # Saving tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir.parent)

    print(f"Successfully exported {model_id} to {output_dir}")

if __name__ == "__main__":
    main()