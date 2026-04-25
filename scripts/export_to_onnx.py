# This stack works on Python 3.11 + transformers 4.51.3. This stack is compatible with ONNX,
# at the time of implementation the lastest version of transformers is not compatible with ONNX.

from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer
from pathlib import Path

def main():

    model_id = "gpt2"  # or the local folder path where the model is saved
    output_dir = Path("model_onnx")

    print(f'Exporting {model_id} to ONNX...')

    output_dir.mkdir(exist_ok=True)

    # Export to ONNX
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="text-generation-with-past",     # Double check this parameter other option "causal-lm", "text-generation"
        do_validation=True,         # Checks if ONNX output matches PyTorch output
        no_post_process=False,      # Default setting
        device="cpu"
    )

    # Saving tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    print(f"Successfully exported {model_id} to {output_dir}")

if __name__ == "__main__":
    main()