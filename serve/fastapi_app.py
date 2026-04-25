# serve/fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time

app = FastAPI()

# The ONNX [or other model] model should be loaded at startup — not inside the endpoint
model = ORTModelForCausalLM.from_pretrained("model_onnx")
tokenizer = AutoTokenizer.from_pretrained("model_onnx")
tokenizer.pad_token = tokenizer.eos_token

class GenerateRequest(BaseModel):  # use Pydantic model, not query params
    prompt: str
    max_new_tokens: int = 100

@app.post("/generate")
def generate(request: GenerateRequest):
    start = time.perf_counter()

    inputs = tokenizer(request.prompt, return_tensors="pt")
    input_tokens = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs, 
        # max_new_tokens=request.max_new_tokens
        max_length=request.max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        )

    latency = time.perf_counter() - start
    output_tokens = output.shape[1] - input_tokens

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "text": decoded,
        "latency_s": round(latency, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(output_tokens / latency, 2),
    }

@app.get("/health")
def health():
    return {"status": "ok"}