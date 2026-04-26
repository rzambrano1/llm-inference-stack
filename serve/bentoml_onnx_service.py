# Run: $bentoml serve serve.bentoml_onnx_service:GPT2OnnxService --port 3000
# Access in browser at: http://localhost:3000             <--- main UI (Swagger-like interface)
#                       http://localhost:3000/docs.json   <---  OpenAPI spec

import bentoml
from pydantic import BaseModel

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

import time
import psutil
import os

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class GPT2OnnxService:

    def __init__(self):
        # runs once at startup — equivalent to FastAPI lifespan
        self.model = ORTModelForCausalLM.from_pretrained("model_onnx")
        self.tokenizer = AutoTokenizer.from_pretrained("model_onnx")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @bentoml.api
    def generate(self, request: GenerateRequest) -> dict:

        # System snapshot BEFORE. To get RAM before inference (MB)... could be overkill
        # Documenting in case I need this ina larger project
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)

        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]

        # ---> Inference starts here
        e2e_start = time.perf_counter()

        output = self.model.generate(
            **inputs, 
            max_length=request.max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            )

        e2e_latency = time.perf_counter() - e2e_start

        # System snapshot AFTER
        ram_after = process.memory_info().rss / (1024 * 1024)
        cpu_after = process.cpu_percent(interval=None)  # % of one core

        output_tokens = output.shape[1] - input_tokens

        # Derived metrics
        tokens_per_second = round(output_tokens / e2e_latency, 2)
        tpot_ms = round((e2e_latency / output_tokens) * 1000, 2)  # average time per each output token - ms per token

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return {
            "text": decoded,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "latency_sec": round(e2e_latency, 4),
            "tpot_msec": tpot_ms,

            "tokens_per_second": tokens_per_second,

            "cpu_percent": round(cpu_after, 1),
            "ram_used_mb": round(ram_after - ram_before, 2),
            "peak_ram_mb": round(ram_after, 2),
        }
