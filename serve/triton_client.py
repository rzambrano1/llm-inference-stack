# Requires running triton via Docker and installing tritonclient [this script].
# Docker instructions:
#
# Linux / WSL2
# ------------
#
# docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#   -v $(pwd)/model_repository:/models \
#   nvcr.io/nvidia/tritonserver:24.10-py3 \
#   tritonserver --model-repository=/models
#
# Windows PowerShell
# ------------------
#
# docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 `
#   -v {rest_of_absolute_path}/model_repository:/models `
#   nvcr.io/nvidia/tritonserver:24.10-py3 `
#   tritonserver --model-repository=/models
#
# --------------------------------------------------------------------------
# Explanation of the Docker command
# --------------------------------------------------------------------------
# port 8000 = HTTP, port 8001 = gRPC, port 8002 = performance_stats/metrics
# The volume mounting command [-v ${PWD}/model_repository:/models]
# connects the local folder [local path section: ${PWD}/model_repository
# where config.pbtxt and ./1/model.onnx live] to the `:/models` fake path
# inside the container. the result of this setup is that Triton "thinks"
# the models are stored in /models, but it's actually reading them directly 
# from the Windows/Linux folder passed in the local path section above.
#
# The image section `nvcr.io/nvidia/tritonserver:24.10-py3` is the official 
# blueprint for the server.
#
# Finally, the container command `tritonserver --model-repository=/models`
# tells the tritonserver application to "look" for the model folders in the
# /models folder in the container (which, remember, is reading the models 
# from the local path).
# --------------------------------------------------------------------------
#
# Client installing instructions:
# ===============================
#
# Run: $pip install tritonclient[http]
# --------------------------------------------------------------------------
# NOTE:
# In triton one gets real TTFT (time to first token) because generation loop
# runs token by token. FastAPI and BentoML generate all tokens before return
# thus TTFT == latency.
#
# In triton Parameters like temperature, top_k, top_p, and repetition_penalty
# seem to require custom implementation :(
# ---------------------------------------------------------------------------

import tritonclient.http as httpclient
from transformers import AutoTokenizer

import numpy as np
import time
import psutil
import os
from pathlib import Path

# Setup
# -----

# Hardwired library
TRITON_URL = "localhost:8000"
MODEL_NAME = "gpt2_onnx"
MODEL_ONNX_PATH = Path("model_onnx/model.onnx")

tokenizer = AutoTokenizer.from_pretrained("model_onnx")
tokenizer.pad_token = tokenizer.eos_token

client = httpclient.InferenceServerClient(url=TRITON_URL)

model_file_size_mb = round(MODEL_ONNX_PATH.stat().st_size / (1024 * 1024), 2)

def triton_forward_pass(input_ids_np, attention_mask_np):
    """Single forward pass through Triton — returns logits."""

    triton_input_ids = httpclient.InferInput(
        "input_ids", input_ids_np.shape, "INT64"
    )

    triton_input_ids.set_data_from_numpy(input_ids_np)

    triton_attn = httpclient.InferInput(
        "attention_mask", attention_mask_np.shape, "INT64"
    )

    triton_attn.set_data_from_numpy(attention_mask_np)

    seq_len = input_ids_np.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    triton_pos_ids = httpclient.InferInput("position_ids", position_ids.shape, "INT64")
    triton_pos_ids.set_data_from_numpy(position_ids)

    triton_output = httpclient.InferRequestedOutput("logits")

    response = client.infer(
        MODEL_NAME,
        inputs=[triton_input_ids, triton_attn, triton_pos_ids],
        outputs=[triton_output]
    )
    return response.as_numpy("logits")  # shape: [1, seq_len, vocab_size]

def generate(prompt: str, max_new_tokens: int = 100):
    """
    Autoregressive generation loop on the client side.
    Each iteration calls Triton for one forward pass,
    picks the next token greedily, appends it, repeats.
    """

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    process.cpu_percent(interval=None)  

    # Tokenize
    enc = tokenizer(prompt, return_tensors="np") # Numpy not PyTorch [pt stands for pytorch]
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    input_tokens = input_ids.shape[1]

    generated_tokens = []
    first_token_time = None

    e2e_start = time.perf_counter()

    for _ in range(max_new_tokens):
        logits = triton_forward_pass(input_ids, attention_mask)

        # greeddy strategy next token, will get stuck repeating
        # next_token_id = int(np.argmax(logits[0, -1, :])) 

        # Temperature sampling — much less repetitive than greedy argmax
        probs = logits[0, -1, :].astype(np.float64)
        probs = probs - np.max(probs)               # numerical stability
        probs = np.exp(probs / 0.9)                 # temperature=0.9
        probs = probs / probs.sum()                 # normalize to probabilities
        next_token_id = int(np.random.choice(len(probs), p=probs))
        generated_tokens.append(next_token_id)

        # Capture TTFT on first token
        if first_token_time is None:
            first_token_time = time.perf_counter()

        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        # Append token and extend attention mask
        input_ids = np.concatenate(
            [input_ids, np.array([[next_token_id]], dtype=np.int64)], axis=1
        )

        attention_mask = np.concatenate(
            [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
        )

    e2e_latency = time.perf_counter() - e2e_start
    ttft = first_token_time - e2e_start if first_token_time else e2e_latency # Remember difference in formulas and approx. estimation

    ram_after = process.memory_info().rss / (1024 * 1024)
    cpu_after = process.cpu_percent(interval=None)

    output_tokens = len(generated_tokens)
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + decoded

    return {
        "text": full_text,

        # Token counts
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,

        # Latency
        "latency_sec": round(e2e_latency, 4),
        "ttft_s": round(ttft, 4),              # Real TTFT (not an approximation) if conditional above is met
        "tpot_msec": round((e2e_latency / output_tokens) * 1000, 2),

        # Throughput
        "tokens_per_second": round(output_tokens / e2e_latency, 2),

        # Efficiency
        "cpu_percent": round(cpu_after, 1),
        "ram_used_mb": round(ram_after - ram_before, 2),
        "peak_ram_mb": round(ram_after, 2),
        "model_file_size_mb": model_file_size_mb,
    }

if __name__ == "__main__":
    prompt = "Once upon a time there was a dog that was a tenth dan in Taekwondo"

    print(f"Prompt: {prompt}\n")
    print("Running inference via Triton...\n")

    result = generate(prompt, max_new_tokens=100)

    print(f"Output:\n  {result['text']}\n")
    print("Metrics:")
    for k, v in result.items():
        if k != "text":
            print(f"  {k}: {v}")