"""
Benchmarks FastAPI, BentoML, and Triton servers against the same prompts.
"""

# Session Setup
# -------------

import argparse
import json
import time
import numpy as np
import requests
import psutil
import os
from pathlib import Path
from datetime import datetime

import tritonclient.http as httpclient
from transformers import AutoTokenizer

# Configs
# -------

FASTAPI_URL = "http://localhost:8000"
BENTOML_URL = "http://localhost:3000"
TRITON_URL  = "localhost:8001"    # host:port — no http:// for tritonclient, remember the port is shifted to avoid conflict

PROMPTS = [
    "Once upon a time there was a dog that learned how to make fancy burgers from talking to famous chefs",
    "Isaac Asimov was right about the future of robots artificial intelligence",
    "In a galaxy far far away Dom Cobb was trying to impland an idea into someone's dream, the idea read 42",
]

MAX_NEW_TOKENS = 100
N_RUNS = 3          # number of runs per prompt per server (for averaging)
RESULTS_DIR = Path("benchmarks/results")

# triton config
MODEL_NAME = "gpt2_onnx"
MODEL_ONNX_PATH = Path("model_onnx/model.onnx")

# Fast API client 

def benchmark_fastapi(prompt: str) -> dict:
    response = requests.post(
        f"{FASTAPI_URL}/generate",
        json={
            "prompt": prompt, 
            "max_new_tokens": MAX_NEW_TOKENS
            },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

# Bento ML client

def benchmark_bentoml(prompt: str) -> dict:
    response = requests.post(
        f"{BENTOML_URL}/generate",
        json={
            "request": {
                "prompt": prompt, 
                "max_new_tokens": MAX_NEW_TOKENS
                }
            },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

# Triton

client = httpclient.InferenceServerClient(url=TRITON_URL)
tokenizer = AutoTokenizer.from_pretrained("model_repository/gpt2_onnx")
tokenizer.pad_token = tokenizer.eos_token
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

def benchmark_triton(prompt: str) -> dict:
    """Runs the full generation loop client-side against Triton."""
 
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    process.cpu_percent(interval=None) 

    enc = tokenizer(prompt, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    input_tokens = input_ids.shape[1]
 
    generated_tokens = []
    first_token_time = None

    e2e_start = time.perf_counter()
 
    for _ in range(MAX_NEW_TOKENS):

        logits = triton_forward_pass(input_ids, attention_mask)
 
        # Temperature sampling
        probs = logits[0, -1, :].astype(np.float64)
        probs -= np.max(probs)
        probs = np.exp(probs / 0.9)
        probs /= probs.sum()
        next_token_id = int(np.random.choice(len(probs), p=probs))
        generated_tokens.append(next_token_id)
 
        if first_token_time is None:
            first_token_time = time.perf_counter()
 
        if next_token_id == tokenizer.eos_token_id:
            break
 
        input_ids = np.concatenate(
            [input_ids, np.array([[next_token_id]], dtype=np.int64)], axis=1
        )

        attention_mask = np.concatenate(
            [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
        )
 
    e2e_latency = time.perf_counter() - e2e_start
    ttft = (first_token_time - e2e_start) if first_token_time else e2e_latency

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
        "ttft_s": round(ttft, 4),
        "tpot_msec": round((e2e_latency / output_tokens) * 1000, 2),

        # Throughput
        "tokens_per_second": round(output_tokens / e2e_latency, 2),

        # Efficiency
        "cpu_percent": round(cpu_after, 1),
        "ram_used_mb": round(ram_after - ram_before, 2),
        "peak_ram_mb": round(ram_after, 2),
        "model_file_size_mb": model_file_size_mb,
    }

# Runner

SERVERS = {
    "fastapi": benchmark_fastapi,
    "bentoml": benchmark_bentoml,
    "triton":  benchmark_triton,
}
 
def is_server_up(name: str) -> bool:
    """Quick health check before benchmarking."""
    try:
        if name == "fastapi":
            requests.get(f"{FASTAPI_URL}/health", timeout=3)
        elif name == "bentoml":
            requests.get(f"{BENTOML_URL}/health", timeout=3)
        elif name == "triton":
            requests.get(f"http://{TRITON_URL}/v2/health/ready", timeout=3)
        return True
    except Exception:
        return False
    
def run_server_benchmark(name: str, fn) -> dict:
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name.upper()}")
    print(f"{'='*60}")
 
    if not is_server_up(name):
        print(f"{name} is not reachable — skipping.")
        return {}
 
    all_results = []
 
    for prompt in PROMPTS:
        prompt_results = []
        print(f"\n  Prompt: \"{prompt[:50]}...\"")
 
        for run in range(N_RUNS):
            try:
                result = fn(prompt)
                prompt_results.append(result)
                print(f"    Run {run+1}/{N_RUNS} --> "
                      f"{result.get('tokens_per_second', '?')} tok/s | "
                      f"latency {result.get('latency_sec', '?')}s")
            except Exception as e:
                print(f"    Run {run+1}/{N_RUNS} --> ERROR: {e}")
 
        if prompt_results:
            # Average metrics across runs for this prompt
            avg = {
                "prompt":           prompt,
                "output_tokens":    np.mean([r["output_tokens"] for r in prompt_results]),
                "latency_sec":      round(np.mean([r["latency_sec"] for r in prompt_results]), 4),
                "ttft_sec":         round(np.mean([r.get("ttft_sec", r["latency_sec"]) for r in prompt_results]), 4),
                "tpot_msec":        round(np.mean([r["tpot_msec"] for r in prompt_results]), 2),
                "tokens_per_second":round(np.mean([r["tokens_per_second"] for r in prompt_results]), 2),
                "ram_used_mb":      round(np.mean([r.get("ram_used_mb", 0) for r in prompt_results]), 2),
                "peak_ram_mb":      round(np.mean([r.get("peak_ram_mb", 0) for r in prompt_results]), 2),
                "output":    prompt_results[0].get("text", ""),
            }
            all_results.append(avg)
 
    return {"server": name, "results": all_results}
 
 
def print_summary(all_server_results: list):
    print(f"\n\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Server':<12} {'tok/s':>8} {'latency(s)':>12} {'TTFT(s)':>10} {'TPOT(ms)':>10} {'RAM(MB)':>10}")
    print(f"  {'-'*64}")
 
    for sr in all_server_results:
        if not sr or not sr.get("results"):
            continue
        name = sr["server"]
        results = sr["results"]
        avg_tps     = np.mean([r["tokens_per_second"] for r in results])
        avg_lat     = np.mean([r["latency_sec"] for r in results])
        avg_ttft    = np.mean([r["ttft_sec"] for r in results])
        avg_tpot    = np.mean([r["tpot_msec"] for r in results])
        avg_ram     = np.mean([r["ram_used_mb"] for r in results])
        print(f"  {name:<12} {avg_tps:>8.2f} {avg_lat:>12.4f} {avg_ttft:>10.4f} {avg_tpot:>10.2f} {avg_ram:>10.2f}")
 
 
def save_results(all_server_results: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(all_server_results, f, indent=2)
    print(f"\n  Results saved --> {path}")
 
 
# -----
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument(
        "--server",
        choices=["fastapi", "bentoml", "triton"],
        help="Benchmark a single server (default: all)",
    )
    args = parser.parse_args()
 
    targets = {args.server: SERVERS[args.server]} if args.server else SERVERS
 
    print(f"\nLLM Inference Stack Benchmark")
    print(f"Model: GPT-2 (117M) | Prompts: {len(PROMPTS)} | Runs/prompt: {N_RUNS}")
    print(f"Servers: {list(targets.keys())}")
 
    all_server_results = []
    for name, fn in targets.items():
        result = run_server_benchmark(name, fn)
        if result:
            all_server_results.append(result)
 
    print_summary(all_server_results)
    save_results(all_server_results)