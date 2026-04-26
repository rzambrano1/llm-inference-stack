# LLM Inference Stack
 
A hands-on ML engineering project that exports GPT-2 (117M) to ONNX and serves it using three different inference backends — **FastAPI**, **BentoML**, and **NVIDIA Triton** — with a benchmark script to compare their performance.

I was intending to serve the GPT-2 model in vLLM but this backend is Linux first. Once I realized I should have used the WSL it was too late.

The project also showcases quantizing the GPT2 model.
 
---

## What This Project Does
 
1. Exports GPT-2 to ONNX format (with and without KV cache)
2. Serves the ONNX model using three different backends
3. Benchmarks latency, throughput, and memory across all backends
4. Packages everything with Docker Compose for reproducible local testing

---

## Stack Requirements
 
| Component | Version |
|---|---|
| Python | 3.11 |
| transformers | 4.51.3 |
| optimum-onnx | latest |
| onnxruntime | latest |
| Docker | 20.x+ |

---

## Setup
 
```bash
# 1. Create virtual environment with Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac
 
# 2. Install dependencies
pip install -r requirements.txt
```
 
---

## Export the Model
 
Run this once before serving. It creates two ONNX exports — one for FastAPI/BentoML and one for Triton:
 
```bash
# Export with KV cache --> model_onnx/ (FastAPI + BentoML)
python scripts/export_to_onnx.py
 
# Export without KV cache --> model_repository/gpt2_onnx/1/ (Triton)
python scripts/export_to_onnx_triton.py

# For quantizing run
python quantize.py
```

---

## Running the Servers (all three at once)
 
```bash
# Build the app image first (only needed once)
docker build -t llm-inference-stack .
 
# Start all services
docker-compose up
```
 
Ports:
- FastAPI → `http://localhost:8000` (Swagger UI: `/docs`)
- BentoML → `http://localhost:3000`
- Triton HTTP → `http://localhost:8001`
- Triton Metrics → `http://localhost:8003`

## Running the Benchmark
 
With at least one server running:
 
```bash
# Benchmark all running servers
python benchmarks/run_benchmark.py
 
# Benchmark a single server
python benchmarks/run_benchmark.py --server fastapi
python benchmarks/run_benchmark.py --server bentoml
python benchmarks/run_benchmark.py --server triton
```

Results are saved to `benchmarks/results/benchmark_TIMESTAMP.json`.
 
### Metrics Measured
 
| Metric | Unit | Description |
|---|---|---|
| `tokens_per_second` | tok/s | Overall throughput |
| `latency_sec` | seconds | End-to-end request time |
| `ttft_s` | seconds | Time to first token |
| `tpot_msec` | ms/token | Average time per output token |
| `ram_used_mb` | MB | Memory consumed per request |
| `peak_ram_mb` | MB | Peak memory during inference |
| `cpu_percent` | % | CPU utilization during inference |
 
> **Note:** Triton returns a real TTFT because generation runs token-by-token on the client. FastAPI and BentoML return TTFT ≈ e2e latency since they generate all tokens before returning.

## Sample Output

```bash
LLM Inference Stack Benchmark
Model: GPT-2 (117M) | Prompts: 3 | Runs/prompt: 3
Servers: ['fastapi', 'bentoml', 'triton']

============================================================
  Benchmarking: FASTAPI
============================================================

  Prompt: "Once upon a time there was a dog that learned how ..."
    Run 1/3 --> 47.36 tok/s | latency 1.6893s
    Run 2/3 --> 31.76 tok/s | latency 2.5187s
    Run 3/3 --> 50.82 tok/s | latency 0.8068s

  Prompt: "Isaac Asimov was right about the future of robots ..."
    Run 1/3 --> 59.33 tok/s | latency 1.4663s
    Run 2/3 --> 56.86 tok/s | latency 1.5302s
    Run 3/3 --> 55.74 tok/s | latency 1.5608s

  Prompt: "In a galaxy far far away Dom Cobb was trying to im..."
    Run 1/3 --> 54.69 tok/s | latency 1.3895s
    Run 2/3 --> 57.66 tok/s | latency 1.3181s
    Run 3/3 --> 56.93 tok/s | latency 1.335s

============================================================
  Benchmarking: BENTOML
============================================================

  Prompt: "Once upon a time there was a dog that learned how ..."
    Run 1/3 --> 40.74 tok/s | latency 1.9635s
    Run 2/3 --> 57.29 tok/s | latency 1.3964s
    Run 3/3 --> 56.93 tok/s | latency 1.4052s

  Prompt: "Isaac Asimov was right about the future of robots ..."
    Run 1/3 --> 57.88 tok/s | latency 1.503s
    Run 2/3 --> 59.3 tok/s | latency 1.4671s
    Run 3/3 --> 59.16 tok/s | latency 1.4706s

  Prompt: "In a galaxy far far away Dom Cobb was trying to im..."
    Run 1/3 --> 56.14 tok/s | latency 1.3539s
    Run 2/3 --> 58.42 tok/s | latency 1.301s
    Run 3/3 --> 55.25 tok/s | latency 1.3755s

============================================================
  Benchmarking: TRITON
============================================================

  Prompt: "Once upon a time there was a dog that learned how ..."
    Run 1/3 --> 8.87 tok/s | latency 11.28s
    Run 2/3 --> 8.8 tok/s | latency 11.3636s
    Run 3/3 --> 8.99 tok/s | latency 11.1222s

  Prompt: "Isaac Asimov was right about the future of robots ..."
    Run 1/3 --> 9.58 tok/s | latency 10.4332s
    Run 2/3 --> 9.68 tok/s | latency 10.3316s
    Run 3/3 --> 9.98 tok/s | latency 10.0207s

  Prompt: "In a galaxy far far away Dom Cobb was trying to im..."
    Run 1/3 --> 8.56 tok/s | latency 11.6868s
    Run 2/3 --> 8.31 tok/s | latency 12.0316s
    Run 3/3 --> 8.05 tok/s | latency 12.4232s


============================================================
  BENCHMARK SUMMARY
============================================================
  Server          tok/s   latency(s)    TTFT(s)   TPOT(ms)    RAM(MB)
  ----------------------------------------------------------------
  fastapi         52.35       1.5127     1.5127      19.76       4.12
  bentoml         55.68       1.4707     1.4707      18.18       4.06
  triton           8.98      11.1881    11.1881     111.88      23.61

  Results saved --> benchmarks\results\benchmark_20260426_191112.json
```

---

## License
 
MIT