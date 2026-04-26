# LLM Inference Stack — Makefile
# ================================

PYTHON     = .venv/Scripts/python     
PIP        = .venv/Scripts/pip
PORT_FASTAPI  = 8000
PORT_BENTOML  = 3000
PORT_TRITON   = 8000
MODEL_REPO = $(shell pwd)/model_repository
TRITON_IMAGE = nvcr.io/nvidia/tritonserver:24.10-py3
 
# Installation
install:
	$(PIP) install transformers==4.51.3 optimum-onnx onnxruntime \
	               fastapi uvicorn bentoml psutil requests tritonclient[http]
 
# Export
export:
	@echo "Exporting GPT-2 to ONNX (with cache) --> model_onnx/"
	$(PYTHON) scripts/export_to_onnx.py
	@echo "Exporting GPT-2 to ONNX (no cache)   --> model_repository/gpt2_onnx/1/"
	$(PYTHON) scripts/export_to_onnx_triton.py
 
# Serving
fastapi:
	@echo "Starting FastAPI on port $(PORT_FASTAPI)..."
	@echo "Swagger UI --> http://localhost:$(PORT_FASTAPI)/docs"
	.venv/Scripts/uvicorn serve.fastapi_app_onnx:app \
	    --host 0.0.0.0 --port $(PORT_FASTAPI)
 
bentoml:
	@echo "Starting BentoML on port $(PORT_BENTOML)..."
	@echo "UI --> http://localhost:$(PORT_BENTOML)"
	.venv/Scripts/bentoml serve serve.bentoml_onnx_service:GPT2OnnxService \
	    --port $(PORT_BENTOML)
 
triton:
	@echo "Starting Triton via Docker on ports 8000/8001/8002..."
	@echo "Health --> http://localhost:8000/v2/health/ready"
	@echo "Metrics --> http://localhost:8002/metrics"
	docker run --rm \
	    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
	    -v "$(MODEL_REPO):/models" \
	    $(TRITON_IMAGE) \
	    tritonserver --model-repository=/models
 
stop-triton:
	@echo "Stopping Triton container..."
	docker stop $$(docker ps -q --filter ancestor=$(TRITON_IMAGE)) 2>/dev/null || echo "No Triton container running"
 
# Benchmark 
benchmark:
	@echo "Running benchmark against all servers..."
	$(PYTHON) benchmarks/run_benchmark.py
 
benchmark-fastapi:
	$(PYTHON) benchmarks/run_benchmark.py --server fastapi
 
benchmark-bentoml:
	$(PYTHON) benchmarks/run_benchmark.py --server bentoml
 
benchmark-triton:
	$(PYTHON) benchmarks/run_benchmark.py --server triton
 
# Cleanup
clean:
	@echo "Removing exported model files..."
	rm -rf model_onnx/ model_onnx_nocache/ model_onnx_quantized/
	@echo "Done."
 
.PHONY: install export fastapi bentoml triton stop-triton benchmark \
        benchmark-fastapi benchmark-bentoml benchmark-triton clean
 