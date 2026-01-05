FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- Cache locations (avoid tiny default disk cache paths)
ENV MODEL_ROOT=/models
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HUGGINGFACE_HUB_CACHE=/models/hf
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps (git + ca-certs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY handler.py /app/handler.py

# ---- Pre-download BASE + LORA into the IMAGE (offline runtime)
# IMPORTANT: this assumes BOTH repos are public.
ENV BASE_REPO=mistralai/Mistral-7B-Instruct-v0.3
ENV LORA_REPO=emoria/master_v1

RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download

base_repo = os.environ["BASE_REPO"]
lora_repo = os.environ["LORA_REPO"]

base_dir = os.path.join(os.environ["MODEL_ROOT"], "base")
lora_dir = os.path.join(os.environ["MODEL_ROOT"], "lora")

os.makedirs(base_dir, exist_ok=True)
os.makedirs(lora_dir, exist_ok=True)

print("Downloading BASE:", base_repo, "->", base_dir)
snapshot_download(
    repo_id=base_repo,
    local_dir=base_dir,
    local_dir_use_symlinks=False,
)

print("Downloading LoRA:", lora_repo, "->", lora_dir)
snapshot_download(
    repo_id=lora_repo,
    local_dir=lora_dir,
    local_dir_use_symlinks=False,
)
print("DONE.")
PY

# Runpod serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
