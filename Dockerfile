FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# --- PREBAKE MODELS INTO IMAGE ---
# Optional Hugging Face token (for private repos)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download

base_repo = "mistralai/Mistral-7B-Instruct-v0.3"
lora_repo = "emoria/master_v1"

base_dir = "/models/base"
lora_dir = "/models/lora"

token = os.getenv("HF_TOKEN") or None

print("Downloading base model:", base_repo)
snapshot_download(
    repo_id=base_repo,
    local_dir=base_dir,
    local_dir_use_symlinks=False,
    token=token
)

print("Downloading LoRA adapter:", lora_repo)
snapshot_download(
    repo_id=lora_repo,
    local_dir=lora_dir,
    local_dir_use_symlinks=False,
    token=token
)

print("Done. Models baked into image.")
PY

# Copy worker code
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
