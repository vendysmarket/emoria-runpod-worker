FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Speed: avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Optional defaults (can override in RunPod ENV)
ENV BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
ENV LORA_REPO="emoria/master_v1"
ENV MAX_NEW_TOKENS="160"
ENV TEMPERATURE="0.7"
ENV TOP_P="0.9"

CMD ["python", "-u", "handler.py"]
