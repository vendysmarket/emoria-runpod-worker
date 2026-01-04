# handler.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import runpod


BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
LORA_REPO = os.getenv("LORA_REPO", "emoria/master_v1")

# Inference defaults
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))

_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32

tokenizer = None
model = None


def _load() -> None:
    global tokenizer, model
    if model is not None and tokenizer is not None:
        return

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # Base model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=_dtype,
        device_map="auto" if _device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    # Attach LoRA
    model = PeftModel.from_pretrained(base, LORA_REPO)
    model.eval()

    # Small speedups
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _to_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Expected input payload shape (flexible):
    {
      "messages": [{"role":"system|user|assistant", "content":"..."}],
      OR
      "message": "user text",
      "system": "system text"
    }
    """
    if isinstance(payload.get("messages"), list) and payload["messages"]:
        msgs = []
        for m in payload["messages"]:
            role = (m.get("role") or "user").strip()
            content = (m.get("content") or "").strip()
            if content:
                msgs.append({"role": role, "content": content})
        if msgs:
            return msgs

    system = (payload.get("system") or "").strip()
    user = (payload.get("message") or payload.get("prompt") or "").strip()

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if user:
        msgs.append({"role": "user", "content": user})
    return msgs if msgs else [{"role": "user", "content": ""}]


def _generate(messages: List[Dict[str, str]],
              max_new_tokens: int,
              temperature: float,
              top_p: float) -> Dict[str, Any]:
    # Mistral Instruct chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if _device == "cuda":
        input_ids = input_ids.to("cuda")

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(0.0, temperature),
        "top_p": min(1.0, max(0.0, top_p)),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    # Only decode the newly generated part
    gen_tokens = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    dt = time.time() - t0

    return {
        "ok": True,
        "text": text,
        "usage": {
            "prompt_tokens": int(input_ids.shape[-1]),
            "completion_tokens": int(gen_tokens.shape[-1]),
            "total_tokens": int(input_ids.shape[-1] + gen_tokens.shape[-1]),
        },
        "meta": {
            "base_model": BASE_MODEL,
            "lora_repo": LORA_REPO,
            "device": _device,
            "dtype": str(_dtype),
            "latency_sec": round(dt, 3),
        },
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless expects: {"input": {...}}
    """
    try:
        _load()
        payload = event.get("input") or {}

        messages = _to_messages(payload)

        max_new_tokens = int(payload.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else DEFAULT_TEMPERATURE)
        top_p = float(payload.get("top_p") if payload.get("top_p") is not None else DEFAULT_TOP_P)

        return _generate(messages, max_new_tokens, temperature, top_p)

    except Exception as e:
        return {"ok": False, "error": str(e)}


runpod.serverless.start({"handler": handler})
