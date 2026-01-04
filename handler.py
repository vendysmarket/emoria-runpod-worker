import os
import time
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
LORA_REPO  = os.getenv("LORA_REPO", "emoria/master_v1")
HF_TOKEN   = os.getenv("HF_TOKEN", "").strip()

# Load at cold-start (Serverless worker boot)
def _load():
    tok_kwargs = {}
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, **tok_kwargs)

    model_kwargs = {}
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        **model_kwargs,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(model, LORA_REPO, **model_kwargs)
    model.eval()

    return tokenizer, model


TOKENIZER, MODEL = _load()


def _build_prompt(message: str) -> str:
    # Mistral Instruct chat template
    messages = [{"role": "user", "content": message}]
    return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def handler(job):
    try:
        inp = job.get("input", {}) or {}
        msg = inp.get("message") or inp.get("prompt") or ""
        msg = str(msg).strip()
        if not msg:
            return {"ok": False, "error": "missing_message"}

        prompt = _build_prompt(msg)

        inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

        with torch.no_grad():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=int(inp.get("max_new_tokens", 120)),
                do_sample=True,
                temperature=float(inp.get("temperature", 0.7)),
                top_p=float(inp.get("top_p", 0.9)),
                repetition_penalty=float(inp.get("repetition_penalty", 1.05)),
                eos_token_id=TOKENIZER.eos_token_id,
            )

        text = TOKENIZER.decode(out[0], skip_special_tokens=True)

        # Strip the prompt part if needed
        # (simple heuristic: return last assistant-like segment)
        if msg in text:
            text = text.split(msg, 1)[-1].strip()

        return {"ok": True, "text": text}

    except Exception as e:
        return {"ok": False, "error": str(e)}


runpod.serverless.start({"handler": handler})
