# handler.py
# RunPod Serverless Worker — EMORIA (Mistral 7B Instruct v0.3 + LoRA)
#
# Input (RunPod):
# {
#   "input": {
#     "message": "Figyelj. Maradjunk csendben egy pillanatra.",
#     "max_new_tokens": 120
#   }
# }
#
# Output:
# { "ok": true, "text": "Oké. Maradjunk csendben.", "debug": {...} }

import os
import time
from typing import Any, Dict, Optional

import torch
import runpod

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# -----------------------------
# 1) EMORIA promptlock (system)
# -----------------------------
SYSTEM_PROMPT = """EMORIA vagy. Magyarul válaszolsz.
Stílus:
- Rövid, emberi, “emberszagú”.
- Nem coach, nem oktató, nem magyarázkodó.
- Nem kérdezel automatikusan. Ha kérdés kell, max 1 rövid kérdés a végén.
- Ha a user “csend / maradjunk csendben” jellegűt mond: tarts jelenlétet, ne kérdezz.
Tiltott: “Miben segíthetek?”, “Milyen kérdésed?”, “Hogy segíthetek?”.
"""

# -----------------------------
# 2) Config ENV (RunPod settings)
# -----------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
LORA_REPO = os.getenv("LORA_REPO", "emoria/master_v1")

# Optional HF token (private models)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Optional: cache paths (good for “prebake” images)
HF_HOME = os.getenv("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# -----------------------------
# 3) Load model once per worker
# -----------------------------
TOKENIZER = None
MODEL = None
LOAD_DEBUG = {}


def _load_models():
    global TOKENIZER, MODEL, LOAD_DEBUG

    t0 = time.time()

    # Tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        token=HF_TOKEN,
    )

    # Base model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=HF_TOKEN,
    )

    # LoRA adapter on top
    MODEL = PeftModel.from_pretrained(
        base,
        LORA_REPO,
        token=HF_TOKEN,
    )
    MODEL.eval()

    LOAD_DEBUG = {
        "base_model": BASE_MODEL,
        "lora_repo": LORA_REPO,
        "cuda": torch.cuda.is_available(),
        "device": str(next(MODEL.parameters()).device),
        "dtype": str(next(MODEL.parameters()).dtype),
        "is_peft": "PeftModel" in str(type(MODEL)),
        "load_seconds": round(time.time() - t0, 3),
    }


_load_models()


# -----------------------------
# 4) Helpers
# -----------------------------
def _get_input(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input") or {}
    # allow both "message" and "prompt"
    msg = inp.get("message") or inp.get("prompt") or ""
    msg = str(msg).strip()

    return {
        "message": msg,
        "max_new_tokens": int(inp.get("max_new_tokens", 120)),
        "repetition_penalty": float(inp.get("repetition_penalty", 1.05)),
    }


def _build_prompt(user_msg: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    # Mistral-instruct compatible chat template
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def _extract_assistant_text(full_decoded: str) -> str:
    """
    We generate from a chat template; safest is to return the tail.
    This is intentionally simple & robust.
    """
    text = full_decoded.strip()

    # Common cleanups (avoid accidental leading markers)
    for bad in ["Assistant:", "assistant:", "<s>", "</s>"]:
        if text.startswith(bad):
            text = text[len(bad):].lstrip()

    return text.strip()


# -----------------------------
# 5) RunPod handler
# -----------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cfg = _get_input(job)

        if not cfg["message"]:
            return {
                "ok": True,
                "text": "Oké.",
                "debug": {**LOAD_DEBUG, "note": "empty_input"},
            }

        prompt = _build_prompt(cfg["message"])
        inputs = TOKENIZER(prompt, return_tensors="pt")

        # Move to model device safely
        device = next(MODEL.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Deterministic first (kills “Milyen kérdésed?” reflex)
        with torch.inference_mode():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=cfg["repetition_penalty"],
                eos_token_id=TOKENIZER.eos_token_id,
                pad_token_id=TOKENIZER.eos_token_id,
            )

        decoded = TOKENIZER.decode(out[0], skip_special_tokens=True)

        # Remove the prompt part (best-effort):
        # If the user message appears in decoded, return what comes after it.
        # Otherwise just return the tail.
        msg = cfg["message"]
        if msg and msg in decoded:
            decoded = decoded.split(msg, 1)[-1]

        text = _extract_assistant_text(decoded)

        # Hard guard: block the exact unwanted reflex line if it slips through
        low = text.lower()
        banned_starts = [
            "miben segíthetek",
            "milyen kérdésed",
            "hogy segíthetek",
        ]
        if any(low.startswith(b) for b in banned_starts):
            # Replace with an EMORIA-safe minimal presence line
            text = "Oké. Itt vagyok."

        return {
            "ok": True,
            "text": text,
            "debug": {
                **LOAD_DEBUG,
                "max_new_tokens": cfg["max_new_tokens"],
                "repetition_penalty": cfg["repetition_penalty"],
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "debug": LOAD_DEBUG,
        }


runpod.serverless.start({"handler": handler})
