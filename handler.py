import os
import json
import torch

import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_ROOT = os.getenv("MODEL_ROOT", "/models")
BASE_DIR = os.path.join(MODEL_ROOT, "base")
LORA_DIR = os.path.join(MODEL_ROOT, "lora")

# Hard EMORIA-style system prompt (you can tighten later)
EMORIA_SYSTEM = """Te EMORIA vagy.
Stílus: rövid, direkt, emberi. Nincs magyarázkodás, nincs meta, nincs coach-hang.
Ne köszönj.
Max 2 mondat.
A végén legfeljebb 1 kérdés (ha tényleg indokolt).
Ne írd: "User:" "Assistant:".
Nyelv: a felhasználó nyelve (ha magyarul ír, magyarul válaszolsz).
"""

# Load once per worker
TOKENIZER = None
MODEL = None


def _load_model_once():
    global TOKENIZER, MODEL
    if TOKENIZER is not None and MODEL is not None:
        return

    # Force offline load (no runtime HF download)
    TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_DIR,
        local_files_only=True,
        use_fast=True
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply LoRA (also local)
    MODEL = PeftModel.from_pretrained(
        base,
        LORA_DIR,
        local_files_only=True
    )
    MODEL.eval()


def _build_prompt(user_text: str) -> str:
    messages = [
        {"role": "system", "content": EMORIA_SYSTEM},
        {"role": "user", "content": user_text.strip()},
    ]

    # Mistral Instruct chat template
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def _postprocess(text: str) -> str:
    # Basic cleanup
    t = text.strip()

    # Defensive trims: remove obvious generic greetings if they slip through
    for bad in ["Szia", "Hello", "Igen, megérkeztem", "Megérkeztem"]:
        if t.lower().startswith(bad.lower()):
            t = t[len(bad):].lstrip(" .,-:;")
    return t.strip()


def handler(job):
    try:
        _load_model_once()

        inp = job.get("input", {}) or {}
        user_text = inp.get("message") or inp.get("prompt") or ""
        if not user_text.strip():
            return {"ok": False, "error": "Missing input.message (or input.prompt)."}

        prompt = _build_prompt(user_text)

        inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

        max_new_tokens = int(inp.get("max_new_tokens", 90))
        temperature = float(inp.get("temperature", 0.6))
        top_p = float(inp.get("top_p", 0.9))
        repetition_penalty = float(inp.get("repetition_penalty", 1.08))

        with torch.inference_mode():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=TOKENIZER.eos_token_id,
            )

        decoded = TOKENIZER.decode(out[0], skip_special_tokens=True)

        # Remove the prompt portion robustly
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]

        final_text = _postprocess(decoded)

        return {"ok": True, "text": final_text}

    except Exception as e:
        return {"ok": False, "error": str(e)}


runpod.serverless.start({"handler": handler})
