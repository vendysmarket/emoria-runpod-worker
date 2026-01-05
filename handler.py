import os
import runpod
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_PATH = "/models/base"
LORA_PATH = "/models/lora"

# Offline hardening: runtime ne próbáljon netre menni
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER = None
MODEL = None


def _load_once():
    global TOKENIZER, MODEL
    if TOKENIZER is not None and MODEL is not None:
        return

    # Tokenizer + base model LOCAL-ból
    TOKENIZER = AutoTokenizer.from_pretrained(
        BASE_PATH,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=False,
    )

    # LoRA adapter LOCAL-ból
    MODEL = PeftModel.from_pretrained(
        base,
        LORA_PATH,
        local_files_only=True,
        is_trainable=False,
    )
    MODEL.eval()


def _build_prompt(user_text: str) -> str:
    """
    Mistral-Instruct kompatibilis chat prompt.
    Ha van chat_template, használjuk; különben fallback.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return ""

    try:
        messages = [{"role": "user", "content": user_text}]
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback (nem ideális, de működik)
        return f"<s>[INST] {user_text} [/INST]"


def handler(job):
    try:
        _load_once()

        inp = job.get("input") or {}
        # RunPod UI példában: {"input":{"prompt":"Hello World"}}
        # Te nálad néha: {"input":{"message":"..."}}
        user_text = inp.get("prompt") or inp.get("message") or ""

        prompt = _build_prompt(user_text)
        if not prompt:
            return {"ok": False, "error": "Empty prompt/message."}

        max_new_tokens = int(inp.get("max_new_tokens", 140))
        temperature = float(inp.get("temperature", 0.7))
        top_p = float(inp.get("top_p", 0.9))
        repetition_penalty = float(inp.get("repetition_penalty", 1.05))

        inputs = TOKENIZER(prompt, return_tensors="pt")
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=TOKENIZER.eos_token_id,
            )

        text = TOKENIZER.decode(out[0], skip_special_tokens=True)

        # Ha benne maradt a prompt eleje, vágjuk le egyszerűen
        # (az apply_chat_template-nél általában nem kell, de safe)
        if user_text and user_text in text:
            # a user_text utáni részt próbáljuk visszaadni
            idx = text.rfind(user_text)
            tail = text[idx + len(user_text):].strip()
            if tail:
                text = tail

        return {"ok": True, "text": text.strip()}

    except Exception as e:
        return {"ok": False, "error": str(e)}


# Local dev: python handler.py (opcionális)
if __name__ == "__main__":
    # runpod serverless mode
    runpod.serverless.start({"handler": handler})
