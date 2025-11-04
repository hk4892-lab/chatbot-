from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Optional

from . import config

LOGGER = logging.getLogger("bankbot.slm")


@lru_cache(maxsize=1)
def get_pipe():  # pragma: no cover - heavy dependency
    if not config.USE_SLM:
        LOGGER.info("SLM disabled via configuration")
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Transformers unavailable: %s", exc)
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                torch_dtype=torch.float32,
            )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.2,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        return generator
    except Exception as exc:  # pragma: no cover - model load failures
        LOGGER.error("Failed to load SLM %s: %s", config.MODEL_ID, exc)
        return None


def phi_prompt(system: str, messages: Iterable[dict[str, str]]) -> str:
    segments = [f"<|system|>\n{system.strip()}\n"]
    for message in messages:
        role = message.get("role")
        content = message.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            segments.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            segments.append(f"<|assistant|>\n{content}\n")
    segments.append("<|assistant|>\n")
    return "".join(segments)


def generate(
    system: str,
    messages: Iterable[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> Optional[str]:  # pragma: no cover - depends on heavy model
    pipe = get_pipe()
    if pipe is None:
        return None
    prompt = phi_prompt(system, messages)
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("SLM generation failed: %s", exc)
        return None
    if not outputs:
        return None
    generated = outputs[0]["generated_text"].strip()
    return generated
