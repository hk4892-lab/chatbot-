"""Evaluation script for the chatbot."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List

import requests

API_URL = "http://localhost:8000"


@dataclass
class Prompt:
    text: str
    expected_contains: str


def load_prompts(path: Path) -> List[Prompt]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [Prompt(text=row["prompt"], expected_contains=row["expected_contains"]) for row in reader]


def evaluate(prompts: List[Prompt]) -> None:
    answers: List[str] = []
    confidences: List[float] = []
    refusals = 0
    for prompt in prompts:
        response = requests.post(f"{API_URL}/chat/turn", json={"message": prompt.text, "threshold": 0.18}, timeout=10)
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("answer") or ""
        if payload.get("route") in {"ask", "escalate"}:
            refusals += 1
        answers.append(answer)
        confidences.append(payload.get("confidence", 0.0))
        print(json.dumps({"prompt": prompt.text, "route": payload.get("route"), "answer": answer}, ensure_ascii=False))

    accuracy = sum(1 for prompt, answer in zip(prompts, answers) if prompt.expected_contains.lower() in answer.lower()) / len(prompts)
    refusal_rate = refusals / len(prompts)
    avg_conf = mean(confidences) if confidences else 0.0
    print(f"Top-1 accuracy: {accuracy:.2f}")
    print(f"Refusal rate: {refusal_rate:.2f}")
    print(f"Average confidence: {avg_conf:.2f}")


def main() -> None:
    prompts_path = Path(__file__).resolve().with_name("prompts.csv")
    prompts = load_prompts(prompts_path)
    evaluate(prompts)


if __name__ == "__main__":
    main()
