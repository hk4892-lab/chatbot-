"""Offline evaluation script for the chatbot."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List

from bankbot.app.core.engine import ChatEngine
from bankbot.app.core.policy import DEFAULT_THRESHOLD
from bankbot.app.core.render import format_answer


@dataclass
class Prompt:
    text: str
    expected_contains: str


def load_prompts(path: Path) -> List[Prompt]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [Prompt(text=row["prompt"], expected_contains=row["expected_contains"]) for row in reader]


def evaluate(prompts: List[Prompt], engine: ChatEngine, threshold: float, use_sentence_transformers: bool) -> None:
    answers: List[str] = []
    confidences: List[float] = []
    refusals = 0
    for prompt in prompts:
        result = engine.run(
            prompt.text,
            threshold=threshold,
            use_sentence_transformers=use_sentence_transformers,
        )
        decision = result.decision
        answer = decision.answer or ""
        if answer:
            answer = format_answer(answer, decision.citations)
        if decision.route in {"ask", "escalate"}:
            refusals += 1
        answers.append(answer)
        confidences.append(decision.confidence)
        print(
            json.dumps(
                {"prompt": prompt.text, "route": decision.route, "answer": answer},
                ensure_ascii=False,
            )
        )

    accuracy = sum(1 for prompt, answer in zip(prompts, answers) if prompt.expected_contains.lower() in answer.lower()) / len(prompts)
    refusal_rate = refusals / len(prompts)
    avg_conf = mean(confidences) if confidences else 0.0
    print(f"Top-1 accuracy: {accuracy:.2f}")
    print(f"Refusal rate: {refusal_rate:.2f}")
    print(f"Average confidence: {avg_conf:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for BankBot")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).resolve().with_name("prompts.csv"),
        help="Path to a CSV with 'prompt' and 'expected_contains' columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold to apply during evaluation.",
    )
    parser.add_argument(
        "--use-sentence-transformers",
        action="store_true",
        help="Enable sentence-transformers if installed.",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    data_path = Path(__file__).resolve().parents[1] / "app" / "data"
    log_path = Path(__file__).resolve().parents[1] / "logs" / "audit.log"
    engine = ChatEngine(data_path=data_path, log_path=log_path)
    evaluate(prompts, engine, args.threshold, args.use_sentence_transformers)


if __name__ == "__main__":
    main()
