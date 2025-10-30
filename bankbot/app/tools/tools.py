"""Mock tool implementations."""
from __future__ import annotations

import math
import random
import string
from typing import Dict


def calculate_emi(P: float, annual_rate_percent: float, months: int) -> float:
    """Calculate the EMI using the standard formula."""
    if months <= 0:
        raise ValueError("months must be positive")
    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        return round(P / months, 2)
    numerator = P * monthly_rate * math.pow(1 + monthly_rate, months)
    denominator = math.pow(1 + monthly_rate, months) - 1
    emi = numerator / denominator
    return round(emi, 2)


def block_card(tokenized_card: str, reason: str) -> str:
    """Mock card blocking tool returning a ticket ID."""
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"CARD-BLOCK-{suffix}"


def fetch_rate(product: str) -> Dict[str, object]:
    """Return mock rate information."""
    product_key = product.strip().lower()
    rates = {
        "savings": {"interest_rate_percent": 3.5},
        "fixed deposit": {"interest_rate_percent": 6.75},
        "home loan": {"interest_rate_percent": 8.1},
        "personal loan": {"interest_rate_percent": 11.9},
    }
    return rates.get(product_key, {"interest_rate_percent": 0.0, "note": "No data"})


__all__ = ["calculate_emi", "block_card", "fetch_rate"]
