from __future__ import annotations

import math
import uuid
from typing import Dict


def emi_calculator(principal: float, annual_rate: float, months: int) -> Dict[str, float]:
    monthly_rate = annual_rate / (12 * 100)
    if months <= 0:
        raise ValueError("Months must be positive")
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = principal * monthly_rate * math.pow(1 + monthly_rate, months) / (math.pow(1 + monthly_rate, months) - 1)
    total_payment = emi * months
    total_interest = total_payment - principal
    return {
        "emi": round(float(emi), 2),
        "total_interest": round(float(total_interest), 2),
        "total_payment": round(float(total_payment), 2),
    }


def block_card_ticket(name: str, last4: str, reason: str) -> Dict[str, str]:
    ticket_id = str(uuid.uuid4())[:8]
    return {
        "ticket_id": ticket_id,
        "status": "raised",
        "message": f"Ticket for {name} ending {last4} due to {reason} is raised.",
    }


INTEREST_TABLE = {
    "savings": 3.5,
    "fixed_deposit": 6.2,
    "home_loan": 8.1,
}


def interest_rates(product: str) -> Dict[str, float | str]:
    key = product.lower().strip()
    rate = INTEREST_TABLE.get(key, 4.0)
    return {"product": key, "rate_percent": rate}
