"""Cost tracking utilities for API usage."""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import json


@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation: str = "extraction"


class CostTracker:
    """Track API costs across experiments."""

    def __init__(self):
        self.calls: List[APICall] = []
        self.budget_limit: float = 100.0  # Default $100 limit

    def log_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        operation: str = "extraction"
    ):
        """Log an API call."""
        self.calls.append(APICall(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation=operation
        ))

    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return sum(c.cost for c in self.calls)

    def get_cost_by_provider(self) -> Dict[str, float]:
        """Get cost breakdown by provider."""
        costs = {}
        for call in self.calls:
            costs[call.provider] = costs.get(call.provider, 0) + call.cost
        return costs

    def get_summary(self) -> Dict:
        """Get cost summary."""
        return {
            "total_cost": self.get_total_cost(),
            "total_calls": len(self.calls),
            "total_input_tokens": sum(c.input_tokens for c in self.calls),
            "total_output_tokens": sum(c.output_tokens for c in self.calls),
            "by_provider": self.get_cost_by_provider()
        }

    def check_budget(self) -> bool:
        """Check if within budget."""
        return self.get_total_cost() < self.budget_limit

    def save(self, path: str):
        """Save cost log to file."""
        data = {
            "summary": self.get_summary(),
            "calls": [
                {
                    "timestamp": c.timestamp.isoformat(),
                    "provider": c.provider,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost": c.cost,
                    "operation": c.operation
                }
                for c in self.calls
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
