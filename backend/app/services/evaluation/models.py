from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationInput:
    question: str
    output: str
    expected_output: str | None = None
    context: str = ""
    retrieved_sources: list[dict[str, Any]] = field(
        default_factory=list
    )


@dataclass
class EvaluationResult:
    evaluator: str
    score: float
    passed: bool
    reason: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EvaluationReport:
    total_cases: int
    evaluator_results: dict[str, list[EvaluationResult]]
    overall_scores: dict[str, float]