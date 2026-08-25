import json
from pathlib import Path
from typing import Any

from backend.app.services.evaluation.engine import EvaluationEngine
from backend.app.services.evaluation.evaluators import (
    ContainsEvaluator,
    GroundednessEvaluator,
    NumberConsistencyEvaluator,
    SourceRetrievalEvaluator,
)
from backend.app.services.evaluation.models import EvaluationInput


class RAGEvaluationRunner:

    def __init__(self):
        self.engine = EvaluationEngine(
            evaluators=[
                ContainsEvaluator(),
                NumberConsistencyEvaluator(),
                SourceRetrievalEvaluator(),
                GroundednessEvaluator(),
            ]
        )

    def load_dataset(
        self,
        path: str,
    ) -> list[dict[str, Any]]:

        dataset = []

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                dataset.append(json.loads(line))

        return dataset

    def evaluate_response(
        self,
        item: dict[str, Any],
        response: dict[str, Any],
        context: str,
    ):

        evaluation_input = EvaluationInput(
            question=item["question"],
            output=response.get("answer", ""),
            expected_output=item.get("expected_output"),
            context=context,
            retrieved_sources=response.get(
                "sources",
                [],
            ),
        )

        return self.engine.evaluate_case(
            evaluation_input
        )

    def calculate_summary(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, float]:

        scores: dict[str, list[float]] = {}

        for item in results:

            for result in item["evaluations"]:

                name = result.evaluator

                if name not in scores:
                    scores[name] = []

                scores[name].append(
                    result.score
                )

        summary = {}

        for name, values in scores.items():

            summary[name] = round(
                sum(values) / len(values),
                4,
            )

        return summary

    def save_report(
        self,
        report: dict[str, Any],
        path: str,
    ):

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )