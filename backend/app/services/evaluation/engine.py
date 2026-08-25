from backend.app.services.evaluation.base import Evaluator
from backend.app.services.evaluation.models import (
    EvaluationInput,
    EvaluationReport,
    EvaluationResult,
)


class EvaluationEngine:

    def __init__(
        self,
        evaluators: list[Evaluator],
    ):
        self.evaluators = evaluators

    def evaluate_case(
        self,
        evaluation_input: EvaluationInput,
    ) -> list[EvaluationResult]:

        results = []

        for evaluator in self.evaluators:
            result = evaluator.evaluate(
                evaluation_input
            )

            results.append(result)

        return results

    def evaluate_dataset(
        self,
        dataset: list[EvaluationInput],
    ) -> EvaluationReport:

        evaluator_results = {
            evaluator.name: []
            for evaluator in self.evaluators
        }

        for item in dataset:

            results = self.evaluate_case(item)

            for result in results:
                evaluator_results[
                    result.evaluator
                ].append(result)

        overall_scores = {}

        for evaluator_name, results in evaluator_results.items():

            if not results:
                overall_scores[evaluator_name] = 0.0
                continue

            average = sum(
                result.score
                for result in results
            ) / len(results)

            overall_scores[evaluator_name] = round(
                average,
                4,
            )

        return EvaluationReport(
            total_cases=len(dataset),
            evaluator_results=evaluator_results,
            overall_scores=overall_scores,
        )