from abc import ABC, abstractmethod

from backend.app.services.evaluation.models import (
    EvaluationInput,
    EvaluationResult,
)


class Evaluator(ABC):

    name: str

    @abstractmethod
    def evaluate(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationResult:
        raise NotImplementedError