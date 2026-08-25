import re

from backend.app.services.evaluation.base import Evaluator
from backend.app.services.evaluation.models import (
    EvaluationInput,
    EvaluationResult,
)


class ExactMatchEvaluator(Evaluator):

    name = "exact_match"

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationResult:

        expected = evaluation_input.expected_output

        if expected is None:
            return EvaluationResult(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason="Expected output is required.",
            )

        actual = evaluation_input.output.strip()
        expected = expected.strip()

        passed = actual.lower() == expected.lower()

        return EvaluationResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=(
                "Output matches expected answer."
                if passed
                else "Output does not match expected answer."
            ),
        )


class ContainsEvaluator(Evaluator):

    name = "contains"

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationResult:

        expected = evaluation_input.expected_output

        if not expected:
            return EvaluationResult(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason="Expected output is required.",
            )

        required_facts = [
            fact.strip().lower()
            for fact in expected.split(";")
            if fact.strip()
        ]

        actual = evaluation_input.output.lower()

        matched = [
            fact
            for fact in required_facts
            if fact in actual
        ]

        score = (
            len(matched) / len(required_facts)
            if required_facts
            else 0.0
        )

        passed = score >= 0.75

        return EvaluationResult(
            evaluator=self.name,
            score=round(score, 4),
            passed=passed,
            reason=(
                "Most required facts were found."
                if passed
                else "Too many required facts were missing."
            ),
            metadata={
                "required_facts": required_facts,
                "matched_facts": matched,
                "missing_facts": [
                    fact
                    for fact in required_facts
                    if fact not in matched
                ],
            },
        )


class NumberConsistencyEvaluator(Evaluator):

    name = "number_consistency"

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationResult:

        output_numbers = set(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                evaluation_input.output,
            )
        )

        context_numbers = set(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                evaluation_input.context,
            )
        )

        unsupported = output_numbers - context_numbers

        passed = not unsupported

        return EvaluationResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=(
                "All numeric claims exist in the retrieved context."
                if passed
                else (
                    "Unsupported numbers found: "
                    + ", ".join(sorted(unsupported))
                )
            ),
            metadata={
                "output_numbers": sorted(output_numbers),
                "context_numbers": sorted(context_numbers),
                "unsupported_numbers": sorted(unsupported),
            },
        )


class SourceRetrievalEvaluator(Evaluator):

    name = "source_retrieval"

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationResult:

        sources = evaluation_input.retrieved_sources

        passed = len(sources) > 0

        return EvaluationResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=(
                "Relevant sources were retrieved."
                if passed
                else "No sources were retrieved."
            ),
            metadata={
                "source_count": len(sources),
            },
        )


class GroundednessEvaluator(Evaluator):

    name = "groundedness"

    def evaluate(
        self,
        evaluation_input: EvaluationInput,
    ) -> EvaluationResult:

        if not evaluation_input.output.strip():
            return EvaluationResult(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason="Empty model output.",
            )

        output_words = set(
            re.findall(
                r"\b[a-zA-Z]{4,}\b",
                evaluation_input.output.lower(),
            )
        )

        context_words = set(
            re.findall(
                r"\b[a-zA-Z]{4,}\b",
                evaluation_input.context.lower(),
            )
        )

        if not output_words:
            return EvaluationResult(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason="No meaningful words in output.",
            )

        overlap = output_words.intersection(context_words)

        score = len(overlap) / len(output_words)

        passed = score >= 0.60

        return EvaluationResult(
            evaluator=self.name,
            score=round(score, 4),
            passed=passed,
            reason=(
                "Output has sufficient lexical overlap with context."
                if passed
                else "Output has insufficient overlap with context."
            ),
            metadata={
                "matched_words": len(overlap),
                "output_words": len(output_words),
            },
        )