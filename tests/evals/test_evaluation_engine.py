from backend.app.services.evaluation.engine import EvaluationEngine
from backend.app.services.evaluation.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    GroundednessEvaluator,
    NumberConsistencyEvaluator,
    SourceRetrievalEvaluator,
)
from backend.app.services.evaluation.models import EvaluationInput


def test_exact_match():

    engine = EvaluationEngine(
        evaluators=[
            ExactMatchEvaluator(),
        ]
    )

    result = engine.evaluate_case(
        EvaluationInput(
            question="Test",
            output="Correct answer",
            expected_output="Correct answer",
        )
    )

    assert result[0].passed is True
    assert result[0].score == 1.0


def test_contains():

    engine = EvaluationEngine(
        evaluators=[
            ContainsEvaluator(),
        ]
    )

    result = engine.evaluate_case(
        EvaluationInput(
            question="Test",
            output="The answer is correct.",
            expected_output="correct",
        )
    )

    assert result[0].passed is True


def test_number_consistency():

    engine = EvaluationEngine(
        evaluators=[
            NumberConsistencyEvaluator(),
        ]
    )

    result = engine.evaluate_case(
        EvaluationInput(
            question="Test",
            output="The account unlocks after 30 minutes.",
            context="The account automatically unlocks after 30 minutes.",
        )
    )

    assert result[0].passed is True


def test_unsupported_number():

    engine = EvaluationEngine(
        evaluators=[
            NumberConsistencyEvaluator(),
        ]
    )

    result = engine.evaluate_case(
        EvaluationInput(
            question="Test",
            output="The account unlocks after 3 days.",
            context="The account automatically unlocks after 30 minutes.",
        )
    )

    assert result[0].passed is False


def test_source_retrieval():

    engine = EvaluationEngine(
        evaluators=[
            SourceRetrievalEvaluator(),
        ]
    )

    result = engine.evaluate_case(
        EvaluationInput(
            question="Test",
            output="Answer",
            retrieved_sources=[
                {
                    "document_id": 4,
                    "chunk_id": 2,
                }
            ],
        )
    )

    assert result[0].passed is True


def test_groundedness():

    engine = EvaluationEngine(
        evaluators=[
            GroundednessEvaluator(),
        ]
    )

    result = engine.evaluate_case(
        EvaluationInput(
            question="Test",
            output="Account temporarily locked.",
            context="The account is temporarily locked after five unsuccessful login attempts.",
        )
    )

    assert result[0].passed is True


def test_dataset_aggregation():

    engine = EvaluationEngine(
        evaluators=[
            ExactMatchEvaluator(),
        ]
    )

    dataset = [
        EvaluationInput(
            question="Q1",
            output="A",
            expected_output="A",
        ),
        EvaluationInput(
            question="Q2",
            output="B",
            expected_output="A",
        ),
    ]

    report = engine.evaluate_dataset(dataset)

    assert report.total_cases == 2
    assert report.overall_scores["exact_match"] == 0.5