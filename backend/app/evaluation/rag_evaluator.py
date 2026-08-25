import json
from pathlib import Path

from backend.app.services.rag_service import RAGService


class RAGEvaluator:

    def __init__(self):
        self.rag_service = RAGService()

    def load_dataset(self, path: str):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def keyword_score(
        self,
        answer: str,
        expected_keywords: list[str],
    ) -> float:

        answer_lower = answer.lower()

        if not expected_keywords:
            return 1.0

        matched = sum(
            keyword.lower() in answer_lower
            for keyword in expected_keywords
        )

        return matched / len(expected_keywords)