import json
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "golden_rag.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_evaluation_report.json"
)

API_URL = "http://127.0.0.1:8000/documents/rag"


def load_dataset():

    dataset = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                dataset.append(
                    json.loads(line)
                )

    return dataset


def call_rag(question: str):

    response = requests.post(
        API_URL,
        json={
            "question": question,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    results = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        question = item["question"]

        print()
        print(
            f"[{index}/{len(dataset)}] "
            f"{question}"
        )

        try:

            response = call_rag(
                question
            )

            answer = response.get(
                "answer",
                "",
            )

            sources = response.get(
                "sources",
                [],
            )

            expected = item.get(
                "expected_output"
            )

            should_answer = item.get(
                "should_answer",
                True,
            )

            # --------------------------------------------------
            # Basic evaluation
            # --------------------------------------------------

            answer_lower = answer.lower()

            fallback = (
                "i don't have enough information "
                "in the available support documentation "
                "to answer that question."
            )

            abstained = (
                fallback in answer_lower
            )

            if should_answer:

                passed = (
                    not abstained
                    and len(sources) > 0
                    and all(
                        expected_fact.lower() in answer_lower
                        for expected_fact in expected.split(";")
                    )
                )

            else:

                passed = abstained

            result = {
                "id": item["id"],
                "question": question,
                "expected": expected,
                "actual": answer,
                "should_answer": should_answer,
                "abstained": abstained,
                "source_count": len(sources),
                "passed": passed,
                "sources": sources,
            }

            results.append(result)

            print(
                "PASS"
                if passed
                else "FAIL"
            )

            print(
                f"Sources: {len(sources)}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "passed": False,
                    "error": str(exc),
                }
            )

    total = len(results)

    passed = sum(
        1
        for result in results
        if result.get("passed")
    )

    accuracy = (
        passed / total
        if total
        else 0.0
    )

    report = {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "accuracy": round(
            accuracy,
            4,
        ),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(
        f"Total:   {total}"
    )
    print(
        f"Passed:  {passed}"
    )
    print(
        f"Failed:  {total - passed}"
    )
    print(
        f"Accuracy: {accuracy:.2%}"
    )
    print()
    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()