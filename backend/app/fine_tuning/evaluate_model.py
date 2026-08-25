import json
from pathlib import Path

from backend.app.fine_tuning.inference_service import FineTunedLLM


EVALUATION_FILE = Path(
    "data/fine_tuning/evaluation.jsonl"
)


def load_questions():
    questions = []

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if line.strip():
                questions.append(
                    json.loads(line)
                )

    return questions


def main():

    model = FineTunedLLM()

    questions = load_questions()

    print("\n")
    print("=" * 80)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 80)

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]
        expected = item["expected"]

        answer = model.generate(
            question
        )

        print("\n" + "-" * 80)

        print(
            f"TEST {index}"
        )

        print(
            f"\nQUESTION:\n{question}"
        )

        print(
            f"\nEXPECTED:\n{expected}"
        )

        print(
            f"\nMODEL:\n{answer}"
        )


if __name__ == "__main__":
    main()