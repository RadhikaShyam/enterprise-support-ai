import json
import random
from pathlib import Path


SOURCE = Path(
    "data/fine_tuning/support_instructions.jsonl"
)

TRAIN = Path(
    "data/fine_tuning/train/train.jsonl"
)

VALIDATION = Path(
    "data/fine_tuning/validation/validation.jsonl"
)


def load_examples():
    examples = []

    with SOURCE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def main():

    examples = load_examples()

    random.seed(42)
    random.shuffle(examples)

    split_index = int(len(examples) * 0.8)

    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]

    TRAIN.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TRAIN.open(
        "w",
        encoding="utf-8",
    ) as file:

        for example in train_examples:
            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with VALIDATION.open(
        "w",
        encoding="utf-8",
    ) as file:

        for example in validation_examples:
            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Total examples: {len(examples)}"
    )

    print(
        f"Training examples: {len(train_examples)}"
    )

    print(
        f"Validation examples: {len(validation_examples)}"
    )


if __name__ == "__main__":
    main()