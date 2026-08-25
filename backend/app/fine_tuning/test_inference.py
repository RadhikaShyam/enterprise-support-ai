from backend.app.fine_tuning.inference_service import (
    FineTunedLLM,
)


def main():

    model = FineTunedLLM()

    questions = [
        "What happens after five failed login attempts?",
        "How do I reset my corporate password?",
        "What are the password requirements?",
        "My MFA is not working. What should I do?",
        "What is the company's vacation policy?",
    ]

    for question in questions:

        print("\n" + "=" * 70)

        print("QUESTION:")
        print(question)

        answer = model.generate(
            question
        )

        print("\nANSWER:")
        print(answer)


if __name__ == "__main__":
    main()