import re


class GroundingService:

    FALLBACK = (
        "I don't have enough information in the available "
        "support documentation to answer that question."
    )

    NUMBER_WORDS = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
    }

    FORBIDDEN_PATTERNS = [
        "i think",
        "i believe",
        "it is likely",
        "likely",
        "typically",
        "generally",
        "usually",
        "according to my knowledge",
        "based on my knowledge",
    ]

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    def _normalize(self, text: str) -> str:

        text = text.lower()

        # Convert number words to digits.
        for word, number in self.NUMBER_WORDS.items():
            text = re.sub(
                rf"\b{word}\b",
                number,
                text,
            )

        # Normalize punctuation.
        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # =========================================================
    # TOKENIZE
    # =========================================================

    def _words(self, text: str) -> set[str]:

        normalized = self._normalize(text)

        return set(
            word
            for word in normalized.split()
            if len(word) >= 3
        )

    # =========================================================
    # VALIDATE LLM ANSWER
    # =========================================================

    def validate(
        self,
        answer: str,
        context: str,
    ) -> bool:

        if not answer or not answer.strip():
            return False

        if not context or not context.strip():
            return False

        answer = answer.strip()
        context = context.strip()

        answer_lower = answer.lower()

        # -----------------------------------------------------
        # Reject obvious hallucination language
        # -----------------------------------------------------

        for pattern in self.FORBIDDEN_PATTERNS:

            if pattern in answer_lower:
                return False

        # -----------------------------------------------------
        # Normalize numbers.
        #
        # "5" and "five" become "5"
        # "30" and "thirty" become "30"
        # -----------------------------------------------------

        normalized_answer = self._normalize(answer)
        normalized_context = self._normalize(context)

        # -----------------------------------------------------
        # Numeric facts in answer MUST exist in context.
        # -----------------------------------------------------

        answer_numbers = set(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                normalized_answer,
            )
        )

        context_numbers = set(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                normalized_context,
            )
        )

        if not answer_numbers.issubset(
            context_numbers
        ):
            return False

        # -----------------------------------------------------
        # Word overlap
        # -----------------------------------------------------

        answer_words = self._words(answer)

        context_words = self._words(context)

        if not answer_words:
            return False

        matched_words = (
            answer_words & context_words
        )

        overlap_ratio = (
            len(matched_words)
            / len(answer_words)
        )

        # -----------------------------------------------------
        # Reasonable grounding threshold.
        #
        # We don't require every word because the LLM may
        # paraphrase:
        #
        # "account will be locked"
        #
        # vs
        #
        # "account is temporarily locked"
        # -----------------------------------------------------

        if overlap_ratio < 0.50:
            return False

        return True

    # =========================================================
    # EXTRACT ANSWER DIRECTLY FROM CONTEXT
    # =========================================================

    def extract_relevant_answer(
        self,
        context: str,
        question: str,
    ) -> str | None:

        if not context or not question:
            return None

        # -----------------------------------------------------
        # Split context into sentences / lines.
        # -----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            context,
        )

        question_words = self._words(question)

        if not question_words:
            return None

        candidates = []

        # -----------------------------------------------------
        # Find sentences related to the question.
        # -----------------------------------------------------

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Ignore metadata.
            if sentence.startswith("SOURCE "):
                continue

            if sentence.startswith("Document ID:"):
                continue

            if sentence.startswith("Chunk ID:"):
                continue

            if sentence.startswith(
                "Similarity distance:"
            ):
                continue

            sentence_words = self._words(
                sentence
            )

            if not sentence_words:
                continue

            overlap = (
                question_words
                & sentence_words
            )

            if not overlap:
                continue

            score = (
                len(overlap)
                / len(question_words)
            )

            candidates.append(
                (
                    score,
                    len(overlap),
                    sentence,
                )
            )

        if not candidates:
            return None

        # Highest relevance first.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # Return the best sentence DIRECTLY.
        #
        # IMPORTANT:
        # This sentence came directly from the retrieved
        # context, therefore it is already grounded.
        #
        # DO NOT call validate() here.
        # -----------------------------------------------------

        best_sentence = candidates[0][2]

        return best_sentence