import logging
import re
import time

from sqlalchemy.orm import Session

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.search_service import SearchService
from backend.app.fine_tuning.inference_service import FineTunedLLM
from backend.app.services.grounding_service import GroundingService


logger = logging.getLogger("rag")


FALLBACK_MESSAGE = (
    "I don't have enough information in the available "
    "support documentation to answer that question."
)


class RAGService:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService()
        self.llm = FineTunedLLM()
        self.grounding_service = GroundingService()

    # =========================================================
    # GET DOCUMENT CHUNK FROM SQLALCHEMY RESULT
    # =========================================================

    def _get_chunk_from_row(self, row):

        # SQLAlchemy Row:
        #
        # (
        #     DocumentChunk,
        #     distance
        # )

        if hasattr(row, "_mapping"):

            chunk = row._mapping.get("DocumentChunk")

            if chunk is not None:
                return chunk

            # Fallback for SQLAlchemy versions / row layouts
            try:
                return row[0]
            except Exception:
                return None

        if isinstance(row, (tuple, list)):

            if not row:
                return None

            return row[0]

        return row

    # =========================================================
    # GET DISTANCE FROM SQLALCHEMY RESULT
    # =========================================================

    def _get_distance_from_row(self, row):

        if hasattr(row, "_mapping"):

            distance = row._mapping.get("distance")

            if distance is not None:
                return distance

        if isinstance(row, (tuple, list)):

            if len(row) > 1:
                return row[1]

        return None

    # =========================================================
    # BUILD CONTEXT
    # =========================================================

    def _build_context(self, results):

        context_parts = []

        for row in results:

            chunk = self._get_chunk_from_row(row)

            if chunk is None:
                continue

            content = getattr(
                chunk,
                "content",
                None,
            )

            if not content:
                continue

            content = str(content).strip()

            if not content:
                continue

            context_parts.append(content)

        return "\n\n".join(context_parts)

    # =========================================================
    # BUILD SOURCES
    # =========================================================

    def _build_sources(self, results):

        sources = []

        for row in results:

            chunk = self._get_chunk_from_row(row)

            if chunk is None:
                continue

            distance = self._get_distance_from_row(row)

            document_id = getattr(
                chunk,
                "document_id",
                None,
            )

            chunk_id = getattr(
                chunk,
                "id",
                None,
            )

            chunk_index = getattr(
                chunk,
                "chunk_index",
                None,
            )

            filename = None

            document = getattr(
                chunk,
                "document",
                None,
            )

            if document is not None:

                filename = getattr(
                    document,
                    "filename",
                    None,
                )

            sources.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "filename": filename,
                    "distance": (
                        float(distance)
                        if distance is not None
                        else None
                    ),
                }
            )

        return sources

    # =========================================================
    # CLEAN ANSWER
    # =========================================================

    @staticmethod
    def _clean_answer(answer):

        if not answer:
            return ""

        answer = str(answer).strip()

        # Remove accidental markdown/code fences
        answer = re.sub(
            r"^```(?:text|json)?\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        answer = re.sub(
            r"\s*```$",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        # Remove "ANSWER:" prefix if Qwen generates it
        answer = re.sub(
            r"^\s*answer\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        return answer.strip()

    # =========================================================
    # CHECK WHETHER QUESTION HAS ANY REAL SUPPORT IN CONTEXT
    # =========================================================

    def _question_has_context_support(
        self,
        question: str,
        context: str,
    ):

        if not question or not context:
            return False

        question_words = set(
            re.findall(
                r"\b[a-z]{4,}\b",
                question.lower(),
            )
        )

        context_words = set(
            re.findall(
                r"\b[a-z]{4,}\b",
                context.lower(),
            )
        )

        if not question_words:
            return True

        overlap = question_words & context_words

        # We intentionally keep this permissive.
        #
        # Example:
        #
        # Question:
        # "How long does the account remain locked?"
        #
        # Context:
        # "The account automatically unlocks after 30 minutes."
        #
        # "locked" may not literally appear in the context,
        # therefore requiring a high overlap would incorrectly
        # reject valid RAG questions.

        if len(overlap) >= 1:
            return True

        return False

    # =========================================================
    # ANSWER
    # =========================================================

    def answer(
        self,
        db: Session,
        user_id: int,
        question: str,
        top_k: int = 3,
    ):

        total_start = time.perf_counter()

        question = (question or "").strip()

        logger.info(
            "rag_started user_id=%s question=%r top_k=%s",
            user_id,
            question,
            top_k,
        )

        # =====================================================
        # 0. INVALID QUESTION
        # =====================================================

        if not question:

            logger.warning(
                "rag_empty_question user_id=%s",
                user_id,
            )

            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
            }

        # =====================================================
        # 1. EMBEDDING
        # =====================================================

        embedding_start = time.perf_counter()

        try:

            query_embedding = self.embedding_service.embed(
                question
            )

        except Exception:

            logger.exception(
                "rag_embedding_failed user_id=%s",
                user_id,
            )

            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
            }

        embedding_ms = (
            time.perf_counter()
            - embedding_start
        ) * 1000

        logger.info(
            "rag_embedding user_id=%s embedding_ms=%.2f",
            user_id,
            embedding_ms,
        )

        # =====================================================
        # 2. VECTOR SEARCH
        #
        # IMPORTANT:
        # SearchService accepts:
        #
        # db
        # user_id
        # query_embedding
        # limit
        #
        # DO NOT PASS similarity_threshold.
        # =====================================================

        retrieval_start = time.perf_counter()

        try:

            results = self.search_service.search(
                db=db,
                user_id=user_id,
                query_embedding=query_embedding,
                limit=max(1, min(top_k, 5)),
            )

        except Exception:

            logger.exception(
                "rag_retrieval_failed user_id=%s",
                user_id,
            )

            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
            }

        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        logger.info(
            "rag_retrieval user_id=%s retrieved=%s "
            "retrieval_ms=%.2f",
            user_id,
            len(results),
            retrieval_ms,
        )

        # =====================================================
        # 2A. LOG EVERY RETRIEVED RESULT
        #
        # This is extremely important for debugging pgvector.
        # =====================================================

        for index, row in enumerate(results):

            chunk = self._get_chunk_from_row(row)
            distance = self._get_distance_from_row(row)

            logger.info(
                "rag_result user_id=%s rank=%s "
                "document_id=%s chunk_id=%s "
                "distance=%s content=%r",
                user_id,
                index + 1,
                getattr(
                    chunk,
                    "document_id",
                    None,
                ),
                getattr(
                    chunk,
                    "id",
                    None,
                ),
                distance,
                getattr(
                    chunk,
                    "content",
                    None,
                ),
            )

        # =====================================================
        # 3. NO RESULTS
        # =====================================================

        if not results:

            logger.warning(
                "rag_no_results user_id=%s question=%r",
                user_id,
                question,
            )

            return {
                "answer": FALLBACK_MESSAGE,
                "sources": [],
            }

        # =====================================================
        # 4. BUILD CONTEXT
        # =====================================================

        context_start = time.perf_counter()

        context = self._build_context(results)

        context_ms = (
            time.perf_counter()
            - context_start
        ) * 1000

        logger.info(
            "rag_context_built user_id=%s "
            "chunks=%s context_chars=%s "
            "context_ms=%.2f",
            user_id,
            len(results),
            len(context),
            context_ms,
        )

        logger.info(
            "rag_context content=%r",
            context,
        )

        # =====================================================
        # 5. EMPTY CONTEXT
        # =====================================================

        if not context:

            logger.warning(
                "rag_empty_context user_id=%s question=%r",
                user_id,
                question,
            )

            return {
                "answer": FALLBACK_MESSAGE,
                "sources": self._build_sources(results),
            }

        # =====================================================
        # 6. GENERATE LLM ANSWER
        # =====================================================

        llm_start = time.perf_counter()

        try:

            generated_answer = self.llm.generate(
                question=question,
                context=context,
            )

        except Exception:

            logger.exception(
                "rag_llm_generation_failed user_id=%s",
                user_id,
            )

            generated_answer = ""

        llm_ms = (
            time.perf_counter()
            - llm_start
        ) * 1000

        generated_answer = self._clean_answer(
            generated_answer
        )

        logger.info(
            "rag_llm_raw_answer user_id=%s answer=%r",
            user_id,
            generated_answer,
        )

        logger.info(
            "rag_llm_generation user_id=%s "
            "llm_ms=%.2f answer_chars=%s",
            user_id,
            llm_ms,
            len(generated_answer),
        )

        # =====================================================
        # 7. GROUNDING
        # =====================================================

        validation_start = time.perf_counter()

        grounded = False

        if generated_answer:

            try:

                grounded = self.grounding_service.validate(
                    answer=generated_answer,
                    context=context,
                )

            except Exception:

                logger.exception(
                    "rag_grounding_validation_failed "
                    "user_id=%s",
                    user_id,
                )

                grounded = False

        validation_ms = (
            time.perf_counter()
            - validation_start
        ) * 1000

        logger.info(
            "rag_grounding_validation "
            "user_id=%s grounded=%s "
            "validation_ms=%.2f",
            user_id,
            grounded,
            validation_ms,
        )

        # =====================================================
        # 8. FINAL ANSWER
        # =====================================================

        if grounded:

            answer = generated_answer

            logger.info(
                "rag_answer_accepted user_id=%s",
                user_id,
            )

        else:

            logger.warning(
                "rag_grounding_failed "
                "user_id=%s question=%r",
                user_id,
                question,
            )

            answer = FALLBACK_MESSAGE

            logger.info(
                "rag_fallback_used "
                "user_id=%s question=%r",
                user_id,
                question,
            )

        # =====================================================
        # 9. SOURCES
        # =====================================================

        sources = self._build_sources(
            results
        )

        # =====================================================
        # 10. COMPLETE
        # =====================================================

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        logger.info(
            "rag_completed "
            "user_id=%s sources=%s grounded=%s "
            "total_ms=%.2f",
            user_id,
            len(sources),
            grounded,
            total_ms,
        )

        return {
            "answer": answer,
            "sources": sources,
        }