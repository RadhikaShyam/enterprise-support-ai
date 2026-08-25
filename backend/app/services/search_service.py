from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, DocumentChunk


class SearchService:

    def search(
        self,
        db: Session,
        query_embedding: list[float],
        user_id: int,
        limit: int = 5,
    ):

        distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(
                DocumentChunk,
                distance.label("distance"),
            )
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .where(
                Document.user_id == user_id,
                DocumentChunk.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(limit)
        )

        return db.execute(statement).all()