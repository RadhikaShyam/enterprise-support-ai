from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, DocumentChunk


class DocumentRepository:

    def create_document(
        self,
        db: Session,
        user_id: int,
        filename: str,
        content_type: str,
    ) -> Document:

        document = Document(
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            status="processing",
        )

        db.add(document)
        db.flush()

        return document

    def create_chunks(
        self,
        db: Session,
        document: Document,
        chunks: list[str],
    ) -> list[DocumentChunk]:

        records = []

        for index, content in enumerate(chunks):

            record = DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=content,
            )

            db.add(record)
            records.append(record)

        return records

    def update_chunk_embeddings(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match"
            )

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    def mark_completed(
        self,
        document: Document,
    ) -> None:

        document.status = "completed"

    def mark_failed(
        self,
        document: Document,
    ) -> None:

        document.status = "failed"

    def get_user_documents(
        self,
        db: Session,
        user_id: int,
    ) -> list[Document]:

        statement = (
            select(Document)
            .where(
                Document.user_id == user_id
            )
            .order_by(
                Document.created_at.desc()
            )
        )

        return list(
            db.execute(statement)
            .scalars()
            .all()
        )

    def get_admin_documents(
            self,
            db: Session,
        ) -> list[Document]:
    
            statement = (
                select(Document)
                .order_by(
                    Document.created_at.desc()
                )
            )
    
            return list(
                db.execute(statement)
                .scalars()
                .all()
            )

    def get_user_document(
        self,
        db: Session,
        document_id: int,
        user_id: int,
    ) -> Document | None:

        statement = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )

        return (
            db.execute(statement)
            .scalar_one_or_none()
        )

    def get_admin_document(
            self,
            db: Session,
            document_id: int,
        ) -> Document | None:
    
            statement = (
                select(Document)
                .where(
                    Document.id == document_id,
                )
            )
    
            return (
                db.execute(statement)
                .scalar_one_or_none()
            )

    def delete_document(
        self,
        db: Session,
        document: Document,
    ) -> None:

        db.delete(document)