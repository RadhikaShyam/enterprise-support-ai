from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_current_user,require_admin
from backend.app.models.user import User

from backend.app.core.database import get_db

from backend.app.services.rag_service import RAGService
from backend.app.services.search_service import SearchService
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.document_repository import DocumentRepository
from backend.app.services.document_service import DocumentService
from backend.app.services.embedding_service import EmbeddingService


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


rag_service = RAGService()
search_service = SearchService()
document_service = DocumentService()
chunking_service = ChunkingService()
document_repository = DocumentRepository()
embedding_service = EmbeddingService()


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class RAGRequest(BaseModel):
    question: str
    top_k: int = 3


UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    # =========================================================
    # 1. Validate filename
    # =========================================================

    filename = file.filename or "unknown"

    safe_filename = Path(filename).name

    extension = Path(
        safe_filename
    ).suffix.lower()

    # =========================================================
    # 2. Validate file type
    # =========================================================

    if extension not in DocumentService.SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}",
        )

    # =========================================================
    # 3. Read file
    # =========================================================

    content = await file.read()

    # =========================================================
    # 4. Validate file size
    # =========================================================

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB",
        )

    # =========================================================
    # 5. Save file
    # =========================================================

    file_path = UPLOAD_DIR / safe_filename

    file_path.write_bytes(content)

    # =========================================================
    # 6. Extract text
    # =========================================================

    try:

        text = document_service.extract_text(
            str(file_path)
        )

        cleaned_text = document_service.clean_text(
            text
        )

        chunks = chunking_service.split(
            cleaned_text
        )

        # =====================================================
        # 7. Create document
        # =====================================================

        document = document_repository.create_document(
            db=db,
            user_id=current_user.id,
            filename=safe_filename,
            content_type=file.content_type
            or "text/plain",
        )

        # =====================================================
        # 8. Create chunks
        # =====================================================

        chunk_records = (
            document_repository.create_chunks(
                db=db,
                document=document,
                chunks=chunks,
            )
        )

        # =====================================================
        # 9. Generate embeddings
        # =====================================================

        embeddings = embedding_service.embed_many(
            chunks
        )

        # =====================================================
        # 10. Store embeddings
        # =====================================================

        document_repository.update_chunk_embeddings(
            chunk_records,
            embeddings,
        )

        # =====================================================
        # 11. Mark completed
        # =====================================================

        document_repository.mark_completed(
            document
        )

        db.commit()

    except Exception:

        db.rollback()

        raise

    # =========================================================
    # 12. Response
    # =========================================================

    return {
        "document_id": document.id,
        "filename": document.filename,
        "status": document.status,
        "chunk_count": len(chunks),
        "embedding_dimensions": (
            len(embeddings[0])
            if embeddings
            else 0
        ),
    }

@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents = document_repository.get_user_documents(
        db=db,
        user_id=current_user.id,
    )

    return {
        "documents": [
            {
                "id": document.id,
                "filename": document.filename,
                "content_type": document.content_type,
                "status": document.status,
                "created_at": document.created_at,
            }
            for document in documents
        ]
    }


@router.get("/{document_id}")
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = document_repository.get_user_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id,
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "status": document.status,
        "created_at": document.created_at,
    }


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = document_repository.get_user_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id,
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    document_repository.delete_document(
        db=db,
        document=document,
    )

    db.commit()

    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
    }
# ============================================================
# SEARCH
# ============================================================

@router.post("/search")
async def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    query_embedding = embedding_service.embed(
        request.query
    )

    results = search_service.search(
        db=db,
        query_embedding=query_embedding,
        user_id=current_user.id,
        limit=request.limit,
    )

    return {
        "query": request.query,
        "results": [
            {
                "document_id": row[0].document_id,
                "chunk_id": row[0].id,
                "chunk_index": row[0].chunk_index,
                "content": row[0].content,
                "distance": float(row[1]),
            }
            for row in results
        ],
    }


# ============================================================
# RAG DEBUG
# ============================================================

@router.post("/rag/debug")
async def rag_debug(
    request: RAGRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    results = rag_service.retrieve(
        db=db,
        question=request.question,
        user_id=current_user.id,
        top_k=request.top_k,
    )

    context = rag_service.build_context(
        results
    )

    return {
        "question": request.question,
        "retrieved_chunks": len(results),
        "context": context,
        "sources": [
            {
                "document_id": result["chunk"].document_id,
                "chunk_id": result["chunk"].id,
                "chunk_index": result["chunk"].chunk_index,
                "distance": result["distance"],
            }
            for result in results
        ],
    }


# ============================================================
# RAG ANSWER
# ============================================================

@router.post("/rag")
async def rag_answer(
    request: RAGRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    return rag_service.answer(
        db=db,
        question=request.question,
        user_id=current_user.id,
        top_k=request.top_k,
    )

@router.get("")
def get_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    documents = document_repository.get_user_documents(
        db=db,
        user_id=current_user.id,
    )

    return [
        {
            "id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "status": document.status,
            "created_at": document.created_at,
        }
        for document in documents
    ]

@router.get("/admin/test")
def admin_test(
    current_user: User = Depends(require_admin),
):
    return {
        "message": "You are an admin",
        "user_id": current_user.id,
    }