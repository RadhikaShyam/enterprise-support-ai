from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from backend.app.api.dependencies import require_admin
from backend.app.core.database import get_db
from backend.app.models.user import User
from backend.app.models.document import Document, DocumentChunk


router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)

@router.get("/users")
def get_all_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    users = (
        db.execute(
            select(User)
            .order_by(User.created_at.desc())
        )
        .scalars()
        .all()
    )

    return {
        "users": [
            {
                "id": user.id,
                "email": user.email,
                "role": user.role,
                "created_at": user.created_at,
            }
            for user in users
        ]
    }

@router.get("/statistics")
def get_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    total_users = db.scalar(
        select(func.count(User.id))
    )

    total_documents = db.scalar(
        select(func.count(Document.id))
    )

    total_chunks = db.scalar(
        select(func.count(DocumentChunk.id))
    )

    completed_documents = db.scalar(
        select(func.count(Document.id))
        .where(Document.status == "completed")
    )

    processing_documents = db.scalar(
        select(func.count(Document.id))
        .where(Document.status == "processing")
    )

    failed_documents = db.scalar(
        select(func.count(Document.id))
        .where(Document.status == "failed")
    )

    return {
        "total_users": total_users,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "completed_documents": completed_documents,
        "processing_documents": processing_documents,
        "failed_documents": failed_documents,
    }

@router.get("/documents")
def list_all_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    documents = (
            db.execute(
                select(Document)
                .order_by(Document.created_at.desc())
            )
            .scalars()
            .all()
        )
    
    return documents
