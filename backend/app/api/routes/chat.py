from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_current_user
from backend.app.core.database import get_db
from backend.app.models.user import User
from backend.app.services.chat_repository import ChatRepository
from backend.app.services.rag_service import RAGService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


chat_repository = ChatRepository()
rag_service = RAGService()

class ChatRequest(BaseModel):
    conversation_id: int | None = None
    question: str
    top_k: int = 3


@router.post("")
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty",
            )

        if request.conversation_id is None:
            conversation = (
                chat_repository.create_conversation(
                    db=db,
                    user_id=current_user.id,
                    title=request.question[:100],
                )
            )
        else:

            conversation = (
                chat_repository.get_user_conversation(
                    db=db,
                    conversation_id=request.conversation_id,
                    user_id=current_user.id,
                )
            )

            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found",
                )

        chat_repository.create_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        content=request.question,
    )
        rag_result = rag_service.answer(
        db=db,
        question=request.question,
        user_id=current_user.id,
        top_k=request.top_k,
    )

        chat_repository.create_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        content=rag_result["answer"],
    )
        db.commit()

        return {
        "conversation_id": conversation.id,
        "question": request.question,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
    }

@router.get("")
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    conversations = (
        chat_repository.get_user_conversations(
            db=db,
            user_id=current_user.id,
        )
    )

    return {
        "conversations": [
            {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
            }
            for conversation in conversations
        ]
    }

@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    conversation = (
        chat_repository.get_user_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = chat_repository.get_messages(
        db=db,
        conversation_id=conversation.id,
    )

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
            }
            for message in messages
        ],
    }

@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = chat_repository.get_user_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    db.delete(conversation)
    db.commit()

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id,
    }