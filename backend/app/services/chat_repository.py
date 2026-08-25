from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.conversation import Conversation
from backend.app.models.message import Message


class ChatRepository:

    def create_conversation(
        self,
        db: Session,
        user_id: int,
        title: str = "New Conversation",
    ) -> Conversation:

        conversation = Conversation(
            user_id=user_id,
            title=title,
        )

        db.add(conversation)
        db.flush()

        return conversation

    def get_user_conversation(
        self,
        db: Session,
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:

        statement = (
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )

        return db.execute(statement).scalar_one_or_none()

    def get_user_conversations(
        self,
        db: Session,
        user_id: int,
    ) -> list[Conversation]:

        statement = (
            select(Conversation)
            .where(
                Conversation.user_id == user_id
            )
            .order_by(
                Conversation.updated_at.desc()
            )
        )

        return list(
            db.execute(statement)
            .scalars()
            .all()
        )

    def create_message(
        self,
        db: Session,
        conversation_id: int,
        role: str,
        content: str,
    ) -> Message:

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        db.add(message)
        db.flush()

        return message

    def get_messages(
        self,
        db: Session,
        conversation_id: int,
    ) -> list[Message]:

        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(Message.created_at.asc())
        )

        return list(
            db.execute(statement)
            .scalars()
            .all()
        )