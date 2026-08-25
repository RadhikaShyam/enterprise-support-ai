from backend.app.core.database import Base, engine
from backend.app.models.user import User
from backend.app.models.document import Document, DocumentChunk


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()