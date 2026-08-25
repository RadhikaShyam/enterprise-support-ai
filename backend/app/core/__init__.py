from backend.app.core.database import Base, engine
from backend.app.models.document import Document, DocumentChunk
from backend.app.models.user import User


def init_db() -> None:
    Base.metadata.create_all(bind=engine)