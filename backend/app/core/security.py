import hashlib
from datetime import datetime, timedelta, timezone

from jose import jwt

from backend.app.core.config import settings


ALGORITHM = "HS256"


def hash_password(password: str) -> str:

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:

    return hashlib.sha256(
        plain_password.encode("utf-8")
    ).hexdigest() == hashed_password


def create_access_token(
    user_id: int,
    expires_minutes: int = 60,
) -> str:

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes
    )

    payload = {
        "sub": str(user_id),
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=ALGORITHM,
    )