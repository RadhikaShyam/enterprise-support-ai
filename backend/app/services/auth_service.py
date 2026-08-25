from sqlalchemy.orm import Session

from backend.app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from backend.app.models.user import User


class AuthService:

    def register(
        self,
        db: Session,
        email: str,
        password: str,
    ) -> User:

        existing_user = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        if existing_user:
            raise ValueError(
                "User with this email already exists"
            )

        user = User(
            email=email,
            password_hash=hash_password(password),
            role="user",
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    def authenticate(
        self,
        db: Session,
        email: str,
        password: str,
    ) -> str:

        user = (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

        if not user:
            raise ValueError(
                "Invalid email or password"
            )

        if not verify_password(
            password,
            user.password_hash,
        ):
            raise ValueError(
                "Invalid email or password"
            )

        return create_access_token(
            user_id=user.id
        )