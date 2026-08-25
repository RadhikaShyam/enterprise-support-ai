from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.services.auth_service import AuthService


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


auth_service = AuthService()


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):

    try:

        user = auth_service.register(
            db=db,
            email=request.email,
            password=request.password,
        )

        return {
            "id": user.id,
            "email": user.email,
            "role": user.role,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.post("/login")
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):

    try:

        token = auth_service.authenticate(
            db=db,
            email=request.email,
            password=request.password,
        )

        return {
            "access_token": token,
            "token_type": "bearer",
        }

    except ValueError:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )