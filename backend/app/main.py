from fastapi import FastAPI
from backend.app.core.init_db import init_db
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.admin import router as admin_router
from backend.app.api.routes.chat import router as chat_router
from backend.app.core.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from backend.app.middleware.request_logging import (
    request_logging_middleware,
)
setup_logging()
app = FastAPI(
    title="Enterprise Support AI",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(
    request_logging_middleware
)

@app.on_event("startup")
def startup_event():
    init_db()

app.include_router(documents_router)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(chat_router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-support-ai",
    }