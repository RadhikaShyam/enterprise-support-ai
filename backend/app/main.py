from fastapi import FastAPI

app = FastAPI(
    title="Enterprise Support AI",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-support-ai",
    }