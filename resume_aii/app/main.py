from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.analyze import router as analyze_router

app = FastAPI(title="Resume AII API", version="0.1.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
def root():
    return {"message": "Resume AII API is running", "docs": "/docs"}


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


# Routers
app.include_router(analyze_router)


# For `uvicorn resume_aii.app.main:app --reload`
__all__ = ["app"]
