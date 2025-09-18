from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import Depends
from typing import Optional

from ..services.extract import extract_text_from_upload
from ..services.embeddings import get_text_embedding, get_model_info
from ..models import AnalyzeResponse

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/", response_model=AnalyzeResponse)
async def analyze_resume(file: UploadFile = File(...)):
    try:
        text = await extract_text_from_upload(file)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

        embedding = get_text_embedding(text)
        model_info = get_model_info()

        return AnalyzeResponse(
            filename=file.filename,
            content_length=len(text),
            preview=text[:500],
            embedding_dimensions=len(embedding),
            model_name=model_info["name"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
