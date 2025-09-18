import io
from typing import Optional

import pdfplumber
from docx import Document
from fastapi import UploadFile, HTTPException


async def extract_text_from_upload(file: UploadFile) -> str:
    """
    Extract text from an uploaded file. Supports PDF, DOCX, and plain text.
    """
    filename = (file.filename or "").lower()
    content = await file.read()

    if filename.endswith(".pdf"):
        return _extract_pdf_text(content)
    if filename.endswith(".docx"):
        return _extract_docx_text(content)
    if filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    # Try best-effort fallback: attempt PDF first, then text decode
    try:
        return _extract_pdf_text(content)
    except Exception:
        return content.decode("utf-8", errors="ignore")


def _extract_pdf_text(raw: bytes) -> str:
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        texts = []
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)


def _extract_docx_text(raw: bytes) -> str:
    f = io.BytesIO(raw)
    doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs)
