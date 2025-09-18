from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    filename: str
    content_length: int
    preview: str
    embedding_dimensions: int
    model_name: str
