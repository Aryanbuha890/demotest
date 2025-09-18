from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import os


_model = None
_default_model_name = "sentence-transformers/all-MiniLM-L6-v2"
_model_name = os.getenv("RESUME_AII_MODEL", _default_model_name)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_model_name)
    return _model


def get_text_embedding(text: str) -> List[float]:
    model = _get_model()
    emb = model.encode([text], normalize_embeddings=True)[0]
    return emb.tolist()


def get_model_info() -> Dict[str, str]:
    return {"name": _model_name}


def reload_model(model_path_or_name: str) -> None:
    """Load a new model (local fine-tuned path or hub name) at runtime."""
    global _model, _model_name
    _model_name = model_path_or_name
    _model = SentenceTransformer(_model_name)

