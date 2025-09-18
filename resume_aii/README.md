# Resume AII API

A minimal FastAPI service that accepts a resume file (PDF/DOCX/TXT), extracts text, generates an embedding using `sentence-transformers`, and returns basic metadata.

## Project Structure

```
resume_aii/
  app/
    __init__.py
    main.py              # FastAPI app entry
    models.py            # Pydantic models
    routers/
      __init__.py
      analyze.py         # /analyze endpoint with file upload
    services/
      extract.py         # PDF/DOCX/TXT extraction
      embeddings.py      # SentenceTransformer embedding
  requirements.txt
  README.md
```

## Requirements
- Python 3.9+
- Windows PowerShell (for commands below) or your preferred shell

## Setup

1) (Recommended) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Note: The `sentence-transformers` model (`all-MiniLM-L6-v2`) will download on first run.

## Run

From the repo root (where `resume_aii/` is located):

```powershell
uvicorn resume_aii.app.main:app --reload --port 8000
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs
- Healthcheck: http://127.0.0.1:8000/health

## Usage

- Hit `POST /analyze` with `multipart/form-data` and a file field named `file`.
- Supported formats: `.pdf`, `.docx`, `.txt`.
- Response includes filename, content_length, a short preview, embedding dimension, and model name.

## Notes

- PDF extraction uses `pdfplumber`. Quality depends on the PDF (scanned PDFs may require OCR, which is not included).
- To support scanned PDFs/OCR, consider adding `pytesseract` and a Tesseract installation.

## Fine-tuning with your dataset

You can fine-tune the embedding model using `sentence-transformers` on your own dataset.

### 1) Prepare dataset

See `data/README.md` for the required schema. Supported formats:
- CSV with header: `text_a,text_b,score`
- JSONL: one JSON object per line
- JSON array of objects

Each row contains:
- `text_a`: e.g., resume text or a section
- `text_b`: e.g., job description or a section
- `score`: similarity in [0, 1] (1 = highly similar, 0 = dissimilar)

### 2) Run training

Example command (PowerShell) to train and save to `models/my-ft/`:

```powershell
python -m resume_aii.train.train_st --train_path path\to\train.csv --val_path path\to\val.csv --output_dir models\my-ft --epochs 2 --batch_size 16 --lr 2e-5 --warmup_ratio 0.1 --max_seq_length 256
```

Notes:
- The script uses `CosineSimilarityLoss` and evaluates with `EmbeddingSimilarityEvaluator` if `--val_path` is provided.
- On completion, it saves a time-stamped folder under `--output_dir` (e.g., `models/my-ft/final-YYYYMMDD-HHMMSS`). Use that path as the model.

### 3) Use the fine-tuned model in the API

Point the API to your fine-tuned model path using an environment variable:

```powershell
$env:RESUME_AII_MODEL = "models/my-ft/final-YYYYMMDD-HHMMSS"
uvicorn resume_aii.app.main:app --reload --port 8000
```

At runtime, the service reads `RESUME_AII_MODEL` in `app/services/embeddings.py`. If not set, it falls back to `sentence-transformers/all-MiniLM-L6-v2`.

Alternatively, you can update the model without restarting by calling `reload_model()` in code if you add an admin endpoint, or simply restart with the environment variable set.
