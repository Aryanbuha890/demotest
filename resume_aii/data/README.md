# Dataset format for fine-tuning

The training script `train/train_st.py` expects a file in one of the following formats:
- CSV with header: `text_a,text_b,score`
- JSONL with one object per line: `{ "text_a": "...", "text_b": "...", "score": 0.0-1.0 }`
- JSON array with objects in the same schema

Where:
- `text_a`: source text (e.g., resume text or a section of a resume)
- `text_b`: target text (e.g., job description or part of it)
- `score`: similarity label in the range [0, 1]

Example (CSV):

```
text_a,text_b,score
"Experienced Python developer with FastAPI and NLP background","We are hiring a backend engineer with Python and FastAPI",0.92
"Retail manager focusing on store operations","Research scientist role in molecular biology",0.08
```

Typical usage:
- For resume↔job matching, create pairs of (resume_text, job_desc) with a similarity score.
- For positive/negative examples only, use 1 for positive and 0 for negative.

Place your datasets anywhere you prefer. The `train/train_st.py` script takes explicit `--train_path` and optional `--val_path` arguments.
