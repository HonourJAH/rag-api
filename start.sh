#!/bin/bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
uvicorn app.main:app --host 0.0.0.0 --port 8000
