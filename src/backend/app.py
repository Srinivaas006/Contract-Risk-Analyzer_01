"""
Contract Risk Analyzer — FastAPI Backend
Run with:  python run.py
Dashboard: http://localhost:8000
API Docs:  http://localhost:8000/docs
"""
import os
import json
import uuid
import shutil
import threading
import traceback
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request

# ─────────────────────────────────────────────────────────────────
# SAFE IMPORTS — every module wrapped individually.
# If one fails the rest of the app still works.
# ─────────────────────────────────────────────────────────────────

try:
    from src.ocr.pipeline import process_contract_pdf
except Exception as _e:
    print(f"⚠️  OCR not available: {_e}")
    def process_contract_pdf(path):
        return f"Error: OCR unavailable. Install pytesseract + Tesseract."

try:
    from src.nlp.ner_model import extract_contract_entities
except Exception as _e:
    print(f"⚠️  NER not available: {_e}")
    def extract_contract_entities(text):
        return {"ORGANIZATIONS": [], "DATES": [], "MONETARY_VALUES": []}

try:
    from src.nlp.risk_scorer import score_contract_risk
except Exception as _e:
    print(f"⚠️  Risk scorer not available: {_e}")
    def score_contract_risk(text):
        return {
            "risk_score": 0, "risk_grade": "?", "risk_label": "Scorer unavailable",
            "risk_color": "#94a3b8", "high_risk_clauses": 0,
            "medium_risk_clauses": 0, "low_risk_clauses": 0,
            "detected_clauses": [], "recommendations": ["Install dependencies: pip install -r requirements.txt"]
        }

try:
    from src.feature_engineering.recency_features import (
        extract_dates_from_text, compute_deadline_urgency
    )
except Exception as _e:
    print(f"⚠️  Recency features not available: {_e}")
    def extract_dates_from_text(text): return []
    def compute_deadline_urgency(text): return {}

try:
    from src.feature_engineering.sequence_features import (
        extract_contract_sections, analyze_clause_sequence, compute_contract_complexity
    )
except Exception as _e:
    print(f"⚠️  Sequence features not available: {_e}")
    def extract_contract_sections(text): return []
    def analyze_clause_sequence(sections): return {"completeness_grade": "unknown", "completeness_score": 0}
    def compute_contract_complexity(text):
        words = len(text.split())
        return {"word_count": words, "sentence_count": 0, "complexity_level": "unknown",
                "avg_sentence_length_words": 0, "legal_jargon_count": 0}

try:
    from src.backend.vector_store import ContractVectorStore
    _vector_store = ContractVectorStore()
except Exception as _e:
    print(f"⚠️  Vector store not available: {_e}")
    class _FakeStore:
        def __len__(self): return 0
        def add_contract(self, **kw): pass
        def get_all(self): return []
        def search(self, emb, top_k=5): return []
    _vector_store = _FakeStore()

try:
    from src.feature_engineering.embeddings import generate_embedding
    _EMBEDDING_AVAILABLE = True
except Exception as _e:
    print(f"⚠️  Embeddings not available: {_e}")
    _EMBEDDING_AVAILABLE = False
    def generate_embedding(text): raise RuntimeError("sentence-transformers not installed")

# ─────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Contract Risk Analyzer",
    description="AI-Powered Contract Intelligence and Risk Scoring System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Always return JSON for unhandled errors — never plain text "Internal Server Error"
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "hint": "Check the server terminal for the full traceback."
        }
    )

# Serve frontend static files
frontend_dir = os.path.join(os.getcwd(), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
else:
    print(f"⚠️  Frontend folder not found at: {frontend_dir}")

# In-memory cache
analysis_cache: dict = {}


# ─────────────────────────────────────────────────────────────────
# ROUTES — FRONTEND
# ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Contract Risk Analyzer</h2><p><a href='/docs'>API Docs</a></p>")


# ─────────────────────────────────────────────────────────────────
# ROUTES — SYSTEM
# ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "online",
        "version": "1.0.0",
        "contracts_indexed": len(_vector_store),
        "modules": {
            "risk_scorer":  "✅ ready",
            "ner":          "✅ ready" if callable(extract_contract_entities) else "❌",
            "embeddings":   "✅ ready" if _EMBEDDING_AVAILABLE else "⚠️  not loaded yet",
            "vector_store": "✅ ready",
        },
        "timestamp": datetime.now().isoformat()
    }


# ─────────────────────────────────────────────────────────────────
# BACKGROUND EMBEDDING (never blocks the HTTP response)
# ─────────────────────────────────────────────────────────────────

def _embed_in_background(contract_id: str, text: str, metadata: dict):
    try:
        emb = generate_embedding(text)
        _vector_store.add_contract(contract_id=contract_id, embedding=emb, metadata=metadata)
        print(f"✅ Embedding stored for {contract_id}")
    except Exception as e:
        print(f"⚠️  Embedding skipped for {contract_id}: {e}")


# ─────────────────────────────────────────────────────────────────
# CORE ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_analysis(contract_id: str, filename: str, text: str) -> dict:
    """
    Runs NER → Risk Scoring → Structure → Dates → returns result instantly.
    Embedding is generated in the background (won't delay response).
    """
    ts = datetime.now().isoformat()

    # Step 1 — Named Entity Recognition
    try:
        entities = extract_contract_entities(text)
    except Exception as e:
        entities = {"ORGANIZATIONS": [], "DATES": [], "MONETARY_VALUES": [], "error": str(e)}

    # Step 2 — Risk Clause Detection (pure Python, always fast)
    try:
        risk = score_contract_risk(text)
    except Exception as e:
        risk = {
            "risk_score": 0, "risk_grade": "?", "risk_label": "Error during scoring",
            "risk_color": "#94a3b8", "high_risk_clauses": 0, "medium_risk_clauses": 0,
            "low_risk_clauses": 0, "detected_clauses": [], "recommendations": [str(e)]
        }

    # Step 3 — Structure & Complexity
    try:
        sections = extract_contract_sections(text)
        structure = analyze_clause_sequence(sections)
        complexity = compute_contract_complexity(text)
    except Exception as e:
        structure = {"completeness_grade": "unknown", "completeness_score": 0, "error": str(e)}
        complexity = {"word_count": len(text.split()), "sentence_count": 0,
                      "complexity_level": "unknown", "avg_sentence_length_words": 0,
                      "legal_jargon_count": 0}

    # Step 4 — Dates & Deadlines
    try:
        dates = extract_dates_from_text(text)
        urgency = compute_deadline_urgency(text)
    except Exception:
        dates, urgency = [], {}

    # Step 5 — Embedding in background thread (non-blocking)
    if _EMBEDDING_AVAILABLE:
        threading.Thread(
            target=_embed_in_background,
            args=(contract_id, text[:3000], {
                "filename": filename, "analyzed_at": ts,
                "risk_score": risk.get("risk_score", 0),
                "risk_grade": risk.get("risk_grade", "?"),
                "word_count": complexity.get("word_count", 0),
            }),
            daemon=True
        ).start()

    result = {
        "contract_id":      contract_id,
        "filename":         filename,
        "analyzed_at":      ts,
        "status":           "complete",
        "text_preview":     text[:500] + "..." if len(text) > 500 else text,
        "word_count":       complexity.get("word_count", 0),
        "entities":         entities,
        "risk_analysis":    risk,
        "structure_analysis": structure,
        "complexity_metrics": complexity,
        "dates_found":      dates,
        "deadline_urgency": urgency,
        "embedding_indexed": "background" if _EMBEDDING_AVAILABLE else False,
        "summary": {
            "risk_score":        risk.get("risk_score", 0),
            "risk_grade":        risk.get("risk_grade", "?"),
            "risk_label":        risk.get("risk_label", ""),
            "risk_color":        risk.get("risk_color", "#94a3b8"),
            "high_risk_clauses": risk.get("high_risk_clauses", 0),
            "parties_found":     entities.get("ORGANIZATIONS", []),
            "completeness":      structure.get("completeness_grade", "unknown"),
        }
    }

    # Cache + save
    analysis_cache[contract_id] = result
    os.makedirs("data/processed", exist_ok=True)
    try:
        with open(f"data/processed/{contract_id}_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────
# ROUTES — ANALYSIS
# ─────────────────────────────────────────────────────────────────

@app.post("/analyze-text/", tags=["Analysis"])
async def analyze_text(body: dict):
    """
    Analyze raw contract text.

    Body: `{ "text": "This agreement...", "title": "My Contract" }`
    """
    text = body.get("text", "").strip()
    title = body.get("title", "Untitled Contract")

    if not text:
        raise HTTPException(status_code=400, detail="The 'text' field is required.")
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Contract text must be at least 50 characters.")

    contract_id = str(uuid.uuid4())[:8]
    return run_analysis(contract_id, title, text)


@app.post("/upload-contract/", tags=["Analysis"])
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a PDF contract. Requires Tesseract OCR.
    For plain text input use /analyze-text/ instead.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted. Use /analyze-text/ for plain text.")

    contract_id = str(uuid.uuid4())[:8]
    os.makedirs("data/raw", exist_ok=True)
    save_path = f"data/raw/{contract_id}_{file.filename}"

    with open(save_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    extracted = process_contract_pdf(save_path)

    if extracted.startswith("Error"):
        raise HTTPException(
            status_code=500,
            detail=f"{extracted}. Install Tesseract OCR or use /analyze-text/ for raw text."
        )

    return run_analysis(contract_id, file.filename, extracted)


# ─────────────────────────────────────────────────────────────────
# ROUTES — CONTRACTS & SEARCH
# ─────────────────────────────────────────────────────────────────

@app.get("/contracts", tags=["Contracts"])
def list_contracts():
    """List all analyzed and indexed contracts."""
    return {"total": len(_vector_store), "contracts": _vector_store.get_all()}


@app.get("/contracts/{contract_id}", tags=["Contracts"])
def get_contract(contract_id: str):
    """Get the full analysis for a contract by ID."""
    if contract_id in analysis_cache:
        return analysis_cache[contract_id]
    path = f"data/processed/{contract_id}_analysis.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found.")


@app.post("/search/", tags=["Search"])
async def search_contracts(body: dict):
    """Semantic search across all indexed contracts."""
    query = body.get("query", "").strip()
    top_k = min(int(body.get("top_k", 5)), 20)

    if not query:
        raise HTTPException(status_code=400, detail="'query' field is required.")
    if len(_vector_store) == 0:
        return {"query": query, "results": [], "message": "No contracts indexed yet."}

    try:
        emb = generate_embedding(query)
        results = _vector_store.search(emb, top_k=top_k)
        return {"query": query, "total_results": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))