# 🏛️ Contract Intelligence Engine

**Project 1: AI-Powered Contract Intelligence & Risk Scoring (NLP)**  
*Production-Level Data Science & Machine Learning Project — Zaalima Development*

---

## 📌 What This Project Does

This system ingests legal contracts (PDF or raw text) and automatically:

| Feature | Description |
|---------|-------------|
| 📄 **OCR** | Converts scanned PDF contracts to searchable text via Tesseract |
| 🔍 **NER** | Extracts parties, dates, and monetary values using spaCy |
| ⚠️ **Risk Scoring** | Detects 16 CUAD-aligned risky clause types with evidence |
| 📊 **Structure Analysis** | Checks contract completeness and readability complexity |
| ⏰ **Deadline Urgency** | Flags auto-renewal windows and tight notice periods |
| 🔎 **Semantic Search** | Finds similar contracts via vector embeddings |
| 🌐 **REST API** | Full FastAPI backend with Swagger docs |

---

## 🏗️ Project Architecture

```
PDF Contract
    │
    ▼
[OCR — Tesseract]                src/ocr/pipeline.py
    │
    ▼
[Text Cleaning]                  src/data_engineering/preprocess.py
    │
    ├──▶ [NER Extraction]        src/nlp/ner_model.py
    │         (spaCy)            → Organizations, Dates, Money
    │
    ├──▶ [Risk Scoring]          src/nlp/risk_scorer.py
    │         (Heuristics)       → Risk Score, Grade, Clause Evidence
    │
    ├──▶ [Structure Analysis]    src/feature_engineering/sequence_features.py
    │                            → Sections, Completeness, Complexity
    │
    ├──▶ [Date Analysis]         src/feature_engineering/recency_features.py
    │                            → Deadlines, Notice Periods, Age
    │
    └──▶ [Embeddings + VectorDB] src/feature_engineering/embeddings.py
                                 src/backend/vector_store.py
                                 → Semantic Search
    │
    ▼
[FastAPI REST API]               src/backend/app.py
    │
    ▼
[HTML Dashboard + Swagger UI]    http://localhost:8000
```

---

## 🚀 How to Run (Local — No Docker Needed)

### Step 1: Install Python Dependencies

```bash
cd c:\Projects1\AI-control-and-risk-main

pip install fastapi uvicorn[standard] python-multipart
pip install torch transformers sentence-transformers
pip install spacy pandas numpy scikit-learn
pip install pytesseract pdf2image Pillow
```

### Step 2: Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### Step 3: (For PDF support only) Install Tesseract OCR

> **Skip this step if you only want to analyze text** — you can paste contract text directly.

- **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  - Install to `C:\Program Files\Tesseract-OCR\`
  - Add to PATH: `C:\Program Files\Tesseract-OCR\`
- **Linux**: `sudo apt-get install tesseract-ocr poppler-utils`
- **Mac**: `brew install tesseract poppler`

### Step 4: Start the Server

```bash
# From the project root directory:
python run.py
```

You will see:
```
🏛️  Zaalima Contract Intelligence Engine
📡  Starting API server...
🌐  Dashboard:  http://localhost:8000
📖  Swagger UI: http://localhost:8000/docs
```

### Step 5: Use the System

Open your browser and go to **http://localhost:8000**

---

## 🎯 How to Use This Model

### Option A: Web Dashboard (Easiest)

1. Open **http://localhost:8000**
2. Click **"Load Sample Contract"** to pre-fill a test contract
3. Click **"Analyze Contract"**
4. See the full risk report instantly!

### Option B: Swagger UI (API Testing)

1. Open **http://localhost:8000/docs**
2. Click on **POST /analyze-text/**
3. Click "Try it out"
4. Paste this sample body:
```json
{
  "text": "This agreement shall automatically renew for successive one-year periods unless either party provides 90 days written notice. IN NO EVENT SHALL LICENSOR BE LIABLE FOR INDIRECT DAMAGES. During the term and two years thereafter, Licensee shall not compete with Licensor.",
  "title": "My Test Contract"
}
```
5. Click "Execute" → see the full risk analysis!

### Option C: Python Code

```python
import requests

# Analyze raw text
response = requests.post("http://localhost:8000/analyze-text/", json={
    "text": "This agreement automatically renews unless 90 days notice is given...",
    "title": "My Contract"
})
result = response.json()

print(f"Risk Score: {result['summary']['risk_score']}/100")
print(f"Risk Grade: {result['summary']['risk_grade']} — {result['summary']['risk_label']}")
print(f"High Risk Clauses: {result['summary']['high_risk_clauses']}")
```

### Option D: Upload a PDF

```python
import requests

with open("my_contract.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-contract/",
        files={"file": ("contract.pdf", f, "application/pdf")}
    )
result = response.json()
print(result["risk_analysis"]["risk_label"])
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Interactive HTML dashboard |
| `GET`  | `/health` | System status |
| `POST` | `/analyze-text/` | Analyze raw contract text |
| `POST` | `/upload-contract/` | Upload PDF → full pipeline |
| `GET`  | `/contracts` | List all analyzed contracts |
| `GET`  | `/contracts/{id}` | Get full analysis by ID |
| `POST` | `/search/` | Semantic search across contracts |
| `GET`  | `/docs` | Swagger interactive API docs |

---

## 🧠 Understanding the Risk Score

| Score | Grade | Label |
|-------|-------|-------|
| 0–17  | **A** | 🟢 LOW RISK |
| 18–31 | **B** | 🟡 LOW-MEDIUM RISK |
| 32–47 | **C** | 🟠 MEDIUM RISK |
| 48–64 | **D** | 🔴 HIGH RISK |
| 65+   | **F** | 🔴 VERY HIGH RISK |

The system detects **16 clause types** including:
- 🚨 **HIGH**: Auto-Renewal, Non-Compete, Liability Cap, IP Ownership, Liquidated Damages
- ⚠️ **MEDIUM**: Confidentiality, Arbitration, Payment Terms, Exclusivity
- ℹ️ **LOW**: Force Majeure, Audit Rights

---

## 🔬 Training the Transformer Model (Week 2)

To train the RoBERTa-based Legal Clause Classifier on real CUAD data:

```bash
# 1. Install datasets library
pip install datasets

# 2. Run training (downloads CUAD + fine-tunes RoBERTa)
python -m src.nlp.train_transformer
```

The trained model is saved to `models/legal_classifier.pth`.

To use real CUAD data:
```python
from src.data_engineering.cuad_loader import load_cuad_from_huggingface
samples = load_cuad_from_huggingface()
```

---

## 🐳 Docker Deployment (Production Only)

> Docker is **NOT required** to run locally. Use it only for deploying to a server.

```bash
# Build and start all services
docker-compose up --build

# Stop
docker-compose down
```

The API will be available at `http://your-server-ip:8000`.

---

## 📁 Project Structure

```
AI-control-and-risk-main/
├── run.py                          ← 🚀 START HERE — easy launcher
├── requirements.txt
├── Dockerfile                      ← Production deployment
├── docker-compose.yml
├── data/
│   ├── raw/                        ← Uploaded PDFs
│   ├── processed/                  ← Analysis JSON results
│   └── vector_store/               ← Contract embedding index
├── models/
│   └── recommendation_model.pth
└── src/
    ├── ocr/
    │   └── pipeline.py             ← Tesseract OCR
    ├── nlp/
    │   ├── ner_model.py            ← spaCy entity extraction
    │   ├── risk_scorer.py          ← Clause risk detection
    │   ├── classifier.py           ← RoBERTa classifier model
    │   └── train_transformer.py    ← Fine-tuning training loop
    ├── data_engineering/
    │   ├── preprocess.py           ← Text cleaning
    │   └── cuad_loader.py          ← CUAD dataset loader
    ├── feature_engineering/
    │   ├── embeddings.py           ← Document embeddings
    │   ├── recency_features.py     ← Date/deadline analysis
    │   └── sequence_features.py    ← Structural analysis
    ├── backend/
    │   ├── app.py                  ← FastAPI application
    │   ├── vector_store.py         ← Local vector database
    │   └── redis_store.py
    └── airflow/
        └── retraining_dag.py       ← Weekly model retraining DAG
```

---

## ❓ Troubleshooting

**Q: I get `ModuleNotFoundError: No module named 'src'`**  
A: Run from the project root using `python run.py` (not `uvicorn` directly).

**Q: spaCy model not found**  
A: Run `python -m spacy download en_core_web_sm`

**Q: PDF upload fails**  
A: Install Tesseract OCR (see Step 3 above). For text-only analysis, use `/analyze-text/` instead.

**Q: First request is slow**  
A: The sentence-transformers model (~90MB) downloads on first use. Subsequent requests are fast.

**Q: Do I need a GPU?**  
A: No. The risk scorer works entirely on CPU. The transformer classifier also runs on CPU (slower but functional).

---

## 📊 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| NLP | HuggingFace Transformers (RoBERTa), spaCy |
| OCR | Tesseract + pdf2image |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | Local NumPy store (Pinecone/Milvus for production) |
| API | FastAPI + Uvicorn |
| ML Framework | PyTorch |
| Dataset | CUAD (Contract Understanding Atticus Dataset) |
| Deployment | Docker + AWS EC2 |
| Orchestration | Apache Airflow (optional) |

---

*Zaalima Development — DSML Production Project Plan — Project 1*
