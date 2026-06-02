"""
run.py - Easy launcher for the Contract Risk Analyzer API.
Run from the project root:

    python run.py

Then open:
    http://localhost:8000         <- Dashboard
    http://localhost:8000/docs    <- Swagger UI
"""
import sys
import os

# Ensure the project root is on the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fix Windows console encoding so emoji/special chars don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import uvicorn

if __name__ == "__main__":
    print("=" * 55)
    print("  Contract Risk Analyzer")
    print("=" * 55)
    print("  Dashboard:   http://localhost:8000")
    print("  Swagger UI:  http://localhost:8000/docs")
    print("  Health:      http://localhost:8000/health")
    print("=" * 55)
    print("  Press CTRL+C to stop.\n")

    uvicorn.run(
        "src.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
