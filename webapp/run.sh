#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r webapp/requirements.txt

echo ""
echo "Starting LeafCutter2 Web App at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 --reload
