#!/usr/bin/env bash
# macOS / Linux: install dependencies and launch the dashboard.
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
