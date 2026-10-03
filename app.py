"""Entry point so the dashboard starts from the project root:

    streamlit run app.py

It simply runs app/streamlit_app.py, which holds the dashboard code.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"),
               run_name="__main__")
