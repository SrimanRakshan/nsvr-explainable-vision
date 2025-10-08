from pathlib import Path
import sys
from streamlit.web import cli as stcli

ROOT = Path(__file__).resolve().parent
script = ROOT / "main_app.py"

# Hand off to Streamlit (separate entrypoint so main_app.py never self-invokes)
sys.argv = ["streamlit", "run", str(script)]
sys.exit(stcli.main())
