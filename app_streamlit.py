import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_langchain.web.streamlit_app import *

if __name__ == "__main__":
    import sys
    if "streamlit" not in sys.modules:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())