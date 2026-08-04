"""Hub Streamlit — lance st.navigation entre les pages Documents / Chat / Connecteurs."""
from __future__ import annotations

import streamlit as st
from pathlib import Path


def main() -> None:
    st.set_page_config(
        page_title="RAG LangChain",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    pages_dir = Path(__file__).parent / "pages"
    nav = st.navigation([
        st.Page(str(pages_dir / "0_Documents.py"), title="Documents", icon="📄", default=True),
        st.Page(str(pages_dir / "1_Chat.py"), title="Chat", icon="💬"),
        st.Page(str(pages_dir / "2_Connecteurs.py"), title="Connecteurs", icon="🔌"),
    ])
    nav.run()


if __name__ == "__main__":
    main()