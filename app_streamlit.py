from rag_langchain.web.streamlit_app import *

if __name__ == "__main__":
    import sys
    if "streamlit" not in sys.modules:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
