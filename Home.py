import streamlit as st
from ui import STEPS, session_progress

st.set_page_config(page_title="Tourism Forecast Lab", page_icon="", layout="wide")




page = st.navigation({
    "Welcome": [st.Page("overview.py", title="Overview", icon="🌴", default=True)],
    "Prepare data": [st.Page(path, title=f"{i + 1}. {name}", url_path=f"step-{name.lower()}") for i, (name, path, _, _) in enumerate(STEPS[:4])],
    "Model & insights": [st.Page(path, title=f"{i + 5}. {name}", url_path=f"step-{name.lower()}") for i, (name, path, _, _) in enumerate(STEPS[4:])],
})
try:
    page.run()
finally:
    session_progress()
