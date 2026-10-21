"""start the app with: uv run streamlit run app/streamlit_app.py."""

import streamlit as st

from ui import apply_styles, demo_banner

st.set_page_config(page_title="NZ Bird Call Recognition", layout="wide")
apply_styles()

pages = [
    st.Page("pages/1_Classify.py", title="Classify recording", default=True),
    st.Page("pages/2_History.py", title="Session history"),
    st.Page("pages/3_About.py", title="About the project"),
]
page = st.navigation(pages, position="hidden")  # we draw the menu ourselves, below the app name

with st.sidebar:
    st.markdown("**NZ Bird Call Recognition**")
    st.divider()
    for p in pages:
        st.page_link(p)
    st.divider()
    st.caption("University project demo. No trained model is connected. "
               "Recordings stay in memory and history lasts for this session only.")

demo_banner()
page.run()
