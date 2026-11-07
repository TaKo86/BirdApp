"""show previous results from this session and let the user clear them."""

import streamlit as st

import services
from ui import page_heading

page_heading("YOUR FIELD NOTES", "Session history", "A record of the sounds you've explored, newest first.")
rows = services.load_history()
if rows:
    summary, action = st.columns([3, 1])
    summary.caption(f"{len(rows)} analysis entries · Times shown in New Zealand local time")
    if action.button("Clear history", width="stretch"):
        services.clear_history()
        st.rerun()
    st.dataframe(
        [{"Time": row["time"], "Filename": row["filename"], "Species": row["species"],
          "Confidence": f"{row['confidence']:.1%}"} for row in rows],
        hide_index=True, width="stretch",
    )
    st.caption("These are placeholder predictions. History is kept only for this session and may reset when you reload.")
else:
    with st.container(border=True):
        st.subheader("Your field notes start here")
        st.write("No recordings analysed yet. Try an upload or the example clip to create your first entry.")
        st.page_link("pages/1_Classify.py", label="Go to Classify recording")
