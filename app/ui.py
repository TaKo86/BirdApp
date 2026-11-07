"""shared layout helpers for the app pages."""

import streamlit as st
from matplotlib.figure import Figure

from services import Prediction

# match the chart colours to the theme in .streamlit/config.toml
PRIMARY, MUTED, LIGHT = "#0f6b73", "#6b7280", "#b8c4c6"
GREEN = PRIMARY  # old name, kept so any page that still imports GREEN keeps working
ACCENT = "#b45309"


def apply_styles() -> None:
    """adjust spacing and headings; the theme handles colours and fonts."""
    st.markdown("""<style>
    .block-container { max-width: 1100px; padding-top: 4.5rem; padding-bottom: 3rem; }
    h1 { font-size: 2rem !important; font-weight: 600 !important; }
    @media (max-width: 640px) { .block-container { padding-top: 4rem; } }
    </style>""", unsafe_allow_html=True)


def demo_banner() -> None:
    """show the demo reminder on every page."""
    st.info("Demo mode: predictions are placeholders until the model is trained.")


def page_heading(kicker: str, title: str, description: str) -> None:
    """show the title and intro; keep kicker so older page calls still work."""
    st.title(title)
    st.write(description)


def result_card(result: Prediction, species: dict[str, str]) -> None:
    """show the result and bird details from services."""
    with st.container(border=True):
        st.caption("Result (demo)" if result["mock"] else "Result")
        st.subheader(result["display_name"])
        st.write(species["maori_name"])
        st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.1%}")
        if result["mock"]:
            st.caption("Placeholder confidence, not a measure of identification accuracy.")
        st.divider()
        st.write(species["fact"])
        if species["source"]:
            st.caption(f"[Species notes (Department of Conservation)]({species['source']})")
        if result["confidence"] < 0.6:
            st.warning("Low confidence. Please verify this result with an expert.")
        else:
            st.caption("Always verify important detections.")
        st.caption(result["model"])


def probability_chart(probabilities: dict[str, float]) -> Figure:
    """plot the top five probabilities on a 0-100% scale."""
    items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    fig = Figure(figsize=(6, 3.4), layout="constrained")
    ax = fig.subplots()
    bars = ax.barh([name for name, _ in items], [value for _, value in items],
                   color=[PRIMARY] + [LIGHT] * (len(items) - 1), height=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=10)
    ax.bar_label(bars, labels=[f"{value:.1%}" for _, value in items], padding=4, color="#1d2327", fontsize=10)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig
