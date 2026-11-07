"""explain the project, demo birds and models we plan to compare."""

import streamlit as st

import services
from ui import page_heading

page_heading("THE PROJECT", "Listening to Aotearoa", "Exploring how machine learning could help recognise New Zealand bird calls.")
st.subheader("From recording to a possible match")
for column, title, text in zip(st.columns(4), ["01 / Upload", "02 / Process", "03 / Classify", "04 / Verify"], [
    "Bring a field recording or try our synthetic example.",
    "Listen back and view a real Mel spectrogram of the sound.",
    "Explore a placeholder match and five demo probabilities.",
    "Listen carefully and check important findings with an expert.",
]):
    with column.container(border=True):
        st.subheader(title)
        st.write(text)
st.subheader("Models we plan to compare")
st.dataframe([
    {"Approach": "SVM", "Planned input": "Summarised MFCC audio features", "Status": "Planned"},
    {"Approach": "Random Forest", "Planned input": "Summarised MFCC audio features", "Status": "Planned"},
    {"Approach": "CNN", "Planned input": "Log-Mel spectrograms", "Status": "Planned"},
], hide_index=True, width="stretch")
st.caption("No training, model comparison or measured accuracy is included in this demo.")
species = services.get_demo_species()
st.subheader(f"Meet the demo birds · {len(species)} placeholders")
st.caption("An illustrative list, not an authoritative set of model classes. Kiwi represents a group of species.")
columns = st.columns(2, gap="medium")
for index, bird in enumerate(species):
    with columns[index % 2].container(border=True):
        st.subheader(bird["display_name"])
        st.caption(f"Māori name · {bird['maori_name']}")
        st.write(bird["fact"])
        st.caption(f"[Species notes · DOC]({bird['source']})")
st.subheader("Know the limits")
st.write("This is a demo with no trained model or connected dataset. Matches and confidence values are invented, "
         "even for silence, synthetic tones or recordings without birds. The spectrogram alone is genuine audio analysis.")
st.write("Noise, overlapping calls and birds outside the class list will need careful evaluation in a real system. "
         "Always verify results. This app is not a replacement for expert judgement.")
st.subheader("Credits")
st.write("Project team, dataset sources, recording licences and acknowledgements: to be added before release.")
st.caption("Species notes link to the Department of Conservation. The example tone is generated in memory; no bird audio is bundled.")
