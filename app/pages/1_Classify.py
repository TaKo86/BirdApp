"""upload a clip or try the example, then show the demo result and spectrogram."""

import hashlib
import time

import streamlit as st

import services
from ui import page_heading, probability_chart, result_card

page_heading("LISTEN / DISCOVER", "What can you hear?",
             "Explore a recording from Aotearoa's soundscape, or try the example to see how it works.")
left, right = st.columns([1.7, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Your recording")
        upload = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"],
                                  max_upload_size=25, help="Recordings are decoded in memory and are never saved to disk.")
        example = st.toggle("Use example clip", help="A five-second synthetic tone, not a real bird recording.")
        st.caption("WAV, MP3, M4A, OGG or FLAC · Up to 25 MB")
with right:
    st.subheader("A small window into the wild")
    st.write("Upload a recording, listen back, then explore a possible match and the shape of its sound.")
    st.caption("The match is a demo placeholder. The spectrogram is made from the audio you choose.")
    st.caption("01  Choose audio\n\n02  Explore the result\n\n03  Verify what you heard")

raw = services.get_example_clip() if example else (upload.getvalue() if upload is not None else None)
filename = "synthetic-example.wav" if example else (upload.name if upload is not None else "")
if raw is None:
    st.session_state.pop("active_recording", None)
    st.session_state.pop("analysis", None)
    st.info("Ready when you are. Upload a recording or turn on Use example clip.")
else:
    identity = (filename, hashlib.sha256(raw).hexdigest())
    if st.session_state.get("active_recording") != identity:
        st.session_state["active_recording"] = identity
        st.session_state.pop("analysis", None)
    try:
        details = services.get_audio_details(raw)
        playback = services.get_playback_audio(raw)
    except services.AudioError as exc:
        st.error(str(exc))
        st.stop()
    spectrogram = None
    with st.container(border=True):
        name_col, duration_col, rate_col = st.columns([2, 1, 1])
        name_col.caption("RECORDING")
        name_col.text(filename)
        duration_col.metric("Duration", f"{details['duration']:.2f} s")
        rate_col.metric("Sample rate", f"{details['sample_rate']:,} Hz")
        st.audio(playback, format="audio/wav")
        if example:
            st.caption("Synthetic example · generated in memory · not a bird call")
        if st.button("Analyse recording", type="primary", width="stretch"):
            st.session_state.pop("analysis", None)
            try:
                with st.spinner("Listening to your recording…"):
                    started = time.monotonic()
                    result = services.classify(raw, filename)
                    spectrogram = services.get_spectrogram_figure(raw)
                    time.sleep(max(0, 1.0 - (time.monotonic() - started)))
                    services.save_to_history(result, filename)
                    st.session_state["analysis"] = result
            except services.AudioError as exc:
                st.error(str(exc))
    result = st.session_state.get("analysis")
    if result is not None:
        st.divider()
        st.subheader("Your recording, explored")
        match_col, chart_col = st.columns([1, 1.2], gap="large")
        with match_col:
            result_card(result, services.get_species_details(result["label"]))
        with chart_col:
            with st.container(border=True):
                st.subheader("Top 5 possible matches")
                st.caption("Demo probabilities · other classes account for the remaining probability")
                chart = probability_chart(result["probabilities"])
                st.pyplot(chart)
                chart.clear()
        with st.container(border=True):
            st.subheader("See the sound")
            st.caption("Real log-Mel spectrogram · first 5 seconds, padded if shorter · brighter colours show stronger sound")
            figure = spectrogram if spectrogram is not None else services.get_spectrogram_figure(raw)
            st.pyplot(figure)
            figure.clear()
        st.caption("Saved to Session history. Your audio is not saved to disk.")
