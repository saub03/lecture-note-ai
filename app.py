"""Streamlit entrypoint for lecture-note-ai."""

import streamlit as st

st.set_page_config(page_title="lecture-note-ai", page_icon="🎙️")
st.title("🎙️ lecture-note-ai")

st.file_uploader("Upload lecture slides (PDF/PPT)", type=["pdf", "ppt", "pptx"])
st.button("🎙️ Start recording")
st.text_area("Structured notes", placeholder="Lecture notes will appear here...")
