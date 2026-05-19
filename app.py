"""
Top-level entry point for the combined Streamlit demo.

Run from the project root:
    streamlit run app.py

Two pages share one sidebar selector:
  • Trotter Circuit            → circuits/app.py
  • Dipolar Ladder (QuSpin)    → qspin/app_qspin.py

Each sub-app keeps its own logic file (circuits/trotter_circuit.py,
qspin/qspin_dipoles.py) and can also be run standalone:
    streamlit run circuits/app.py
    streamlit run qspin/app_qspin.py
"""
import os
import sys

import streamlit as st

# Make geometry.py and both sub-packages importable from the sub-app files.
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "circuits", "qspin"):
    p = os.path.join(HERE, sub) if sub else HERE
    if p not in sys.path:
        sys.path.insert(0, p)

# set_page_config must be called exactly once, here, before any other st.* call.
st.set_page_config(
    page_title="Quantum Many-Body Scar Demos",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Branding shown above the page picker in the sidebar.
with st.sidebar:
    st.title("QMBS Demos")
    st.caption(
        "Companion code for kinetic-frustration scars "
        "(Ding, Verresen & Yan, arXiv:2603.11191)."
    )
    st.divider()

pages = [
    st.Page(
        "circuits/app.py",
        title="Trotter Circuit",
        icon="🌀",
        default=True,
    ),
    st.Page(
        "qspin/app_qspin.py",
        title="Dipolar Ladder (QuSpin)",
        icon="🧲",
    ),
]

pg = st.navigation(pages)
pg.run()
