import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from geometry import coords_for_zigzag_chain, indices_nearest_neighbors
from circuits.trotter_circuit import compute_spin_imbalance, make_figure, create_trotter_circuit


st.set_page_config(page_title="Dipolar Spin Chain", layout="wide")
st.title("Dipolar Zigzag Spin Chain — Trotter Simulation")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Chain geometry")
    n_pairs = st.slider("Number of spin pairs", min_value=2, max_value=5, value=3)
    magic_angle = st.slider("Magic angle (°)", min_value=0.0, max_value=90.0, value=48.2, step=0.1)

    st.header("Quantization axis")
    q_theta = st.slider("Polar angle θ (°)", 0.0, 180.0, 60.0, step=1.0,
                         help="Angle from z-axis")
    q_phi = st.slider("Azimuthal angle φ (°)", 0.0, 360.0, 90.0, step=1.0,
                       help="Angle in the xy-plane from x-axis")
    q_theta_r = np.deg2rad(q_theta)
    q_phi_r = np.deg2rad(q_phi)
    quantization_axis = (
        np.sin(q_theta_r) * np.cos(q_phi_r),
        np.sin(q_theta_r) * np.sin(q_phi_r),
        np.cos(q_theta_r),
    )
    st.caption(f"q̂ = ({quantization_axis[0]:.3f}, {quantization_axis[1]:.3f}, {quantization_axis[2]:.3f})")

    st.header("Simulation")
    n_trotter_steps = st.slider("Trotter steps per time point", 1, 30, 10)
    max_time_pi = st.slider("Max time (× π)", 1, 10, 7)
    n_time_points = st.slider("Number of time points", 10, 100, 60)

    run = st.button("Run simulation", type="primary", use_container_width=True)

# ── Geometry panel ────────────────────────────────────────────────────────────
coords = coords_for_zigzag_chain(n_pairs=n_pairs, theta_deg=magic_angle)
pairs = indices_nearest_neighbors(coords)
n_qubits = len(coords)
neighbors = [[] for _ in range(n_qubits)]
for i, j in pairs:
    neighbors[i].append(j)
    neighbors[j].append(i)

spin_up = [i * 2 + 1 for i in range(n_pairs)]
spin_down = [i * 2 for i in range(n_pairs)]

col_geo, col_sim = st.columns([1, 2])

with col_geo:
    st.subheader("Chain geometry")
    coords_arr = np.array(coords)
    fig_geo, ax_geo = plt.subplots(figsize=(4, 4))
    colors = ["tab:blue" if i in spin_up else "tab:red" for i in range(n_qubits)]
    ax_geo.scatter(coords_arr[:, 0], coords_arr[:, 1], c=colors, s=80, zorder=3)
    for i, j in pairs:
        ax_geo.plot([coords_arr[i, 0], coords_arr[j, 0]],
                    [coords_arr[i, 1], coords_arr[j, 1]], "k-", lw=1)
    for i, (x, y, _) in enumerate(coords):
        label = f"{i}↑" if i in spin_up else f"{i}↓"
        ax_geo.text(x, y + 0.12, label, fontsize=8, ha="center")
    ax_geo.set_aspect("equal")
    ax_geo.axis("off")
    st.pyplot(fig_geo)
    plt.close(fig_geo)

# ── Simulation panel ──────────────────────────────────────────────────────────
with col_sim:
    st.subheader("Spin dynamics")

    if run:
        times = np.linspace(0, max_time_pi * np.pi, n_time_points)

        with st.spinner("Running Trotter simulation…"):
            t, site_mag, staggered_imbalance, sublattice = compute_spin_imbalance(
                coords, neighbors, spin_up, spin_down, quantization_axis,
                n_trotter_steps=n_trotter_steps, times=times,
            )

        fig_sim = make_figure(t, site_mag, staggered_imbalance, sublattice)

        # Re-label x-axis ticks to match chosen max_time_pi
        axes = fig_sim.get_axes()
        tick_locs = np.arange(0, max_time_pi + 1) * np.pi
        tick_labels = [r"$0$"] + [rf"${n}\pi$" for n in range(1, max_time_pi + 1)]
        axes[-1].set_xticks(tick_locs)
        axes[-1].set_xticklabels(tick_labels)
        fig_sim.tight_layout()

        st.pyplot(fig_sim)
        plt.close(fig_sim)
    else:
        st.info("Set parameters in the sidebar, then click **Run simulation**.")

# ── Circuit diagram ───────────────────────────────────────────────────────────
st.subheader("Single Trotter step circuit")
single_step = create_trotter_circuit(
    coords, neighbors, spin_up, spin_down, quantization_axis,
    n_trotter_steps=1, total_time=1.0,
)
fig_circ = single_step.draw(output="mpl", fold=-1)
st.pyplot(fig_circ)
plt.close(fig_circ)
