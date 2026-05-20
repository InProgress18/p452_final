import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from geometry import (
    coords_for_zigzag_chain,
    indices_nearest_neighbors,
    indices_next_nearest_neighbors,
)
from circuits.trotter_circuit import (
    compute_spin_imbalance,
    make_figure,
    create_trotter_circuit,
)

st.set_page_config(page_title="Dipolar Spin Chain", layout="wide")
st.title("Dipolar Zigzag Spin Chain — Trotter Simulation")

# ── Session state ─────────────────────────────────────────────────────────────
if "current_sim" not in st.session_state:
    st.session_state.current_sim = None
if "ref_sim" not in st.session_state:
    st.session_state.ref_sim = None

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Chain geometry")
    n_pairs = st.slider("Number of spin pairs", min_value=2, max_value=5, value=3)
    if "_pending_magic_angle" in st.session_state:
        st.session_state.magic_angle = st.session_state.pop("_pending_magic_angle")
    magic_angle = st.slider(
        "Magic angle (°)",
        min_value=0.0,
        max_value=90.0,
        value=48.2,
        step=0.1,
        key="magic_angle",
    )

    st.header("Quantization axis")
    q_theta = st.slider(
        "Polar angle θ (°)", 0.0, 180.0, 60.0, step=1.0, help="Angle from z-axis"
    )
    q_theta_r = np.deg2rad(q_theta)
    q_phi = 90.0
    q_phi_r = np.deg2rad(q_phi)  # WOLG
    quantization_axis = (
        np.sin(q_theta_r) * np.cos(q_phi_r),
        np.sin(q_theta_r) * np.sin(q_phi_r),
        np.cos(q_theta_r),
    )
    st.caption(
        f"q̂ = ({quantization_axis[0]:.3f}, {quantization_axis[1]:.3f}, {quantization_axis[2]:.3f})"
    )

    _magic_arg = 1.0 / (np.sqrt(3) * np.sin(q_theta_r)) if np.sin(q_theta_r) != 0 else np.inf
    if abs(_magic_arg) <= 1.0:
        _auto_angle = np.degrees(np.arccos(_magic_arg))
        if st.button(
            f"Auto-set magic angle → {_auto_angle:.1f}°",
            help="Sets magic angle to arccos(sin(θ)/√3)",
            use_container_width=True,
        ):
            st.session_state._pending_magic_angle = _auto_angle
            st.rerun()
    else:
        st.caption("No real magic angle for this θ.")

    st.header("Position disorder")
    z_noise_sigma = st.slider(
        "z-position noise σ",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Standard deviation of the Gaussian distribution used to sample z-offsets for each dipole before the circuit runs",
    )

    st.header("Interactions")
    include_nnn = st.checkbox(
        "Include next-nearest neighbor (NNN) interactions",
        value=False,
        help="Adds the second distance shell of dipole couplings to the Hamiltonian",
    )

    st.header("Simulation")
    n_trotter_steps = st.slider("Trotter steps per time point", 1, 30, 10)
    max_time_pi = st.slider("Max time (× π)", 1, 10, 7)
    n_time_points = st.slider("Number of time points", 10, 100, 60)

    run = st.button("Run simulation", type="primary", use_container_width=True)

# ── Geometry panel ────────────────────────────────────────────────────────────
coords = coords_for_zigzag_chain(n_pairs=n_pairs, theta_deg=magic_angle)
pairs = indices_nearest_neighbors(coords)
nnn_pairs = indices_next_nearest_neighbors(coords) if include_nnn else []
# Remove duplicates from nnn_pairs that are already in pairs
nnn_pairs = [
    pair for pair in nnn_pairs if pair not in pairs and (pair[1], pair[0]) not in pairs
]
n_qubits = len(coords)
neighbors = [[] for _ in range(n_qubits)]
for i, j in pairs + nnn_pairs:
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
        ax_geo.plot(
            [coords_arr[i, 0], coords_arr[j, 0]],
            [coords_arr[i, 1], coords_arr[j, 1]],
            "k-",
            lw=1,
        )
    for i, j in nnn_pairs:
        ax_geo.plot(
            [coords_arr[i, 0], coords_arr[j, 0]],
            [coords_arr[i, 1], coords_arr[j, 1]],
            "k--",
            lw=0.7,
            alpha=0.45,
        )
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

        if z_noise_sigma > 0.0:
            z_offsets = np.random.normal(0.0, z_noise_sigma, size=n_qubits)
            sim_coords = [(x, y, z + dz) for (x, y, z), dz in zip(coords, z_offsets)]
            sim_pairs = indices_nearest_neighbors(sim_coords)
            sim_nnn_pairs = (
                indices_next_nearest_neighbors(sim_coords) if include_nnn else []
            )
            sim_neighbors = [[] for _ in range(n_qubits)]
            for i, j in sim_pairs + sim_nnn_pairs:
                sim_neighbors[i].append(j)
                sim_neighbors[j].append(i)
        else:
            z_offsets = np.zeros(n_qubits)
            sim_coords = coords
            sim_neighbors = neighbors

        with st.spinner("Running Trotter simulation…"):
            t, site_mag, staggered_imbalance, sublattice = compute_spin_imbalance(
                sim_coords,
                sim_neighbors,
                spin_up,
                spin_down,
                quantization_axis,
                n_trotter_steps=n_trotter_steps,
                times=times,
            )

        if z_noise_sigma > 0.0:
            with st.expander("Sampled z-offsets"):
                for i, dz in enumerate(z_offsets):
                    st.write(f"Spin {i}: Δz = {dz:+.4f}")

        st.session_state.current_sim = {
            "t": t,
            "site_mag": site_mag,
            "staggered_imbalance": staggered_imbalance,
            "sublattice": sublattice,
            "max_time_pi": max_time_pi,
            "params_label": (
                f"N={n_pairs} pairs, chain θ={magic_angle:.1f}°, "
                f"q̂=({q_theta:.0f}°,{q_phi:.0f}°), σ={z_noise_sigma:.2f}, "
                f"steps={n_trotter_steps}" + (", +NNN" if include_nnn else "")
            ),
        }

    if st.session_state.current_sim is not None:
        curr = st.session_state.current_sim
        fig_sim = make_figure(
            curr["t"], curr["site_mag"], curr["staggered_imbalance"], curr["sublattice"]
        )
        axes = fig_sim.get_axes()
        mtp = curr["max_time_pi"]
        tick_locs = np.arange(0, mtp + 1) * np.pi
        tick_labels = [r"$0$"] + [rf"${n}\pi$" for n in range(1, mtp + 1)]
        axes[-1].set_xticks(tick_locs)
        axes[-1].set_xticklabels(tick_labels)
        fig_sim.tight_layout()
        st.pyplot(fig_sim)
        plt.close(fig_sim)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Save as Reference (Sim A)", use_container_width=True):
                st.session_state.ref_sim = dict(st.session_state.current_sim)
        with col_b2:
            if st.session_state.ref_sim is not None:
                if st.button("Clear Reference", use_container_width=True):
                    st.session_state.ref_sim = None

        if st.session_state.ref_sim is not None:
            st.caption(f"Reference saved: {st.session_state.ref_sim['params_label']}")
    else:
        st.info("Set parameters in the sidebar, then click **Run simulation**.")

# ── Comparison panel ──────────────────────────────────────────────────────────
if st.session_state.ref_sim is not None and st.session_state.current_sim is not None:
    ref = st.session_state.ref_sim
    curr = st.session_state.current_sim

    st.divider()
    st.subheader("Comparison: Sim A vs Sim B")

    col_a_info, col_b_info = st.columns(2)
    with col_a_info:
        st.markdown(f"**Sim A (reference):** {ref['params_label']}")
    with col_b_info:
        st.markdown(f"**Sim B (current):** {curr['params_label']}")

    same_n_qubits = ref["site_mag"].shape[1] == curr["site_mag"].shape[1]
    n_panels = 3 if same_n_qubits else 2
    fig_cmp, axes_cmp = plt.subplots(
        n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=False
    )

    # Staggered imbalance overlay
    axes_cmp[0].plot(
        ref["t"], ref["staggered_imbalance"], color="tab:blue", lw=2, label="Sim A"
    )
    axes_cmp[0].plot(
        curr["t"],
        curr["staggered_imbalance"],
        color="tab:orange",
        lw=2,
        ls="--",
        label="Sim B",
    )
    axes_cmp[0].axhline(0, color="k", ls="--", lw=0.5)
    axes_cmp[0].set_ylabel(r"$\frac{1}{N}\sum_i \varepsilon_i \langle Z_i \rangle$")
    axes_cmp[0].set_title("Staggered spin imbalance")
    axes_cmp[0].legend()

    # Difference (B − A) with interpolation onto a shared time grid
    t_min = max(ref["t"][0], curr["t"][0])
    t_max = min(ref["t"][-1], curr["t"][-1])
    if t_max > t_min:
        t_dense = np.linspace(t_min, t_max, 400)
        ref_interp = np.interp(t_dense, ref["t"], ref["staggered_imbalance"])
        curr_interp = np.interp(t_dense, curr["t"], curr["staggered_imbalance"])
        diff = curr_interp - ref_interp
        axes_cmp[1].plot(t_dense, diff, color="tab:red", lw=1.5)
        axes_cmp[1].fill_between(t_dense, diff, 0, alpha=0.2, color="tab:red")
    axes_cmp[1].axhline(0, color="k", ls="--", lw=0.5)
    axes_cmp[1].set_ylabel(r"$\Delta$ (B $-$ A)")
    axes_cmp[1].set_title("Difference (Sim B $-$ Sim A)")

    # Site magnetization overlay (only when chain sizes match)
    if same_n_qubits:
        n_q = ref["site_mag"].shape[1]
        site_colors = plt.cm.tab10(np.linspace(0, 0.9, n_q))
        for i in range(n_q):
            spin_sym = "↑" if ref["sublattice"][i] > 0 else "↓"
            axes_cmp[2].plot(
                ref["t"],
                ref["site_mag"][:, i],
                color=site_colors[i],
                lw=1.5,
                label=f"A: site {i}{spin_sym}",
            )
            axes_cmp[2].plot(
                curr["t"],
                curr["site_mag"][:, i],
                color=site_colors[i],
                lw=1.5,
                ls="--",
                label=f"B: site {i}{spin_sym}",
            )
        axes_cmp[2].axhline(0, color="k", ls="--", lw=0.5)
        axes_cmp[2].set_ylabel(r"$\langle Z_i \rangle$")
        axes_cmp[2].set_title("Site magnetizations  (solid = Sim A,  dashed = Sim B)")
        axes_cmp[2].legend(fontsize=7, ncol=2)

    mtp_max = max(ref["max_time_pi"], curr["max_time_pi"])
    tick_locs = np.arange(0, mtp_max + 1) * np.pi
    tick_labels = [r"$0$"] + [rf"${n}\pi$" for n in range(1, mtp_max + 1)]
    axes_cmp[-1].set_xlabel("Time")
    axes_cmp[-1].set_xticks(tick_locs)
    axes_cmp[-1].set_xticklabels(tick_labels)

    fig_cmp.tight_layout()
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

# ── Circuit diagram ───────────────────────────────────────────────────────────
st.subheader("Single Trotter step circuit")
single_step = create_trotter_circuit(
    coords,
    neighbors,
    spin_up,
    spin_down,
    quantization_axis,
    n_trotter_steps=1,
    total_time=1.0,
)
fig_circ = single_step.draw(output="mpl", fold=-1)
st.pyplot(fig_circ)
plt.close(fig_circ)
