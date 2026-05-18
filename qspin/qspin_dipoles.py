"""
Coordinate-based tilted dipolar ladder simulation using QuSpin.

"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


# ============================================================
# Import original geometry.py
# ============================================================

import geometry
geometry = importlib.reload(geometry)


# ============================================================
# Helper functions not defined in geometry.py
# ============================================================

def get_bond(Jij, i, j):
    """Return J_ij from a Jij dictionary."""
    if i > j:
        i, j = j, i
    return Jij.get((i, j), 0.0)


def bond_category(i, j, L):
    """
    Categorize bonds using the convention:

        top:     0 -- 2 -- 4 -- ...
        bottom:  1 -- 3 -- 5 -- ...

    Categories:
        rung
        top_leg
        bottom_leg
        plaquette_diagonal
        other_long_range
    """
    if i > j:
        i, j = j, i

    # Rungs: (0,1), (2,3), (4,5), ...
    for r in range(L):
        if (i, j) == tuple(sorted((2*r, 2*r + 1))):
            return "rung"

    # Inter-rung bonds and plaquette diagonals
    for r in range(L - 1):
        top_left = 2*r
        bot_left = 2*r + 1
        top_right = 2*(r + 1)
        bot_right = 2*(r + 1) + 1

        if (i, j) == tuple(sorted((top_left, top_right))):
            return "top_leg"

        if (i, j) == tuple(sorted((bot_left, bot_right))):
            return "bottom_leg"

        if (i, j) == tuple(sorted((top_left, bot_right))):
            return "plaquette_diagonal"

        if (i, j) == tuple(sorted((bot_left, top_right))):
            return "plaquette_diagonal"

    return "other_long_range"


def build_all_dipolar_couplings_from_geometry(
    coords,
    quantization_axis,
    normalize_bond=(0, 1),
    target_ref=1.0,
    max_distance=None,
):
    """
    Build all-pair dipolar couplings using original geometry.py.

    Raw coupling from geometry.py:

        J_ij_raw = [3(qhat.rhat)^2 - 1] / r^3

    Then normalize all couplings by a reference bond.
    """
    coords = np.array(coords, dtype=float)
    N = len(coords)

    Jij_raw = {}

    for i in range(N):
        for j in range(i + 1, N):
            dist = geometry.compute_distances(coords[i], coords[j])

            if max_distance is not None and dist > max_distance:
                continue

            Jij_raw[(i, j)] = geometry.compute_dipole_interaction(
                coords[i],
                coords[j],
                quantization_axis,
            )

    i0, j0 = normalize_bond
    if i0 > j0:
        i0, j0 = j0, i0

    J_ref = Jij_raw[(i0, j0)]

    if abs(J_ref) < 1e-12:
        raise ValueError(f"Reference coupling J_{i0}{j0} is too close to zero.")

    scale = target_ref / J_ref

    Jij_norm = {
        bond: scale * val
        for bond, val in Jij_raw.items()
    }

    return Jij_norm, Jij_raw


def keep_designed_bonds_only(Jij, L):
    """
    Keep designed tilted-ladder bonds only.

    For L=3:
        rungs:       (0,1), (2,3), (4,5)
        top leg:     (0,2), (2,4)
        bottom leg:  (1,3), (3,5)
    """
    keep = set()

    # Rungs
    for r in range(L):
        keep.add(tuple(sorted((2*r, 2*r + 1))))

    # Designed inter-rung bonds
    for r in range(L - 1):
        keep.add(tuple(sorted((2*r, 2*(r + 1)))))          # top leg
        keep.add(tuple(sorted((2*r + 1, 2*(r + 1) + 1))))  # bottom leg

    return {
        bond: val
        for bond, val in Jij.items()
        if bond in keep
    }


def print_coordinates_and_couplings(coords, Jij, quantization_axis, L):
    """
    Print coordinates and normalized couplings grouped by category.
    """
    coords = np.array(coords, dtype=float)

    print("\n================ Coordinates ================")
    print("site        x           y           z")
    print("----------------------------------------------")

    for i, (x, y, z) in enumerate(coords):
        print(f"{i:>3d}   {x:+.6f}   {y:+.6f}   {z:+.6f}")

    print("\n================ Normalized couplings by category ================")

    q = np.array(quantization_axis, dtype=float)
    q = q / np.linalg.norm(q)

    categories = [
        "rung",
        "top_leg",
        "bottom_leg",
        "plaquette_diagonal",
        "other_long_range",
    ]

    for cat in categories:
        print(f"\n--- {cat} ---")
        found = False

        for (i, j), val in sorted(Jij.items()):
            if bond_category(i, j, L) != cat:
                continue

            found = True

            dist = geometry.compute_distances(coords[i], coords[j])

            r_vec = coords[j] - coords[i]
            r_hat = r_vec / np.linalg.norm(r_vec)
            cos_theta = np.dot(q, r_hat)
            theta_deg = np.rad2deg(np.arccos(np.clip(abs(cos_theta), 0.0, 1.0)))

            print(
                f"J_{i}{j} = {val:+.6f}    "
                f"distance = {dist:.6f}    "
                f"theta = {theta_deg:.3f} deg"
            )

        if not found:
            print("(none)")


def print_coupling_table(coords, Jij, quantization_axis, L,
                         title="Coupling strengths"):
    """
    Print a compact, easy-to-scan table of every coupling.

    Columns: bond label, category, geometric distance, angle theta between bond
    vector and quantization axis, signed J, |J|. Rows are sorted by |J|
    descending so the dominant couplings appear first.
    """
    coords = np.array(coords, dtype=float)

    q = np.array(quantization_axis, dtype=float)
    q = q / np.linalg.norm(q)

    rows = []
    for (i, j), val in Jij.items():
        dist = geometry.compute_distances(coords[i], coords[j])
        r_vec = coords[j] - coords[i]
        r_hat = r_vec / np.linalg.norm(r_vec)
        cos_theta = float(np.dot(q, r_hat))
        theta_deg = np.rad2deg(np.arccos(np.clip(abs(cos_theta), 0.0, 1.0)))

        rows.append({
            "bond": f"J_{i}{j}",
            "i": i,
            "j": j,
            "category": bond_category(i, j, L),
            "distance": dist,
            "theta_deg": theta_deg,
            "J": val,
            "abs_J": abs(val),
        })

    rows_sorted = sorted(rows, key=lambda r: -r["abs_J"])

    width = 80
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print(
        f"  {'Bond':<8}{'Category':<22}"
        f"{'Distance':>10}{'theta(deg)':>12}"
        f"{'J':>14}{'|J|':>12}"
    )
    print("  " + "-" * (width - 2))
    for r in rows_sorted:
        print(
            f"  {r['bond']:<8}{r['category']:<22}"
            f"{r['distance']:>10.4f}{r['theta_deg']:>12.3f}"
            f"{r['J']:>+14.5f}{r['abs_J']:>12.5f}"
        )
    print("=" * width)
    print()

    return rows_sorted


def print_magic_diagnostics(Jij):
    """
    First plaquette diagnostics.

    Desired frustration:
        J_02 + J_13 ≈ 0

    Magic-angle suppressed diagonals:
        J_03 ≈ 0
        J_12 ≈ 0
    """
    print("\n================ Magic/frustration diagnostics ================")

    J01 = get_bond(Jij, 0, 1)
    J02 = get_bond(Jij, 0, 2)
    J13 = get_bond(Jij, 1, 3)
    J03 = get_bond(Jij, 0, 3)
    J12 = get_bond(Jij, 1, 2)

    print(f"J_01 = {J01:+.6f}")
    print(f"J_02 = {J02:+.6f}")
    print(f"J_13 = {J13:+.6f}")
    print(f"J_02 + J_13 = {J02 + J13:+.6e}")
    print(f"J_03 = {J03:+.6e}")
    print(f"J_12 = {J12:+.6e}")


def _build_value_color_map(Jij, group_precision=3, cmap_name="tab10"):
    """
    Build a {rounded_J -> matplotlib color} map so bonds with the same
    coupling value share a color. Larger |J| values get earlier colors in
    the cycle so dominant bonds stand out.
    """
    groups = {}
    for (i, j), val in Jij.items():
        key = round(val, group_precision)
        groups.setdefault(key, []).append((i, j))

    sorted_keys = sorted(groups.keys(), key=lambda k: (-abs(k), -k))
    cmap = plt.get_cmap(cmap_name)
    color_for_key = {
        k: cmap(idx % cmap.N)
        for idx, k in enumerate(sorted_keys)
    }
    return color_for_key, sorted_keys, groups


def plot_normalized_couplings(
    coords,
    Jij,
    L,
    title="Normalized couplings used in QuSpin",
    label_digits=2,
    label_scale=0.20,
    group_precision=3,
):
    """
    Plot normalized QuSpin couplings.

    Bonds with the same J value within `group_precision` decimals share a
    color. Coupling labels are shown as numbers only.
    """
    coords = np.array(coords, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    if len(Jij) == 0:
        ax.set_title(title + " (empty)")
        plt.show()
        return

    max_abs_J = max(abs(v) for v in Jij.values())

    color_for_key, sorted_keys, _groups = _build_value_color_map(
        Jij,
        group_precision=group_precision,
    )

    for idx, ((i, j), val) in enumerate(sorted(Jij.items())):
        xi, yi = coords[i, 0], coords[i, 1]
        xj, yj = coords[j, 0], coords[j, 1]

        key = round(val, group_precision)
        color = color_for_key[key]

        lw = 1.0 + 2.5 * abs(val) / max_abs_J
        alpha = 0.55 + 0.4 * abs(val) / max_abs_J

        ax.plot(
            [xi, xj],
            [yi, yj],
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=1,
        )

        xm = 0.5 * (xi + xj)
        ym = 0.5 * (yi + yj)

        ri = coords[i, :2]
        rj = coords[j, :2]
        bond_vec = rj - ri
        norm = np.linalg.norm(bond_vec)

        if norm > 1e-12:
            bond_unit = bond_vec / norm
            perp = np.array([-bond_unit[1], bond_unit[0]])
        else:
            perp = np.array([0.0, 0.0])

        cat = bond_category(i, j, L)

        if cat == "top_leg":
            sign = +1.0
        elif cat == "bottom_leg":
            sign = -1.0
        elif cat == "plaquette_diagonal":
            sign = +1.0 if (i % 2 == 0) else -1.0
        elif cat == "rung":
            sign = +0.6
        else:
            sign = +1.0 if (idx % 2 == 0) else -1.0

        offset = sign * label_scale * perp

        ax.text(
            xm + offset[0],
            ym + offset[1],
            f"{val:+.{label_digits}f}",
            fontsize=11,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.30",
                fc="white",
                ec=color,
                lw=1.4,
                alpha=0.95,
            ),
            zorder=5,
        )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=130,
        color="black",
        edgecolor="white",
        linewidth=1.4,
        zorder=3,
    )

    for i, (x, y, _z) in enumerate(coords):
        ax.text(
            x + 0.04,
            y + 0.04,
            str(i),
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=6,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            color=color_for_key[k],
            lw=3,
            label=f"J = {k:+.{label_digits + 1}f}",
        )
        for k in sorted_keys
    ]

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title="Coupling values",
        title_fontsize=10,
        frameon=True,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def set_pi_ticks(ax, xmax):
    """
    Set x-axis ticks at n*pi.
    Assumes x-axis is dimensionless, e.g. |J_02|t.
    """
    nmax = int(np.floor(xmax / np.pi))
    ticks = [n * np.pi for n in range(nmax + 1)]

    labels = []
    for n in range(nmax + 1):
        if n == 0:
            labels.append(r"$0$")
        elif n == 1:
            labels.append(r"$\pi$")
        else:
            labels.append(rf"${n}\pi$")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


# ============================================================
# QuSpin helper functions
# ============================================================

def build_quspin_xy_hamiltonian(basis, Jij):
    """
    Build:

        H = sum_{i<j} J_ij (Sx_i Sx_j + Sy_i Sy_j)

    pauli=False means QuSpin uses spin operators Sx,Sy,Sz.
    """
    J_list = [
        [val, i, j]
        for (i, j), val in Jij.items()
        if abs(val) > 1e-14
    ]

    static = [
        ["xx", J_list],
        ["yy", J_list],
    ]

    H = hamiltonian(
        static,
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_symm=False,
        check_pcon=False,
    )

    return H


def sigma_z_operator(basis, site):
    """
    pauli=False:
        QuSpin 'z' is Sz with eigenvalues +/- 1/2.
        Pauli sigma_z = 2 Sz.
    """
    return hamiltonian(
        [["z", [[2.0, site]]]],
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_symm=False,
        check_pcon=False,
    )


def product_state_from_string(basis, state_str):
    """
    Build product state from QuSpin spin string.
    """
    psi = np.zeros(basis.Ns, dtype=np.complex128)
    psi[basis.index(state_str)] = 1.0
    return psi


def evolve_state(H, psi0, times):
    """
    Evolve psi0 under H.

    QuSpin may return shape (Ns, Nt), so transpose if needed.
    """
    psis = H.evolve(psi0, 0.0, times)

    if psis.shape[0] == psi0.size:
        psis = psis.T

    return psis


def compute_generalized_imbalance(z_ops, psis_t, sigma0):
    """
    Generalized imbalance:

        I(t) = (1/N) sum_i <sigma_z_i(t)> sigma_z_i(0)
    """
    N = len(sigma0)
    out = np.zeros(psis_t.shape[0], dtype=float)

    for ti, psi in enumerate(psis_t):
        val = 0.0

        for i in range(N):
            zi = np.vdot(psi, z_ops[i].dot(psi)).real
            val += zi * sigma0[i]

        out[ti] = val / N

    return out


def run_case(label, basis, psi0, z_ops, sigma0, Jij, times):
    """
    Build Hamiltonian, evolve state, compute imbalance and fidelity.
    """
    H = build_quspin_xy_hamiltonian(basis, Jij)

    psis_t = evolve_state(H, psi0, times)

    imbalance = compute_generalized_imbalance(
        z_ops=z_ops,
        psis_t=psis_t,
        sigma0=sigma0,
    )

    fidelity = np.abs(psis_t @ psi0.conj()) ** 2

    print(f"\n--- {label} ---")
    print(f"Number of couplings = {len(Jij)}")
    print(f"Hilbert-space dimension = {basis.Ns}")

    return imbalance, fidelity


# ============================================================
# Main simulation
# ============================================================

if __name__ == "__main__":
    L = 3
    N = 2 * L
    Nup = L

    alpha_deg = 34.0
    beta_deg = 32.45
    theta_deg = 48.2

    coords = geometry.coords_for_zigzag_chain(
        n_pairs=L,
        theta_deg=theta_deg,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        dx=1.0,
    )

    coords = np.array(coords, dtype=float)

    quantization_axis = (
        0.0,
        np.cos(np.pi / 6),
        np.sin(np.pi / 6),
    )

    # Normalize so that |J_02| = 1, while preserving the original sign of every
    # coupling. This matches the paper's |J_02| t time-axis convention.
    J02_raw = geometry.compute_dipole_interaction(
        coords[0],
        coords[2],
        quantization_axis,
    )

    target_J02 = float(np.sign(J02_raw))

    Jij_full, Jij_raw = build_all_dipolar_couplings_from_geometry(
        coords=coords,
        quantization_axis=quantization_axis,
        normalize_bond=(0, 2),
        target_ref=target_J02,
        max_distance=None,
    )

    Jij_designed = keep_designed_bonds_only(
        Jij=Jij_full,
        L=L,
    )


    # ============================================================
    # Geometry diagnostics, tables, and plots
    # ============================================================

    print_coordinates_and_couplings(
        coords=coords,
        Jij=Jij_full,
        quantization_axis=quantization_axis,
        L=L,
    )

    print_magic_diagnostics(Jij_full)

    # Compact tables sorted by |J|.
    print_coupling_table(
        coords=coords,
        Jij=Jij_designed,
        quantization_axis=quantization_axis,
        L=L,
        title="Designed bonds only: coupling strengths (sorted by |J|)",
    )

    print_coupling_table(
        coords=coords,
        Jij=Jij_full,
        quantization_axis=quantization_axis,
        L=L,
        title="Full all-pair dipolar: coupling strengths (sorted by |J|)",
    )

    # Plot only normalized couplings used in QuSpin.
    plot_normalized_couplings(
        coords=coords,
        Jij=Jij_designed,
        L=L,
        title="Designed bonds only: normalized QuSpin couplings",
        label_digits=2,
        label_scale=0.20,
    )

    plot_normalized_couplings(
        coords=coords,
        Jij=Jij_full,
        L=L,
        title="Full all-pair dipolar: normalized QuSpin couplings",
        label_digits=2,
        label_scale=0.24,
    )


    # ============================================================
    # Time axis in units of |J_02|^{-1}
    # ============================================================

    J02 = get_bond(Jij_full, 0, 2)
    J02_abs = abs(J02)

    if J02_abs < 1e-12:
        raise ValueError("J_02 is too close to zero; cannot use J_02^{-1} as time unit.")

    print(f"\nJ_02 = {J02:+.6f}")
    print("Using x-axis x = |J_02| t, labeled in units of n*pi.")

    times = np.linspace(0.0, 8.0 * np.pi / J02_abs, 2000)
    x_axis = J02_abs * times


    # ============================================================
    # Basis and initial state
    # ============================================================

    basis = spin_basis_1d(N, Nup=Nup, pauli=False)

    state_str = "01" * L

    psi0 = product_state_from_string(basis, state_str)

    sigma0 = np.array(
        [+1 if c == "1" else -1 for c in state_str],
        dtype=float,
    )

    z_ops = [
        sigma_z_operator(basis, i).tocsr()
        for i in range(N)
    ]

    print("\n================ QuSpin basis ================")
    print(f"N = {N}")
    print(f"Nup = {Nup}")
    print(f"Hilbert dimension = {basis.Ns}")
    print(f"Initial state string = {state_str}")
    print(f"sigma0 = {sigma0}")


    # ============================================================
    # Run simulations
    # ============================================================

    imb_designed, fid_designed = run_case(
        label="designed bonds only",
        basis=basis,
        psi0=psi0,
        z_ops=z_ops,
        sigma0=sigma0,
        Jij=Jij_designed,
        times=times,
    )

    imb_full, fid_full = run_case(
        label="full all-pair dipolar",
        basis=basis,
        psi0=psi0,
        z_ops=z_ops,
        sigma0=sigma0,
        Jij=Jij_full,
        times=times,
    )


    # ============================================================
    # Plot imbalance
    # ============================================================

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x_axis, imb_designed, label="designed bonds only")
    ax.plot(x_axis, imb_full, "--", label="full all-pair dipolar")

    set_pi_ticks(ax, xmax=x_axis[-1])

    ax.set_xlabel(r"$|J_{02}|t$")
    ax.set_ylabel(r"$I(t)$")
    ax.set_title(r"Coordinate-based tilted dipolar ladder: imbalance")
    ax.legend()

    plt.tight_layout()
    plt.show()


    # ============================================================
    # Plot fidelity
    # ============================================================

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x_axis, fid_designed, label="designed bonds only")
    ax.plot(x_axis, fid_full, "--", label="full all-pair dipolar")

    set_pi_ticks(ax, xmax=x_axis[-1])

    ax.set_xlabel(r"$|J_{02}|t$")
    ax.set_ylabel(r"$F(t)$")
    ax.set_title(r"Coordinate-based tilted dipolar ladder: fidelity")
    ax.legend()

    plt.tight_layout()
    plt.show()