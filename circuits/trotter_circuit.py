from geometry import (
    coords_for_zigzag_chain,
    compute_dipole_interaction,
    indices_nearest_neighbors,
)
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit.quantum_info import Statevector, SparsePauliOp

magic_angle = 48.2
n_pairs = 3
coords = coords_for_zigzag_chain(n_pairs=n_pairs, theta_deg=magic_angle)

spin_up = [i * 2 + 1 for i in range(n_pairs)]
spin_down = [i * 2 for i in range(n_pairs)]


def create_trotter_circuit(
    coords,
    neighbors,
    spin_up,
    spin_down,
    quantization_axis,
    n_trotter_steps,
    total_time=1.0,
):
    """Create a trotterized circuit for the Heisenberg model for a dipolar zigzag chain.

    Implements the first-order Suzuki-Trotter decomposition of the Heisenberg
    Hamiltonian H = Σ_{i<j} J_ij (S_x^i S_x^j + S_y^i S_y^j + S_z^i S_z^j),
    where J_ij = compute_dipole_interaction(coords[i], coords[j], quantization_axis)
    and S_α = σ_α / 2.

    Each Trotter step applies e^{-i J_ij dt/4 * XX} e^{-i J_ij dt/4 * YY} e^{-i J_ij dt/4 * ZZ}
    per pair, using Qiskit's RXX/RYY/RZZ gates (RXX(θ) = e^{-i θ/2 XX}).

    Args:
        coords (list[tuple[int, int, int]]): A list of 3D coordinates for each spin in the chain
        neighbors (list[list[int]]): A list of lists, where each inner list contains the indices of neighboring spins for the corresponding spin
        spin_up (list[int]): A list of indices for spins initialized in the up state
        spin_down (list[int]): A list of indices for spins initialized in the down state
        quantization_axis (tuple[int, int, int]): The axis along which to quantify the spins
        n_trotter_steps (int): The number of Trotter steps to include in the circuit
        total_time (float): The total evolution time (default 1.0)

    Returns:
        qiskit.QuantumCircuit: The trotterized circuit for the Heisenberg model
    """
    n_qubits = len(coords)
    circuit = qiskit.QuantumCircuit(n_qubits)

    # |0⟩ = spin-up, |1⟩ = spin-down
    for i in spin_down:
        circuit.x(i)
    # Add Trotter steps for the Heisenberg interactions
    

