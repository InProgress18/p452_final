"""Trotterized Heisenberg circuit builder for a 2xN lattice.

This module provides utilities to:
- compute pairwise dipolar coupling strengths from real-space coordinates
- construct a Trotterized Heisenberg evolution circuit using Qiskit

Assumptions / Notes:
- Spins are spin-1/2 and represented by qubits 0..(N-1).
- By default a simple first-order (Lie) Trotter decomposition is used:
  e^{-i H t} ≈ \prod_p \prod_{(i,j)} e^{-i J_{ij} t (X_iX_j + Y_iY_j + Z_iZ_j)/n_steps}
- The dipolar "scalar" coupling used is the common secular form
  J_{ij} = dipole_strength * (1 - 3 cos^2(theta)) / r^3
  where theta is the angle between the connecting vector and the quantization axis.

The code is intentionally self-contained and keeps the physical prefactor
(`dipole_strength`) as a user parameter so it can be set to physical units
or a dimensionless value for simulations.

Example usage:
>>> from trotter_circuit import generate_2xN_coords, trotter_heisenberg_circuit
>>> coords = generate_2xN_coords(3, spacing=(1.0,1.0))
>>> qc, J = trotter_heisenberg_circuit(coords, total_time=1.0, n_steps=2, dipole_strength=1.0, lattice_shape=(2,3))
>>> print(qc)
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
	"generate_2xN_coords",
	"compute_dipolar_couplings",
	"nearest_neighbor_pairs_2xN",
	"pairs_from_adjacency",
	"pairs_from_mask",
	"validate_pairs",
	"trotter_heisenberg_circuit",
]


def generate_2xN_coords(n_cols: int, spacing: Tuple[float, float] = (1.0, 1.0), origin: Tuple[float, float] = (0.0, 0.0), z: float = 0.0) -> np.ndarray:
	"""Generate coordinates for a 2xN rectangular lattice.

	Indexing is row-major: index = row * n_cols + col, with rows 0..1 and cols 0..n_cols-1.

	Args:
		n_cols: number of columns N (there are always 2 rows).
		spacing: (dx, dy) lattice spacing between neighbouring sites.
		origin: (x0, y0) coordinate of the (row=0, col=0) site.
		z: z coordinate for all sites (default 0).

	Returns:
		coords: ndarray of shape (2*n_cols, 3)
	"""
	dx, dy = spacing
	x0, y0 = origin
	coords: List[Tuple[float, float, float]] = []
	for row in range(2):
		for col in range(n_cols):
			x = x0 + col * dx
			y = y0 + row * dy
			coords.append((x, y, z))
	return np.array(coords, dtype=float)


def compute_dipolar_couplings(coords: Sequence[Sequence[float]], quantization_axis: Sequence[float] = (0.0, 0.0, 1.0), dipole_strength: float = 1.0, cutoff: Optional[float] = None, eps: float = 1e-12) -> np.ndarray:
	"""Compute symmetric dipolar coupling matrix J_{ij} from coordinates.

	Uses the scalar secular-like form
		J_ij = dipole_strength * (1 - 3 cos^2(theta_ij)) / r^3
	where theta_ij is the angle between r_ij and the supplied `quantization_axis`.

	Args:
		coords: iterable of (x,y,z) coordinates with length N.
		quantization_axis: preferred quantization axis (will be normalized).
		dipole_strength: overall prefactor (user-specified units).
		cutoff: optional maximum distance beyond which couplings are set to zero.
		eps: small-distance cutoff to avoid division by zero.

	Returns:
		J: ndarray shape (N,N) with symmetric couplings and zero diagonal.
	"""
	coords_arr = np.array(coords, dtype=float)
	if coords_arr.ndim != 2 or coords_arr.shape[1] not in (2, 3):
		raise ValueError("coords must be (N,2) or (N,3)-shaped array-like")
	if coords_arr.shape[1] == 2:
		# append z=0
		zcol = np.zeros((coords_arr.shape[0], 1), dtype=float)
		coords_arr = np.hstack([coords_arr, zcol])

	n_sites = coords_arr.shape[0]
	q_axis = np.array(quantization_axis, dtype=float)
	if q_axis.shape != (3,):
		raise ValueError("quantization_axis must be 3-dimensional")
	q_norm = np.linalg.norm(q_axis)
	if q_norm < eps:
		raise ValueError("quantization_axis must be non-zero")
	q_unit = q_axis / q_norm

	J = np.zeros((n_sites, n_sites), dtype=float)
	for i in range(n_sites):
		for j in range(i + 1, n_sites):
			rij = coords_arr[j] - coords_arr[i]
			r = np.linalg.norm(rij)
			if r < eps:
				raise ValueError(f"Two spins are at the same position (indices {i},{j}).")
			if cutoff is not None and r > cutoff:
				continue
			r_hat = rij / r
			cos_theta = float(np.dot(r_hat, q_unit))
			Jval = float(dipole_strength * (1.0 - 3.0 * cos_theta * cos_theta) / (r ** 3))
			J[i, j] = Jval
			J[j, i] = Jval
	return J


def nearest_neighbor_pairs_2xN(n_cols: int) -> List[Tuple[int, int]]:
	"""Return nearest-neighbour pairs for a 2xN lattice (row-major indexing).

	Pairs include horizontal neighbours within each row and vertical neighbours
	between the two rows at the same column.
	"""
	pairs: List[Tuple[int, int]] = []
	# horizontal neighbors
	for row in range(2):
		for col in range(n_cols - 1):
			a = row * n_cols + col
			b = row * n_cols + (col + 1)
			pairs.append((a, b))
	# vertical neighbours
	for col in range(n_cols):
		a = 0 * n_cols + col
		b = 1 * n_cols + col
		pairs.append((a, b))
	return pairs


def pairs_from_adjacency(adj: np.ndarray) -> List[Tuple[int, int]]:
	"""Convert an adjacency matrix (or boolean mask) to a list of unique pairs (i<j).

	Args:
		adj: square (N,N) array-like adjacency or boolean mask. Non-zero entries are treated as edges.

	Returns:
		List of (i,j) with i<j for which adj[i,j] is truthy.
	"""
	arr = np.array(adj)
	if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
		raise ValueError("adjacency must be a square (N,N) array or matrix")
	n = arr.shape[0]
	pairs: List[Tuple[int, int]] = []
	for i in range(n):
		for j in range(i + 1, n):
			if arr[i, j]:
				pairs.append((i, j))
	return pairs


def pairs_from_mask(mask: Sequence[Sequence[bool]]) -> List[Tuple[int, int]]:
	"""Alias for pairs_from_adjacency accepting Python sequence-of-sequence booleans."""
	return pairs_from_adjacency(np.array(mask, dtype=bool))


def validate_pairs(pairs: Sequence[Tuple[int, int]], n_qubits: int) -> List[Tuple[int, int]]:
	"""Validate and normalize a sequence of index pairs for n_qubits.

	Ensures each pair is (i,j) with 0 <= i < j < n_qubits.
	"""
	out: List[Tuple[int, int]] = []
	for raw in pairs:
		if len(raw) != 2:
			raise ValueError(f"each pair must have length 2, got {raw}")
		i, j = int(raw[0]), int(raw[1])
		if i == j:
			raise ValueError(f"pair indices must be distinct: {raw}")
		if not (0 <= i < n_qubits) or not (0 <= j < n_qubits):
			raise IndexError(f"pair indices out of range for {n_qubits} qubits: {raw}")
		if i > j:
			i, j = j, i
		out.append((i, j))
	# remove duplicates while preserving order
	seen = set()
	uniq: List[Tuple[int, int]] = []
	for p in out:
		if p not in seen:
			seen.add(p)
			uniq.append(p)
	return uniq


def trotter_heisenberg_circuit(coords: Sequence[Sequence[float]], total_time: float, n_steps: int = 1, quantization_axis: Sequence[float] = (0.0, 0.0, 1.0), dipole_strength: float = 1.0, coupling_matrix: Optional[np.ndarray] = None, lattice_shape: Optional[Tuple[int, int]] = None, pairs: Optional[Sequence[Tuple[int, int]]] = None, use_all_pairs: bool = False) -> Tuple["QuantumCircuit", np.ndarray]:
	"""Build a Qiskit QuantumCircuit implementing a Trotterized Heisenberg evolution.

	Args:
		coords: coordinates of the spins (N x 3 or N x 2). The i-th row maps to qubit i.
		total_time: total evolution time t (the circuit implements U = exp(-i H t)).
		n_steps: number of Trotter steps.
		quantization_axis: axis used to compute dipolar couplings.
		dipole_strength: scalar prefactor for dipolar couplings.
		coupling_matrix: optional precomputed (N,N) matrix of J_{ij}. If provided,
			this overrides `coords`/`quantization_axis`/`dipole_strength`.
		lattice_shape: optional (rows, cols) shape; if provided and equals (2,N)
			the default pairs will be nearest neighbours for that lattice.
		pairs: optional explicit list of (i,j) pairs to include in the Trotter product.
		use_all_pairs: if True, include all unique pairs (i<j) using `coupling_matrix`.

	Returns:
		qc: `QuantumCircuit` implementing the Trotterized evolution.
		J: the coupling matrix used (N x N ndarray).
	"""
	coords_arr = np.array(coords, dtype=float)
	if coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
		coords_arr = np.hstack([coords_arr, np.zeros((coords_arr.shape[0], 1), dtype=float)])
	n_qubits = coords_arr.shape[0]

	if coupling_matrix is None:
		J = compute_dipolar_couplings(coords_arr, quantization_axis, dipole_strength)
	else:
		J = np.array(coupling_matrix, dtype=float)
		if J.shape != (n_qubits, n_qubits):
			raise ValueError("coupling_matrix shape must match number of qubits")

	if pairs is None:
		if lattice_shape is not None:
			rows, cols = lattice_shape
			if rows == 2 and cols * rows == n_qubits:
				pairs = nearest_neighbor_pairs_2xN(cols)
			else:
				# fallback to all pairs if lattice_shape doesn't match coords
				pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
		elif use_all_pairs:
			pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
		else:
			# by default for arbitrary coords use all pairs (user can pass pairs to restrict)
			pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
	else:
		# If user passed an adjacency matrix or boolean mask, convert it to pair list
		try:
			arr = np.array(pairs)
		except Exception:
			arr = None
		if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
			pairs = pairs_from_adjacency(arr)
		else:
			pairs = validate_pairs(pairs, n_qubits)

	# import qiskit lazily so compute helpers are import-safe (only when building circuits)
	try:
		from qiskit import QuantumCircuit
		from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
	except Exception as exc:
		raise ImportError("qiskit is required to build the quantum circuit. Install qiskit (pip install qiskit) to use trotter_heisenberg_circuit.") from exc

	qc = QuantumCircuit(n_qubits)
	dt = float(total_time) / float(max(1, n_steps))

	for step in range(n_steps):
		for (i, j) in pairs:
			Jval = float(J[i, j])
			if abs(Jval) < 1e-15:
				continue
			angle = 2.0 * Jval * dt
			# apply e^{-i J dt XX}, e^{-i J dt YY}, e^{-i J dt ZZ}
			qc.append(RXXGate(angle), [i, j])
			qc.append(RYYGate(angle), [i, j])
			qc.append(RZZGate(angle), [i, j])

	return qc, J


if __name__ == "__main__":
	# small demo for a 2x3 lattice
	demo_coords = generate_2xN_coords(3, spacing=(1.0, 1.0))
	try:
		circuit, Jmat = trotter_heisenberg_circuit(demo_coords, total_time=1.0, n_steps=2, dipole_strength=1.0, lattice_shape=(2, 3))
		print("Coupling matrix J:")
		print(np.round(Jmat, 5))
		print("Circuit:")
		print(circuit)
	except ImportError as exc:
		Jmat = compute_dipolar_couplings(demo_coords)
		print("Coupling matrix J:")
		print(np.round(Jmat, 5))
		print("Qiskit not installed; circuit construction skipped:", exc)

	# Demo: arbitrary Cartesian positions and selecting specific couplings
	coords_custom = [(0.0, 0.0), (1.0, 0.2), (0.5, 0.9), (2.0, 0.1)]
	custom_pairs = [(0, 2), (1, 3)]
	try:
		qc2, J2 = trotter_heisenberg_circuit(coords_custom, total_time=0.5, n_steps=1, dipole_strength=1.0, pairs=custom_pairs)
		print("\nCustom coupling matrix J:")
		print(np.round(J2, 5))
		print("Custom circuit (selected pairs):")
		print(qc2)
	except ImportError as exc:
		J2 = compute_dipolar_couplings(coords_custom)
		print("\nCustom coupling matrix J:")
		print(np.round(J2, 5))
		print("Qiskit not installed; custom circuit construction skipped:", exc)

	# Demo: using an adjacency mask (boolean matrix)
	adj = np.zeros((4, 4), dtype=bool)
	adj[0, 1] = adj[1, 0] = True
	adj[2, 3] = adj[3, 2] = True
	try:
		qc3, J3 = trotter_heisenberg_circuit(coords_custom, total_time=0.2, n_steps=1, pairs=adj)
		print("\nAdjacency-selected coupling matrix J:")
		print(np.round(J3, 5))
		print("Adjacency-selected circuit:")
		print(qc3)
	except ImportError as exc:
		J3 = compute_dipolar_couplings(coords_custom)
		print("\nAdjacency-selected coupling matrix J:")
		print(np.round(J3, 5))
		print("Qiskit not installed; adjacency-selected circuit construction skipped:", exc)

