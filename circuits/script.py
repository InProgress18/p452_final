import sys, json
sys.path.insert(0, r'C:\Users\mvayn\OneDrive\Documents\GitHub\p452_final')
import numpy as np
from circuits.trotter_circuit import generate_zigzag_chain_matching_angles, compute_dipolar_couplings, angles_from_reference

coords = generate_zigzag_chain_matching_angles(n_pairs=3, alpha_deg=34.0, beta_deg=32.45, dx=1.0)
q = (0.0, float(np.cos(np.deg2rad(30.0))), float(np.sin(np.deg2rad(30.0))))
J = compute_dipolar_couplings(coords, quantization_axis=q, dipole_strength=1.0)
out = {'coords': coords.tolist(), 'angles_deg': angles_from_reference(coords, ref=0, neigh=(1,3)), 'J': J.tolist(), 'q_axis': list(q)}
print(json.dumps(out))
