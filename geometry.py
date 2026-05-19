import matplotlib.pyplot as plt
import numpy as np


def coords_for_zigzag_chain(
    n_pairs, theta_deg=54.74, alpha_deg=34, beta_deg=32.45, dx=1.0
):
    """
    Generate coordinates for a zigzag chain of pairs of spins.

    Parameters:
    - n_pairs: Number of pairs in the chain
    - theta_deg: Magic angle
    - alpha_deg: Angle between the second spin of one pair and the first spin of the next pair (in degrees)
    - beta_deg: Angle between the second spin of one pair and the first spin of the next pair (in degrees)
    - dx: Distance along the x-axis between consecutive pairs

    Returns:
    - coords: A list of 3D coordinates for each spin in the chain
    """
    coords = []
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)
    x1, y1, z1 = 0.0, 0.0, 0.0
    x2, y2, z2 = dx, 0.0, 0.0
    dx_prime = (
        dx
        * (np.tan(beta_rad) - np.tan(alpha_rad))
        / (np.tan(alpha_rad) + np.tan(beta_rad))
    )
    dy = 2 * dx / (np.tan(alpha_rad) + np.tan(beta_rad))

    for i in range(n_pairs):
        # First spin of the pair
        if i % 2 == 0:
            coords.append((x1, y1, z1))

            # Second spin of the pair
            coords.append((x2, y2, z2))

        else:
            coords.append((x2, y2, z2))

            # Second spin of the pair
            coords.append((x1, y1, z1))

        # Update coordinates for the next pair
        x1 += dx_prime
        y1 += dy
        z1 += 0.0

        x2 += dx_prime
        y2 += dy
        z2 += 0.0  # No change in z for the second spin of the pair
    # compute the angle needed to rotate so the second pair is on the y-axis
    angle_to_y_axis_rad = np.arctan2(
        coords[3][0] - coords[0][0], coords[3][1] - coords[0][1]
    )
    # rotate the entire chain by -angle_to_y_axis_rad to align the second pair with the y-axis
    cos_angle = np.cos(angle_to_y_axis_rad)
    sin_angle = np.sin(angle_to_y_axis_rad)
    coords = [
        (x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle, z)
        for (x, y, z) in coords
    ]

    # rotate the entire chain clockwise by theta around the z-axis
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    coords = [
        (x * cos_theta + y * sin_theta, -x * sin_theta + y * cos_theta, z)
        for (x, y, z) in coords
    ]
    return coords


def compute_distances(coord1, coord2):
    """
    Compute the distance between two 3D coordinates.

    Parameters:
    - coord1: A tuple (x1, y1, z1) representing the first coordinate
    - coord2: A tuple (x2, y2, z2) representing the second coordinate

    Returns:
    - distance: The Euclidean distance between the two coordinates
    """
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def compute_dipole_interaction(coord1, coord2, quantization_axis):
    """
    Compute the dipole-dipole interaction strength between two spins based on their coordinates and a quantization axis.

    Parameters:
    - coord1: A tuple (x1, y1, z1) representing the first coordinate
    - coord2: A tuple (x2, y2, z2) representing the second coordinate
    - quantization_axis: A tuple (qx, qy, qz) representing the quantization axis

    Returns:
    - interaction_strength: The computed dipole-dipole interaction strength
    """
    r_vec = np.array(coord2) - np.array(coord1)
    r = np.linalg.norm(r_vec)
    if r == 0:
        return 0.0  # Avoid division by zero
    r_hat = r_vec / r
    q_hat = np.array(quantization_axis) / np.linalg.norm(quantization_axis)

    # Compute the dipole-dipole interaction strength using the formula:
    # D = (μ₀ / (4π)) * (γ₁ * γ₂ * ħ² / r³) * [3(q̂ · r̂)(q̂ · r̂) - 1]
    # For simplicity, we can set the prefactor to 1 and focus on the angular dependence
    interaction_strength = (3 * np.dot(q_hat, r_hat) ** 2 - 1) / r**3
    return interaction_strength


def indices_nearest_neighbors(coords):
    """
    Compute the indices of nearest neighbors based on the coordinates.

    Parameters:
    - coords: A list of 3D coordinates for each spin in the chain

    Returns:
    - neighbors: A list of tuples (i, j) where i and j are indices of nearest neighbor spins
    """
    neighbors = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if (
                compute_distances(coords[i], coords[j]) < 2
            ):  # Threshold distance for nearest neighbors
                neighbors.append((i, j))
    return neighbors


def indices_next_nearest_neighbors(coords):
    """Return next-nearest neighbor pairs — the second distance shell.

    Identifies the NNN shell by finding all pairwise distances, isolating the
    minimum (NN) distance, then finding the next distinct distance cluster.
    A 15% tolerance is used to group distances into shells.
    """
    neighbors = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if (
                compute_distances(coords[i], coords[j]) < 4
            ):  # Threshold distance for nearest neighbors
                neighbors.append((i, j))
    return neighbors


if __name__ == "__main__":
    coords = coords_for_zigzag_chain(n_pairs=3, theta_deg=48.2)
    coords = np.array(coords)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0], coords[:, 1])
    # print the coordinates in a nice format
    for i, (x, y, z) in enumerate(coords):
        print(f"Spin {i}: ({x:.3f}, {y:.3f}, {z:.3f})")

    # draw lines between nearest neighbors
    neighbors = indices_nearest_neighbors(coords)
    for i, j in neighbors:
        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], "k-")
    # Compute and print the dipole-dipole interaction strengths between all pairs of spins
    quantization_axis = (0, np.cos(np.pi / 6), np.sin(np.pi / 6))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            interaction_strength = compute_dipole_interaction(
                coords[i], coords[j], quantization_axis
            )
            print(
                f"Interaction strength between Spin {i} and Spin {j}: {interaction_strength:.6f}"
            )
    # Display the nearest neighbor interactions on the plot next to the lines
    for i, j in neighbors:
        interaction_strength = compute_dipole_interaction(
            coords[i], coords[j], quantization_axis
        )
        mid_x = (coords[i][0] + coords[j][0]) / 2
        mid_y = (coords[i][1] + coords[j][1]) / 2
        # Display the interaction strength as text on the plot, put them slightly above the line (by translating orthogonally to the line)
        line_vec = np.array(coords[j]) - np.array(coords[i])
        line_length = np.linalg.norm(line_vec)
        if line_length > 0:
            line_unit_vec = line_vec / line_length
            orthogonal_vec = np.array(
                [-line_unit_vec[1], line_unit_vec[0], 0]
            )  # Rotate by 90 degrees in the xy-plane
            text_x = (
                mid_x + orthogonal_vec[0] * 0.1
            )  # Adjust the distance of the text from the line
            text_y = mid_y + orthogonal_vec[1] * 0.1
            ax.text(
                text_x,
                text_y,
                f"{interaction_strength:.2f}",
                fontsize=8,
                ha="center",
                va="center",
            )
    # display the indices of the spins as text on the plot next to the points
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y + 0.1, f"{i}", fontsize=8, ha="center", va="center")

    plt.show()
