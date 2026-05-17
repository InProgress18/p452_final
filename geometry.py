import matplotlib.pyplot as plt
import numpy as np


def coords_for_zigzag_chain(n_pairs, theta_deg=54.74, alpha_deg=34, beta_deg=32.45, dx=1.0):
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
    dx_prime = dx * (np.tan(beta_rad) - np.tan(alpha_rad)) / (np.tan(alpha_rad) + np.tan(beta_rad))
    dy =  2 * dx / (np.tan(alpha_rad) + np.tan(beta_rad))
    
    for i in range(n_pairs):
        # First spin of the pair
        coords.append((x1, y1, z1))

        # Second spin of the pair
        coords.append((x2, y2, z2))
        
        # Update coordinates for the next pair
        x1 += dx_prime
        y1 += dy
        z1 += 0.0  
        
        x2 += dx_prime
        y2 += dy
        z2 += 0.0  # No change in z for the second spin of the pair
    # compute the angle needed to rotate so the second pair is on the y-axis
    angle_to_y_axis_rad = np.arctan2(coords[2][0] - coords[0][0], coords[2][1] - coords[0][1])
    # rotate the entire chain by -angle_to_y_axis_rad to align the second pair with the y-axis
    cos_angle = np.cos(angle_to_y_axis_rad)
    sin_angle = np.sin(angle_to_y_axis_rad)
    coords = [(x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle, z) for (x, y, z) in coords]
    
    
    # rotate the entire chain clockwise by theta around the z-axis
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    coords = [(x * cos_theta + y * sin_theta, -x * sin_theta + y * cos_theta, z) for (x, y, z) in coords]
    return coords

if __name__ == "__main__":
    coords = coords_for_zigzag_chain(n_pairs=3)
    coords = np.array(coords)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0], coords[:, 1])
    plt.show()
    # print the coordinates in a nice format
    for i, (x, y, z) in enumerate(coords):
        print(f"Spin {i}: ({x:.3f}, {y:.3f}, {z:.3f})")
        

