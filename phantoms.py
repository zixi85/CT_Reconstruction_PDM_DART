"""
Requirements:
scikit-image==0.24.0
scikit-learn==1.5.2
scipy==1.13.1
matplotlib==3.9.2
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.draw import disk, ellipse, polygon
from scipy.ndimage import gaussian_filter
import scipy.optimize
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import draw
import numpy as np
from PIL import Image
# Bezier Curve Function
def bezier_curve(t, p0, p1, p2):
    """Quadratic Bezier curve function."""
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def draw_bezier(img, p0, p1, p2, intensity=0.9, thickness=2):
    """Draws a smooth Bezier curve with thickness to avoid dotted appearance."""
    for t in np.linspace(0, 1, 200):  # More points for a smoother curve
        x = int(bezier_curve(t, p0[0], p1[0], p2[0]))
        y = int(bezier_curve(t, p0[1], p1[1], p2[1]))
        if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
            rr, cc = disk((x, y), thickness, shape=img.shape)  # Use a small disk to thicken the curve
            img[rr, cc] = intensity  # Bright curve

    return img

# ---- Main Function: Generate Filled Phantom with Curves & Lines ----
def generate_filled_phantom(resolution=512, num_shapes=50, num_lines=10, num_curves=5, noise_type="both", seed=42):
    """Generates a phantom filled with random shapes, lines, and curved structures.
     Parameters:
    - resolution: Image size (e.g., 512x512)
    - num_shapes: number of random shapes like polygon, ellipse, disk
    - num_lines: Number of straight lines randomly placed on the image
    - num_curves: Number of curved lines using Bezier curves
    - noise_type: None, "poisson", "gaussian", or "both" (for CT phantom)
    - seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    img = np.zeros((resolution, resolution))

    # ---- Add Random Shapes (Circles, Ellipses, Polygons) ----
    for _ in range(num_shapes):
        shape_type = random.choice(["disk", "ellipse", "polygon"])
        x, y = np.random.randint(0, resolution, size=2)
        intensity = np.random.uniform(0.2, 0.9)

        if shape_type == "disk":
            radius = np.random.randint(10, 50)
            rr, cc = disk((x, y), radius, shape=img.shape)
        elif shape_type == "ellipse":
            a, b = np.random.randint(10, 50, size=2)
            rr, cc = ellipse(y, x, a, b, shape=img.shape)
        else:
            num_points = np.random.randint(3, 6)
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            polygon_x = [int(x + np.random.randint(20, 50) * np.cos(a)) for a in angles]
            polygon_y = [int(y + np.random.randint(20, 50) * np.sin(a)) for a in angles]
            rr, cc = polygon(polygon_y, polygon_x, shape=img.shape)

        img[rr, cc] = intensity


    # ---- Add Bezier Curves for Smooth Paths ----
    for _ in range(num_curves):
        p0 = np.random.randint(0, resolution, size=2)
        p1 = np.random.randint(0, resolution, size=2)
        p2 = np.random.randint(0, resolution, size=2)
        img = draw_bezier(img, p0, p1, p2, intensity=np.random.uniform(0.9, 1))


    # ---- Add Noise ----
    img = np.clip(img, 0, 1)

    if noise_type == 'gaussian' or noise_type == 'both':
        img = gaussian_filter(img, sigma=1) + np.random.normal(0, 0.03, img.shape)
        img = np.clip(img, 0, 1)

    if noise_type == 'poisson' or noise_type == 'both':
        img = np.random.poisson(img * 255) / 255

    img = np.clip(img, 0, 1)
    print(f"The grey level intensities for Filled:",len(np.unique(img)))
    return img


def draw_starburst(img, center, radius, num_spikes=8, structure = "ellipses"):
    """ Creates a starburst pattern: a circle with outward-facing triangular spikes. """
    # Draw the main circular shape
    rr, cc = disk((center, center), radius, shape=img.shape)
    img[rr, cc] = 1  # Fill circle

    # Generate spikes using triangles
    angles = np.linspace(0, 2 * np.pi, num_spikes, endpoint=False)  # Evenly spaced angles
    if structure == 'ellipses':
        for i in range(5):
            rr, cc = ellipse(center, center, radius//4 + i*5, radius//3 - i*10, shape=img.shape)
            img[rr, cc] = 0.3 + (i * 0.1)  # Gradient intensity


    for angle in angles:
        # Base points on the circle edge
        base_x = int(center + radius * np.cos(angle))
        base_y = int(center + radius * np.sin(angle))

        # Triangle tip extending outward
        tip_x = int(center + (radius * 1.5) * np.cos(angle))  # Extend further
        tip_y = int(center + (radius * 1.5) * np.sin(angle))

        # Side points of the triangle (adjust width of the base)
        side_angle1 = angle + np.pi / num_spikes
        side_angle2 = angle - np.pi / num_spikes

        side1_x = int(center + radius * np.cos(side_angle1))
        side1_y = int(center + radius * np.sin(side_angle1))

        side2_x = int(center + radius * np.cos(side_angle2))
        side2_y = int(center + radius * np.sin(side_angle2))

        # Draw the triangular spike
        rr, cc = polygon([base_y, side1_y, tip_y, side2_y],
                         [base_x, side1_x, tip_x, side2_x], shape=img.shape)
        img[rr, cc] = 1  # Fill triangle

    return img
def generate_phantom(phantom_type="resolution", resolution=512, num_spikes=10, num_ellipses=10, noise_type=None, seed=None):
    """
    Generates a flexible phantom based on user selection:
    - "basic": Basic starburst pattern.
    - "resolution": Starburst pattern with high-frequency ellipses (for resolution testing).
    - "ct": Realistic CT-like phantom with soft tissue, bones, and air pockets.
    - "filled": Filled phantom with random shapes, lines, and curved structures.

    Parameters:
    - phantom_type: "resolution" or "ct" or "basic" or "filled
    - resolution: Image size (e.g., 512x512)
    - num_spikes: Number of spikes in the resolution phantom (for starburst)
    - num_ellipses: Number of ellipses in the resolution phantom
    - noise_type: None, "poisson", "gaussian", or "both" (for CT phantom)
    - seed: Random seed for reproducibility

    Returns:
    - NumPy array representing the phantom.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    img = np.zeros((resolution, resolution))
    center = resolution // 2
    radius = resolution // 3

    if phantom_type == "basic":
      img = draw_starburst(img, center, radius, num_spikes)
      img = np.nan_to_num(img, nan=0.0)
      img = np.clip(img, 0, 1)

      if noise_type == 'gaussian' or noise_type == 'both':
          img = gaussian_filter(img, sigma=1) + np.random.normal(0, 0.03, img.shape)
          img = np.clip(img, 0, 1)

      if noise_type == 'poisson' or noise_type == 'both':
          img = np.random.poisson(img * 255) / 255

      img = np.clip(img, 0, 1)
      print(f"The grey level intensities for {phantom_type}:",len(np.unique(img)))

    if phantom_type == "resolution":
        #  Create Starburst Shape
        rr, cc = disk((center, center), radius, shape=img.shape)
        img[rr, cc] = 0  # Fill circle

        angles = np.linspace(0, 2 * np.pi, num_spikes , endpoint=False)

        for angle in angles:
            base_x = int(center + radius * np.cos(angle))
            base_y = int(center + radius * np.sin(angle))

            tip_x = int(center + (radius * 1.3) * np.cos(angle))
            tip_y = int(center + (radius * 1.3) * np.sin(angle))

            side_angle1 = angle + np.pi / num_spikes
            side_angle2 = angle - np.pi / num_spikes

            side1_x = int(center + radius * np.cos(side_angle1))
            side1_y = int(center + radius * np.sin(side_angle1))

            side2_x = int(center + radius * np.cos(side_angle2))
            side2_y = int(center + radius * np.sin(side_angle2))

            rr, cc = polygon([base_y, side1_y, tip_y, side2_y],
                             [base_x, side1_x, tip_x, side2_x], shape=img.shape)
            img[rr, cc] = 1  # Fill triangle within the star

        #  Add High-Frequency Ellipses for Resolution Testing
        for i in range(num_ellipses):
            ellipse_x = int(center + random.uniform(radius * 0.3, radius * 0.8) * np.cos(random.uniform(0, 2 * np.pi)))
            ellipse_y = int(center + random.uniform(radius * 0.3, radius * 0.8) * np.sin(random.uniform(0, 2 * np.pi)))

            a = max(2, radius // (5 + i))
            b = max(2, radius // (6 + i))

            rr, cc = ellipse(ellipse_y, ellipse_x, a, b, shape=img.shape)
            img[rr, cc] = 0.2 + (i * 0.07)
        print(f"The grey level intensities for {phantom_type}:",len(np.unique(img)))

    elif phantom_type == "ct":
        print("Gererating CT-like phantom...")
        #  Create Elliptical Body Shape
        body_radius_x = resolution // 2.2
        body_radius_y = resolution // 3
        rr, cc = ellipse(center, center, body_radius_y, body_radius_x, shape=img.shape)
        img[rr, cc] = 0.5  # Soft tissue

        #  Add Organ-Like Structure
        num_organs = num_ellipses
        for _ in range(num_organs):
            cx = random.randint(int(center - body_radius_x // 2), int(center + body_radius_x // 2))
            cy = random.randint(int(center - body_radius_y // 2), int(center + body_radius_y // 2))
            a = random.randint(20, 60)
            b = random.randint(15, 50)
            intensity = random.uniform(0.4, 0.6)

            rr, cc = ellipse(cy, cx, a, b, shape=img.shape)
            img[rr, cc] = intensity

        # Add High-Contrast "Bone" Structure (Spine)
        spine_x = center
        spine_y = center + body_radius_y // 3
        spine_rr, spine_cc = ellipse(spine_y, spine_x, 30, 12, shape=img.shape)
        img[spine_rr, spine_cc] = 0.9  # High-intensity for bone

        #  simulate Low intensity for air regions
        for _ in range(3):
            air_x = random.randint(int(center - body_radius_x // 3), int(center + body_radius_x // 3))
            air_y = random.randint(int(center - body_radius_y // 3), int(center + body_radius_y // 3))
            air_rr, air_cc = ellipse(air_y, air_x, 25, 15, shape=img.shape)
            img[air_rr, air_cc] = 0.1  # Low intensity for air regions

        #  Add Blood Vessel-Like Structures ( Mid-intensity for vessels)
        for _ in range(5):
            vessel_x = random.randint(int(center - body_radius_x // 3),int( center + body_radius_x // 3))
            vessel_y = random.randint(int(center - body_radius_y // 3), int(center + body_radius_y // 3))
            vessel_a = random.randint(10, 20)
            vessel_b = random.randint(5, 15)
            vessel_rr, vessel_cc = ellipse(vessel_y, vessel_x, vessel_a, vessel_b, shape=img.shape)
            img[vessel_rr, vessel_cc] = 0.7

        #  Noise for Realism : guassian or poisson
        img = np.nan_to_num(img, nan=0.0)
        img = np.clip(img, 0, 1)

        if noise_type == 'gaussian' or noise_type == 'both':
            img = gaussian_filter(img, sigma=1) + np.random.normal(0, 0.03, img.shape)
            img = np.clip(img, 0, 1)

        if noise_type == 'poisson' or noise_type == 'both':
            img = np.random.poisson(img * 255) / 255

        img = np.clip(img, 0, 1)
        print(f"The grey level intensities for {phantom_type}:",len(np.unique(img)))
    return img
def create_phantom(phantom_type, noise_type=None, seed=42):
    # Generate and Display All Phantoms
    print(f"create phantom for {phantom_type}")
    
   
    if phantom_type == "filled":
        img = generate_filled_phantom(resolution=512, num_shapes=20, num_lines=0, num_curves=2, noise_type=noise_type, seed=seed)

    else:
                        
        img = generate_phantom(phantom_type=phantom_type, resolution=512, num_spikes=10, num_ellipses=5, noise_type=noise_type, seed=seed)
        
        
    # plt.imshow(img, cmap='gray')
    # plt.title(f"{phantom_type.capitalize()} Phantom")
    # plt.axis("off")
    # plt.show()
    return img
