import numpy as np
import astra
import matplotlib.pyplot as plt
import os
from skimage.io import imsave
from scipy.ndimage import gaussian_filter


def create_astra_geometry(phantom_shape, num_angles, detector_factor=None, geometry_type='parallel'):
    """
    Create ASTRA projection and volume geometry.
    
    Parameters:
        phantom_shape: (height, width) of the volume
        num_angles: number of projection angles
        detector_size: number of detector elements (optional)
        geometry_type: 'parallel' or 'fanflat'
        
    Returns:
        proj_geom, vol_geom
    """
    
    det_count = int(detector_factor * phantom_shape[0])
    
    vol_geom = astra.create_vol_geom(phantom_shape[0], phantom_shape[1])
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    if geometry_type == 'parallel':
        det_spacing = 1.0
        proj_geom = astra.create_proj_geom('parallel', det_spacing, det_count, angles)
    
    elif geometry_type == 'fanflat':
        # Fan beam settings
        det_spacing = 1.0
        DSD = 1200.0  # Distance Source to Detector
        DSO = 1000.0  # Distance Source to Object (volume center)

        proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, DSD)
    
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}")
    
    return proj_geom, vol_geom

def generate_sinogram_astra(phantom, proj_geom, vol_geom, geometry_type='parallel'):
    """
    Generate sinogram of a phantom using ASTRA.

    Parameters:
        phantom: Input phantom image (2D numpy array)
        proj_geom: ASTRA projection geometry
        vol_geom: ASTRA volume geometry

    Returns:
        sinogram: Generated sinogram (angles × detector positions)
    """
    phantom_id = astra.data2d.create('-vol', vol_geom, phantom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, 0)
    
    cfg = astra.astra_dict('FP')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['VolumeDataId'] = phantom_id
    #cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)



    if geometry_type == 'parallel':
        cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)
    elif geometry_type == 'fanflat':
        cfg['ProjectorId'] = astra.create_projector('line_fanflat', proj_geom, vol_geom)
        
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    sinogram = astra.data2d.get(sinogram_id)

    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(phantom_id)
    astra.data2d.delete(sinogram_id)
    
    return sinogram


def SIRT(vol_geom, vol_data, sino_id, iters=2000, use_gpu=False):
    # create starting reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
    # define SIRT config params
    alg_cfg = astra.astra_dict('SIRT_CUDA' if use_gpu else 'SIRT')
    alg_cfg['ProjectionDataId'] = sino_id
    alg_cfg['ReconstructionDataId'] = rec_id
    alg_cfg['option'] = {}
    alg_cfg['option']['MinConstraint'] = 0
    alg_cfg['option']['MaxConstraint'] = 255
    # define algorithm
    alg_id = astra.algorithm.create(alg_cfg)
    # run the algorithm
    astra.algorithm.run(alg_id, iters)
    # create reconstruction data
    rec = astra.data2d.get(rec_id)

    return rec_id, rec

def reconstruct_sirt_astra(sinogram, proj_geom, vol_geom, num_iterations=10, mask=None, geometry_type='parallel'):
    
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    if geometry_type == 'parallel':
        projector_id = astra.create_projector('line', proj_geom, vol_geom)
    elif geometry_type == 'fanflat':
        projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")

    # Case 1: With mask
    if mask is not None:
        phantom_id = astra.data2d.create('-vol', vol_geom, mask.astype(np.float32))
        recon_id = astra.data2d.create('-vol', vol_geom, 0)

        cfg = astra.astra_dict('SIRT')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = projector_id
        cfg['option'] = {
            'ReconstructionMaskId': phantom_id,
            'MinConstraint': 0,
            'MaxConstraint': 255
        }

        alg_id = astra.algorithm.create(cfg)
        
        astra.algorithm.run(alg_id, num_iterations)

        reconstruction = astra.data2d.get(recon_id)

        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(phantom_id)
        astra.data2d.delete(recon_id)

    # Case 2: No mask
    else:
        
        recon_id = astra.data2d.create('-vol', vol_geom, 0)

        cfg = astra.astra_dict('SIRT')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = projector_id
        cfg['option'] = {
            'MinConstraint': 0,
            'MaxConstraint': 255
        }

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, num_iterations)

        reconstruction = astra.data2d.get(recon_id)

        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(recon_id)

    # Always clean up this one
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(projector_id)

    return reconstruction


def display_sinogram(sinogram):
    """Display the sinogram image."""
    plt.figure(figsize=(10, 5))
    plt.imshow(sinogram, cmap='gray', aspect='auto', 
               extent=[0, sinogram.shape[1], 180, 0])
    plt.xlabel('Detector Position')
    plt.ylabel('Projection Angle (degrees)')
    plt.title('Sinogram')
    plt.colorbar()
    plt.show()

def add_noise(sinogram, noise_type='poisson', noise_level=1.0):
    """
    Add noise to the sinogram.

    Parameters:
        sinogram: Input sinogram
        noise_type: Type of noise ('poisson' or 'gaussian')
        noise_level: Intensity of noise

    Returns:
        Noisy sinogram
    """
    if noise_type == 'poisson':
        max_val = np.max(sinogram)
        scaled_sino = sinogram / max_val * noise_level * 255
        noisy_sino = np.random.poisson(scaled_sino) * max_val / (noise_level * 255)
        return noisy_sino
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * np.mean(sinogram), sinogram.shape)
        return sinogram + noise
    else:
        return sinogram

def compute_rmse(original, reconstructed):
    """
    Compute Root Mean Square Error (RMSE) between the original and reconstructed images.
    """
    return np.sqrt(np.mean((original - reconstructed) ** 2))

def compute_rnmp(original, reconstructed):
    """
    Compute relative Normalized Mean Pixel error (rNMP) between the original and reconstructed images.
    """
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

def save_image(image, filepath):
    """
    Save an image to the specified filepath.

    Parameters:
        image: Input image
        filepath: Destination path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    imsave(filepath, image)

def reconstruct_fbp_astra(sinogram, proj_geom, vol_geom, num_iterations):
    """
    Perform FBP reconstruction using ASTRA.
    """
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    projector_id = astra.create_projector('line', proj_geom, vol_geom)  # Add projector ID
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = projector_id  # Include projector ID
    cfg['option'] = {'FilterType': 'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    reconstruction = astra.data2d.get(recon_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.projector.delete(projector_id)  # Clean up projector
    return reconstruction

def display_absolute_difference(image1, image2, title="Absolute Difference"):
    """
    Display the absolute difference between two images.

    Parameters:
        image1: First image (numpy array).
        image2: Second image (numpy array).
        title: Title for the plot.
    """
    abs_diff = np.abs(image1 - image2)
    plt.figure(figsize=(6, 6))
    plt.imshow(abs_diff, cmap='hot')
    plt.colorbar(label="Intensity Difference")
    plt.title(title)
    plt.axis("off")
    plt.show()