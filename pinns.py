import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import astra
import os # For os.devnull if suppressing output

# Import from your custom files
import phantoms
import utils_astra

# --- Configuration ---
IMG_SIZE = 128  # Resolution for the phantom and reconstruction
PHANTOM_TYPE = "ct"  # Or "filled", "resolution", "basic"
NOISE_TYPE_PHANTOM = None # Or "gaussian", "poisson", "both" for phantom generation
SEED = 42

# ASTRA Geometry and Sinogram
# To get DETECTOR_COUNT = 4 with IMG_SIZE=128, detector_factor = 4 / 128 = 0.03125
# If IMG_SIZE changes, this factor needs to change to maintain 4 detectors
DETECTOR_COUNT_TARGET = 4
DETECTOR_FACTOR = 4



NUM_ANGLES = 100  # Number of projection angles
GEOMETRY_TYPE = 'parallel' # As per utils_astra.py default for basic projector

# SIRT Reconstruction
SIRT_ITERATIONS = 40

# PINN-like Reconstruction
PINN_ITERATIONS = 2000  # Needs more iterations than SIRT
PINN_LEARNING_RATE = 0.005 # Adjusted learning rate
USE_GPU_FOR_PINN = torch.cuda.is_available() # Use GPU for PINN if available

# Suppress ASTRA's verbose output if desired (optional)
# class SuppressOutput:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         self._original_stderr = sys.stderr
#         sys.stdout = open(os.devnull, 'w')
#         sys.stderr = open(os.devnull, 'w')
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout
#         sys.stderr.close()
#         sys.stderr = self._original_stderr
# import sys # needed for SuppressOutput

# --- 1. Phantom Generation ---
print("1. Generating custom phantom...")
phantom_image = phantoms.create_phantom(
    phantom_type=PHANTOM_TYPE,
    noise_type=NOISE_TYPE_PHANTOM,
    seed=SEED
)
# Ensure phantom is the desired IMG_SIZE (phantoms.py default is 512)
# if phantom_image.shape[0] != IMG_SIZE:
#     # Basic resize, for more complex phantoms, generate at target size
#     from skimage.transform import resize
#     phantom_image = resize(phantom_image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

phantom_image = phantom_image.astype(np.float32)
# Normalize to [0,1] if not already (phantoms.py seems to do this)
phantom_image = (phantom_image - np.min(phantom_image)) / (np.max(phantom_image) - np.min(phantom_image) + 1e-9)

print(f"Phantom generated with shape: {phantom_image.shape}, Type: {PHANTOM_TYPE}")
print(f"Phantom min: {phantom_image.min()}, max: {phantom_image.max()}")


# --- 2. ASTRA Geometry Setup and Sinogram Generation ---
print("\n2. Setting up ASTRA geometry and generating sinogram...")
proj_geom, vol_geom = utils_astra.create_astra_geometry(
    phantom_shape=phantom_image.shape,
    num_angles=NUM_ANGLES,
    detector_factor=DETECTOR_FACTOR, # This will lead to DETECTOR_COUNT_TARGET detectors
    geometry_type=GEOMETRY_TYPE
)
# Verify actual detector count
actual_det_count = proj_geom['DetectorCount']
print(f"ASTRA proj_geom created with DetectorCount: {actual_det_count} (target was {DETECTOR_COUNT_TARGET})")
if actual_det_count != DETECTOR_COUNT_TARGET:
    print(f"WARNING: Actual detector count {actual_det_count} differs from target {DETECTOR_COUNT_TARGET}. Adjust DETECTOR_FACTOR if needed.")


sino_data = utils_astra.generate_sinogram_astra(
    phantom=phantom_image,
    proj_geom=proj_geom,
    vol_geom=vol_geom,
    geometry_type=GEOMETRY_TYPE
)
sino_data = sino_data.astype(np.float32)
print(f"Sinogram generated with shape: {sino_data.shape}")


# --- 3. SIRT Reconstruction ---
print("\n3. Performing SIRT reconstruction...")
# utils_astra.reconstruct_sirt_astra has constraints [0, 255].
# If phantom is [0,1], this might scale the output. Let's see.
# For a fair comparison, ensure SIRT output is also in a similar range or normalize later.
recon_sirt = utils_astra.reconstruct_sirt_astra(
    sinogram=sino_data,
    proj_geom=proj_geom,
    vol_geom=vol_geom,
    num_iterations=SIRT_ITERATIONS,
    geometry_type=GEOMETRY_TYPE
)
recon_sirt = recon_sirt.astype(np.float32)
# Normalize SIRT output if its range is very different due to internal ASTRA scaling/constraints
recon_sirt_min, recon_sirt_max = recon_sirt.min(), recon_sirt.max()
if recon_sirt_max > 1.5 or recon_sirt_min < -0.5: # Heuristic for significant scaling
    print(f"Normalizing SIRT output from range [{recon_sirt_min:.2f}, {recon_sirt_max:.2f}]")
    recon_sirt = (recon_sirt - recon_sirt_min) / (recon_sirt_max - recon_sirt_min + 1e-9)
recon_sirt = np.clip(recon_sirt, 0, 1) # Clip to [0,1] for consistency

print(f"SIRT reconstruction complete after {SIRT_ITERATIONS} iterations.")
print(f"SIRT recon min: {recon_sirt.min()}, max: {recon_sirt.max()}")


# --- 4. PINN-like Reconstruction ---
print("\n4. Performing PINN-like reconstruction...")


device_for_torch = torch.device("cuda" if USE_GPU_FOR_PINN and torch.cuda.is_available() else "cpu")
print(f"PINN will run on: {device_for_torch}")

# Create ASTRA projector for PINN
pinn_projector_id = None
pinn_projector_is_gpu = False # Flag to track projector type

if device_for_torch.type == 'cuda':
    try:
        # For 2D data, 'cuda' projector type is usually for line-based projections.
        pinn_projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        pinn_projector_is_gpu = True
        print("Using ASTRA 'cuda' (GPU) projector for PINN.")
    except Exception as e_cuda:
        print(f"Failed to create ASTRA 'cuda' projector ({e_cuda}), falling back to CPU 'line' projector for PINN.")
        pinn_projector_id = astra.create_projector('line', proj_geom, vol_geom)
        pinn_projector_is_gpu = False
        device_for_torch = torch.device("cpu") # Ensure torch device matches if GPU projector fails
else:
    pinn_projector_id = astra.create_projector('line', proj_geom, vol_geom)
    pinn_projector_is_gpu = False
    print("Using ASTRA 'line' (CPU) projector for PINN.")

# Custom autograd function to wrap ASTRA forward and backprojection
class AstraProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_tensor, astra_projector_id_fwd): 
        ctx.astra_projector_id = astra_projector_id_fwd
      
        image_np = image_tensor.detach().cpu().numpy().astype(np.float32)
        image_np = np.ascontiguousarray(image_np) # Ensure C-contiguous

  
        sino_id, sino_np = astra.create_sino(image_np, astra_projector_id_fwd)
        
        if sino_id != -1: astra.data2d.delete(sino_id)
        if sino_np is None:
            raise RuntimeError("ASTRA forward projection failed to return sinogram data.")
        return torch.from_numpy(sino_np).to(image_tensor.device)

    @staticmethod
    def backward(ctx, grad_output_sino):
        astra_projector_id_bp = ctx.astra_projector_id
        grad_output_sino_np = grad_output_sino.detach().cpu().numpy().astype(np.float32)
        grad_output_sino_np = np.ascontiguousarray(grad_output_sino_np) # Ensure C-contiguous

      
        bp_id, grad_image_np = astra.create_backprojection(grad_output_sino_np, astra_projector_id_bp)

        if bp_id != -1: astra.data2d.delete(bp_id)
        if grad_image_np is None:
            raise RuntimeError("ASTRA backprojection failed to return gradient image data.")
        # The number of Nones must match the number of non-tensor inputs to forward
        # forward inputs: image_tensor, astra_projector_id_fwd
        # Only image_tensor is a tensor needing grad.
        return torch.from_numpy(grad_image_np).to(grad_output_sino.device), None

pinn_projector_id = None
if device_for_torch.type == 'cuda':
    try:
        pinn_projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        print("Using ASTRA 'cuda' (GPU) projector for PINN.")
    except Exception as e_cuda:
        print(f"Failed to create ASTRA 'cuda' projector ({e_cuda}), falling back to CPU 'line' projector for PINN.")
        pinn_projector_id = astra.create_projector('line', proj_geom, vol_geom)
        device_for_torch = torch.device("cpu")
else:
    pinn_projector_id = astra.create_projector('line', proj_geom, vol_geom)
    print("Using ASTRA 'line' (CPU) projector for PINN.")


# Initialize the image to be learned
recon_pinn_tensor = torch.zeros(phantom_image.shape, dtype=torch.float32, device=device_for_torch, requires_grad=True)
#recon_pinn_tensor = torch.rand(phantom_image.shape, dtype=torch.float32, device=device_for_torch, requires_grad=True) * 0.1
# Alternative initialization: from a blurred version or random

optimizer = optim.Adam([recon_pinn_tensor], lr=PINN_LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

sino_data_torch = torch.from_numpy(sino_data.copy()).to(device_for_torch)

print(f"Starting PINN optimization for {PINN_ITERATIONS} iterations...")
pinn_losses = []
#  inside the PINN training loop 
for i in range(PINN_ITERATIONS):
    optimizer.zero_grad()
    #current_estimate_pinn = torch.relu(recon_pinn_tensor)
    current_estimate_pinn = recon_pinn_tensor

    sino_estimate_torch = AstraProjectFunction.apply(current_estimate_pinn, pinn_projector_id)
    
    loss = loss_fn(sino_estimate_torch, sino_data_torch)
    pinn_losses.append(loss.item())

    loss.backward()
    optimizer.step()
# ...
recon_pinn = recon_pinn_tensor.detach().cpu().numpy()
# Final non-negativity and clipping if not done in loop or if relu wasn't enough
recon_pinn = np.clip(recon_pinn, 0, 1) # Clip to [0,1] to match original phantom scale

print("PINN-like reconstruction complete.")


# --- 5. Comparison and Plotting ---
print("\n5. Comparing results...")

# Normalize all images to [0,1] for consistent plotting, if not already
def normalize_for_plot(img):
    img_min, img_max = np.min(img), np.max(img)
    if img_max - img_min < 1e-9: return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

phantom_plot = normalize_for_plot(phantom_image.copy())
sirt_plot = normalize_for_plot(recon_sirt.copy())
pinn_plot = normalize_for_plot(recon_pinn.copy())

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.imshow(phantom_plot, cmap='gray', vmin=0, vmax=1)
plt.title(f'Original Phantom ({PHANTOM_TYPE}, {IMG_SIZE}x{IMG_SIZE})')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(sino_data, cmap='gray', aspect='auto')
plt.title(f'Sinogram ({actual_det_count} detectors)')
plt.xlabel('Detector Bin')
plt.ylabel('Projection Angle')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.plot(pinn_losses)
plt.title('PINN Training Loss (MSE)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)

mse_sirt = np.mean((phantom_plot - sirt_plot)**2)
plt.subplot(2, 3, 4)
plt.imshow(sirt_plot, cmap='gray', vmin=0, vmax=1)
plt.title(f'SIRT ({SIRT_ITERATIONS} iter, {actual_det_count} det)\nMSE: {mse_sirt:.2e}')
plt.colorbar()

mse_pinn = np.mean((phantom_plot - pinn_plot)**2)
plt.subplot(2, 3, 5)
plt.imshow(pinn_plot, cmap='gray', vmin=0, vmax=1)
plt.title(f'PINN-like ({PINN_ITERATIONS} iter, {actual_det_count} det)\nMSE: {mse_pinn:.2e}')
plt.colorbar()


# Sinogram from PINN reconstruction
sino_from_pinn_id, sino_from_pinn_np = (-1, np.array([]))
temp_recon_pinn_for_sino = recon_pinn.copy()
temp_recon_pinn_for_sino = np.ascontiguousarray(temp_recon_pinn_for_sino, dtype=np.float32)


sino_from_pinn_id, sino_from_pinn_np = astra.create_sino(temp_recon_pinn_for_sino, pinn_projector_id)


if sino_from_pinn_id != -1: astra.data2d.delete(sino_from_pinn_id)

plt.subplot(2, 3, 6)
if sino_from_pinn_np.size > 0:
    plt.imshow(sino_from_pinn_np, cmap='gray', aspect='auto')
    plt.title(f'Sinogram from PINN Recon ({actual_det_count} det)')
else:
    plt.text(0.5, 0.5, "Could not generate\nsinogram from PINN recon", ha='center', va='center')
plt.xlabel('Detector Bin')
plt.ylabel('Projection Angle')
plt.colorbar()

plt.tight_layout()
plt.show()

# --- 6. Final ASTRA Cleanup ---
print("Cleaning up ASTRA objects...")
if pinn_projector_id:
    astra.projector.delete(pinn_projector_id)
# Geometries (vol_geom, proj_geom) are usually managed by ASTRA or small enough not to worry,
# but explicit deletion is good practice if they were created with astra.data2d.create.
# Here they are Python objects returned by utils_astra and not directly ASTRA IDs.

print("Done.")