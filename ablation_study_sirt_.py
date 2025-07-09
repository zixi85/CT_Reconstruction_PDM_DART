import itertools
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

from phantoms import create_phantom
from utils_astra import generate_sinogram_astra, create_astra_geometry, display_sinogram, add_noise, compute_rmse, compute_rnmp, save_image, reconstruct_fbp_astra, reconstruct_sirt_astra 

def run_ablation():
    """
    Runs an ablation study for the PDM-DART algorithm by iterating through
    a grid of specified hyperparameters, performing reconstructions,
    and saving the results (including RMSE and RNMP) to a CSV file.
    """

    # Define the grid of hyperparameters to be tested.
    # Each key is a parameter name, and its value is a list of settings to try for that parameter.
    param_grid = {
        'num_pdm_iterations': [80],        # Number of outer PDM-DART iterations
        'sirt_iterations': [20, 40, 50],   # Number of internal SIRT iterations within each PDM-DART step
        'num_grey_levels': [5],            # Number of discrete grey levels to assume for the reconstruction
        'update_every': [5],               # Frequency (in PDM iterations) at which PDM parameters are re-optimized
      
        'phantom_type': ['ct'],            # Type of phantom to use (options might include 'layered', 'ct', 'resolution', 'filled')
        'noise_type': [None],              # Type of noise to add to the sinogram (None means no noise)
        'num_angles': [100],               # Number of projection angles for the sinogram
        'detector_size_factor': [4],       # Factor to scale the detector size relative to the phantom size
        #'opt_method': ["Nelder-Mead", "Powell", "COBYLA"] # Example of testing multiple optimization methods
        'opt_method': ["COBYLA"]           # Optimization method to use for PDM parameter tuning
    }

    # Generate all possible unique combinations of the hyperparameters from the grid.
    keys = list(param_grid.keys())
    # itertools.product generates the Cartesian product of the input iterables.
    combinations = list(itertools.product(*param_grid.values()))

    records = []  # List to store the results of each experimental run

    # Iterate through each combination of hyperparameters
    for combo in combinations:
        # Create a dictionary for the current combination of settings for easy access
        settings = dict(zip(keys, combo))

        print(f"Running: {settings}") # Log the current settings being tested
        try:
            # 1. Generate phantom based on current settings
            phantom = create_phantom(
                phantom_type=settings['phantom_type'],
                noise_type=settings['noise_type'], # This noise_type seems to be for phantom generation, not sinogram noise
                seed=42  # Seed for reproducibility of phantom generation
            )

            # 2. Create ASTRA projection and volume geometries
            proj_geom, vol_geom = create_astra_geometry(
                phantom.shape, 
                num_angles=settings['num_angles'], 
                detector_factor=settings['detector_size_factor']
            )

            # 3. Generate sinogram from the phantom using ASTRA
            sino = generate_sinogram_astra(phantom, proj_geom, vol_geom)

          

            # 4. Initialize the PDM-DART reconstructor with current settings
            reconstructor = PDMDARTAstra(
                sino, 
                phantom,  # Ground truth phantom, likely used for GMM initialization if PDM-DART uses it internally
                phantom.shape,
                num_angles=settings['num_angles'],
                num_grey_levels=settings['num_grey_levels'],
                detector_factor=settings['detector_size_factor'],
                opt_method=settings['opt_method']
            )

            # 5. Perform the PDM-DART reconstruction
            recon = reconstructor.reconstruct(
                num_iterations=settings['num_pdm_iterations'],      # Corresponds to N_DART
                sirt_iterations=settings['sirt_iterations'],        # Corresponds to t_s
                update_params_every=settings['update_every']        # Corresponds to uf
            )

            # 6. Compute evaluation metrics: Root Mean Square Error (RMSE) and Relative Number of Misclassified Pixels (RNMP)
            rmse = compute_rmse(phantom, recon)
            rnmp = compute_rnmp(phantom, recon) # Assuming RNMP requires a segmented version of recon or handles it internally

            # 7. Save the reconstructed image
            # The filename includes key parameters to identify the reconstruction easily.
            save_image(recon, f"results_ct_sirt/sirt_iterations_{settings['sirt_iterations']}_{settings['phantom_type']}.png")
            
            # 8. Store the current settings and the computed metrics
            record = settings.copy()  # Start with the current hyperparameter settings
            record['rmse'] = rmse     # Add RMSE to the record
            record['rnmp'] = rnmp     # Add RNMP to the record
            
            records.append(record)    # Add the complete record to the list

            print(f"Done: RMSE={rmse:.4f}, RNMP={rnmp:.4f}") # Log successful completion and key metrics

        except Exception as e:
            # Log any errors that occur during the process for a specific combination
            print(f"Failed for {settings}: {e}")

    # After all combinations have been processed, save the collected records to a CSV file.
    df = pd.DataFrame(records)
    df.to_csv('ablation_results_ct_final_.csv', index=False) # index=False prevents pandas from writing row indices to the CSV
    
    print("Ablation study complete. Results saved to ablation_results_ct_final_.csv")

if __name__ == '__main__':
    # This ensures that run_ablation() is called only when the script is executed directly
    # (not when imported as a module).
    run_ablation()
