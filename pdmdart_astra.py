import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
import astra
from utils_astra import reconstruct_sirt_astra, create_astra_geometry
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.ndimage import median_filter

class PDMDARTAstra:
    def __init__(self, sinogram,phantom_image, phantom_shape, num_angles=100, num_grey_levels=2, detector_factor=4, opt_method = "Nelder-Mead"):
        """
        Initialize the PDM-DART reconstructor (using ASTRA).
        
        Parameters:
            sinogram: Input sinogram data
            phantom_shape: Shape of the reconstructed image
            num_grey_levels: Number of grey levels (default 2: binary image)
        """
        self.sinogram = sinogram
        self.phantom_shape = phantom_shape
        self.num_grey_levels = num_grey_levels
        self.phantom_image = phantom_image
        self.num_angles, self.detector_size = sinogram.shape
        self.num_angles = num_angles
        print(sinogram.shape)
        
        # Initialize reconstructed image
        self.reconstruction = np.zeros(phantom_shape)
        self.opt_method = opt_method
        # Create ASTRA geometry
        self.proj_geom, self.vol_geom = create_astra_geometry(phantom_shape, self.num_angles, detector_factor = detector_factor)
        
        # Create projector
        self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
    


 

    def forward_project(self, image):
        """Perform forward projection using ASTRA."""
        # Create volume data object
        volume_id = astra.data2d.create('-vol', self.vol_geom, image)
        
        # Create sinogram storage
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)  # Only sinogram ID is returned
        
        # Configure the projection operator
        cfg = astra.astra_dict('FP')
        cfg['ProjectionDataId'] = sinogram_id
        cfg['VolumeDataId'] = volume_id
        cfg['ProjectorId'] = self.proj_id
        
        # Create and run the projection algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Retrieve sinogram data
        sinogram = astra.data2d.get(sinogram_id)  # Retrieve sinogram separately
        
        # Clean up ASTRA objects
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(volume_id)
        astra.data2d.delete(sinogram_id)
        
        return sinogram
    
    def reconstruct_sirt(self, sinogram, num_iterations=10, mask=None):
        """
        Perform SIRT reconstruction using ASTRA.
        
        Parameters:
            sinogram: Input sinogram
            num_iterations: Number of iterations
            mask: Update only the region specified by the mask
            
        Returns:
            Reconstructed image
        """
        return reconstruct_sirt_astra(sinogram, self.proj_geom, self.vol_geom, 
                                    num_iterations, mask)
    
    def segment_image(self, image):
        """Segment the image using current thresholds and grey levels."""
        segmented = np.zeros_like(image)
        
        # Below the first threshold
        mask = image < self.thresholds[0]
        segmented[mask] = self.grey_levels[0]
        
        # Middle regions
        for i in range(1, len(self.thresholds)):
            mask = (image >= self.thresholds[i-1]) & (image < self.thresholds[i])
            segmented[mask] = self.grey_levels[i]
        
        # Above the last threshold
        mask = image >= self.thresholds[-1]
        segmented[mask] = self.grey_levels[-1]
        
        return segmented
    
    def optimize_grey_levels(self, image, thresholds):
        """
        Optimize grey levels (inner optimization).
        
        Parameters:
            image: Current reconstructed image
            thresholds: Current thresholds
            
        Returns:
            Optimized grey levels
        """
        # Create segmentation masks
        masks = []
        masks.append(image < thresholds[0])
        for i in range(1, len(thresholds)):
            masks.append((image >= thresholds[i-1]) & (image < thresholds[i]))
        masks.append(image >= thresholds[-1])
        
        # Compute projection contribution for each partition
        A = np.zeros((self.sinogram.size, self.num_grey_levels))
        for i, mask in enumerate(masks):
            seg = np.zeros_like(image)
            seg[mask] = 1
            A[:, i] = self.forward_project(seg).flatten()
        
        # Solve linear system (Equation 19)
        Q = A.T @ A
        c = -2 * A.T @ self.sinogram.flatten()
        
        try:
            grey_levels = np.linalg.solve(2 * Q, -c)
            # Ensure grey levels are ordered
            grey_levels = np.sort(grey_levels)
            return grey_levels
        except np.linalg.LinAlgError:
            return self.grey_levels
    
    def optimize_thresholds(self, image, grey_levels):
        """
        Optimize thresholds (outer optimization).
        
        Parameters:
            image: Current reconstructed image
            grey_levels: Current grey levels
            
        Returns:
            Optimized thresholds
        """
        def projection_distance(t):
            # Segment the image
            segmented = self.segment_image_with_given_params(image, t, grey_levels)
            # Compute projection distance
            sino = self.forward_project(segmented)
            return np.linalg.norm(sino - self.sinogram)
        
        # Optimize using Nelder-Mead method
        res = minimize(projection_distance, self.thresholds, method=self.opt_method)
         
        return res.x
    
    def segment_image_with_given_params(self, image, thresholds, grey_levels):
        """Segment the image using given thresholds and grey levels."""
        segmented = np.zeros_like(image)
        
        mask = image < thresholds[0]
        segmented[mask] = grey_levels[0]
        
        for i in range(1, len(thresholds)):
            mask = (image >= thresholds[i-1]) & (image < thresholds[i])
            segmented[mask] = grey_levels[i]
        
        mask = image >= thresholds[-1]
        segmented[mask] = grey_levels[-1]
        
        return segmented




    def normalize_image(self,image, lower_percentile=1, upper_percentile=99):
        """Normalize image to [0, 1] based on percentile range."""
        lower = np.percentile(image, lower_percentile)
        upper = np.percentile(image, upper_percentile)
        return np.clip((image - lower) / (upper - lower), 0, 1)

    def estimate_grey_levels_and_thresholds_gmm(self, image, num_levels):
        """
        Estimate grey levels and thresholds using Gaussian Mixture Model.
    
        Parameters:
            image (ndarray): Input 2D image (e.g., from SIRT)
            num_levels (int): Number of grey levels/classes to estimate
    
        Returns:
            grey_levels (ndarray): Sorted grey level means
            thresholds (ndarray): Midpoints between sorted grey levels
        """
        # Flatten and remove extreme background if needed
        flat = image.flatten().reshape(-1, 1)
        
        
        mask = (flat > np.percentile(flat, 1)) & (flat < np.percentile(flat, 99))
        flat = flat[mask].reshape(-1, 1) 
    
        # Fit GMM
        gmm = GaussianMixture(n_components=num_levels, covariance_type='full', random_state=0)
        gmm.fit(flat)
        print(flat.shape)  # Should print: (n_pixels, 1)

    
        # Extract and sort the grey level means
        grey_levels = np.sort(gmm.means_.flatten())
    
        # Compute thresholds between grey levels
        thresholds = (grey_levels[:-1] + grey_levels[1:]) / 2
    
        return grey_levels, thresholds

    def get_boundary_pixels(self, segmented):
        """Get boundary pixels (pixels different from their neighbors)."""
        boundary = np.zeros_like(segmented, dtype=bool)
        
        # Check 4-neighborhood
        for i in range(1, segmented.shape[0]-1):
            for j in range(1, segmented.shape[1]-1):
                center = segmented[i, j]
                if (center != segmented[i-1, j] or center != segmented[i+1, j] or
                    center != segmented[i, j-1] or center != segmented[i, j+1]):
                    boundary[i, j] = True
        
        return boundary

 

    def check_early_convergence(self, reconstruction, previous_reconstruction, tol_image_change=0.0001):
        
        if np.linalg.norm(reconstruction) > 1e-9: # Avoid division by zero
            image_change = np.linalg.norm(reconstruction - previous_reconstruction) / np.linalg.norm(self.reconstruction)
            print(f"  Relative image change: {image_change:.2e}")
            if image_change < tol_image_change:
                print(f"Early stopping: Relative image change ({image_change:.2e}) below tolerance {tol_image_change} after {k+1} iterations.")
                return True
        
    def reconstruct(self, num_iterations=20, sirt_iterations=10, update_params_every=5, check_convergence=5):
        """
        Main PDM-DART reconstruction algorithm (using ASTRA).
        
        Parameters:
            num_iterations: Number of DART iterations
            sirt_iterations: Number of SIRT iterations per DART iteration
            update_params_every: Update parameters every N iterations
            
        Returns:
            Final reconstructed image
        """
        # Initial SIRT reconstruction
        self.reconstruction = self.reconstruct_sirt(self.sinogram, sirt_iterations)
       
        #self.reconstruction = self.normalize_image(self.reconstruction)
        #print("normalizing")
        try:
            self.grey_levels, self.thresholds = self.estimate_grey_levels_and_thresholds_gmm(self.reconstruction, self.num_grey_levels)
        except:
            min_val, max_val = np.min(self.phantom_image), np.max(phantom_image)
            if num_grey_levels > 1:
                padding = (max_val - min_val) * 0.05 
                self.thresholds = np.linspace(min_val + padding, max_val - padding, num_grey_levels - 1)
            else:
                self.thresholds = np.array([])
            self.grey_levels = np.linspace(min_val, max_val, num_grey_levels)
            
        previous_reconstruction = np.zeros_like(self.reconstruction) # For image change calculation

        
        for k in range(num_iterations):
            print(f"Iteration {k+1}/{num_iterations}")


            if k > 0: # No previous reconstruction for the first iteration
                 np.copyto(previous_reconstruction, self.reconstruction)

            
            
            # Update parameters every N iterations
            if k % update_params_every == 0:
                # Optimize grey levels
                self.grey_levels = self.optimize_grey_levels(self.reconstruction, self.thresholds)
                
                # Optimize thresholds
                self.thresholds = self.optimize_thresholds(self.reconstruction, self.grey_levels)
                print(f"Updated params - grey levels: {self.grey_levels}, thresholds: {self.thresholds}")
            
            # Segment the current reconstruction
            segmented = self.segment_image(self.reconstruction)
            
            # Determine the set of pixels to update (boundary + random pixels)
            boundary = self.get_boundary_pixels(segmented)
            random_pixels = np.random.random(self.phantom_shape) < 0.05  # 5% random pixels
            update_mask = boundary | random_pixels
            
            # Compute residual sinogram
            fixed_pixels = np.where(~update_mask, segmented, 0)
            residual_sino = self.sinogram - self.forward_project(fixed_pixels)
            
            # Reconstruct the residual (update only specified pixels)
            update_recon = self.reconstruct_sirt(residual_sino, sirt_iterations, update_mask)
            
            # Update the reconstructed image
            self.reconstruction = np.where(update_mask, update_recon, segmented)
            
            # Apply Gaussian smoothing
            self.reconstruction = gaussian_filter(self.reconstruction, sigma=0.5)


            # if (k + 1) % check_convergence == 0:
            #     if self.check_early_convergence(self.reconstruction, previous_reconstruction):
            #         break
            
        filtered_image = median_filter(self.reconstruction, size=5)
        # Clean up ASTRA projector
        astra.projector.delete(self.proj_id)

        
        return filtered_image
