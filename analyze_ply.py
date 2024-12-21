from plyfile import PlyData
import numpy as np

from plyfile import PlyData, PlyElement

gs_vertex = PlyData.read('/home/abhimanyu/Diffusion/GEN_bags_VAE/combined_samples.ply')['vertex']
### load centroids[x,y,z] - Gaussian centroid
x = gs_vertex['x'].astype(np.float32)
y = gs_vertex['y'].astype(np.float32)
z = gs_vertex['z'].astype(np.float32)
centroids = np.stack((x, y, z), axis=-1) # [n, 3]

### load o - opacity
opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)


### load scales[sx, sy, sz] - Scale
scale_names = [
    p.name
    for p in gs_vertex.properties
    if p.name.startswith("scale_")
]
scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
scales = np.zeros((centroids.shape[0], len(scale_names)))
for idx, attr_name in enumerate(scale_names):
    scales[:, idx] = gs_vertex[attr_name].astype(np.float32)

### load rotation rots[q_0, q_1, q_2, q_3] - Rotation
rot_names = [
    p.name for p in gs_vertex.properties if p.name.startswith("rot")
]
rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
rots = np.zeros((centroids.shape[0], len(rot_names)))
for idx, attr_name in enumerate(rot_names):
    rots[:, idx] = gs_vertex[attr_name].astype(np.float32)

rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)

### load base sh_base[dc_0, dc_1, dc_2] - Spherical harmonic
sh_base = np.zeros((centroids.shape[0], 3, 1))
sh_base[:, 0, 0] = gs_vertex['f_dc_0'].astype(np.float32)
sh_base[:, 1, 0] = gs_vertex['f_dc_1'].astype(np.float32)
sh_base[:, 2, 0] = gs_vertex['f_dc_2'].astype(np.float32)
sh_base = sh_base.reshape(-1, 3)

import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt

def analyze_gaussian_splats(
    centroids: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    sh_base: np.ndarray
) -> Dict:
    """
    Analyzes Gaussian Splat data and provides statistical insights.
    
    Parameters:
    -----------
    centroids: np.ndarray
        Shape (N, 3) array of x,y,z coordinates
    opacity: np.ndarray
        Shape (N, 1) array of opacity values
    scales: np.ndarray
        Shape (N, 3) array of scale values for each axis
    rotations: np.ndarray
        Shape (N, 4) array of quaternion rotations
    sh_base: np.ndarray
        Shape (N, 3) array of spherical harmonic base coefficients
    
    Returns:
    --------
    Dict containing statistical analysis results
    """
    n_gaussians = centroids.shape[0]
    
    # Spatial distribution analysis
    spatial_stats = {
        'bounds': {
            'min': centroids.min(axis=0),
            'max': centroids.max(axis=0),
            'range': centroids.max(axis=0) - centroids.min(axis=0)
        },
        'mean_position': centroids.mean(axis=0),
        'std_position': centroids.std(axis=0)
    }
    
    # Opacity analysis
    opacity_stats = {
        'mean': opacity.mean(),
        'std': opacity.std(),
        'min': opacity.min(),
        'max': opacity.max(),
        'histogram': np.histogram(opacity, bins=20)[0]
    }
    
    # Scale analysis
    scale_stats = {
        'mean_scales': scales.mean(axis=0),
        'std_scales': scales.std(axis=0),
        'min_scales': scales.min(axis=0),
        'max_scales': scales.max(axis=0),
        'aspect_ratios': scales.max(axis=1) / (scales.min(axis=1) + 1e-6)
    }
    
    # Rotation analysis
    rotation_stats = {
        'mean_magnitude': np.linalg.norm(rotations, axis=1).mean(),
        'std_magnitude': np.linalg.norm(rotations, axis=1).std(),
        'quaternion_stats': {
            'mean': rotations.mean(axis=0),
            'std': rotations.std(axis=0)
        }
    }
    
    # Spherical harmonics analysis
    sh_stats = {
        'mean_coefficients': sh_base.mean(axis=0),
        'std_coefficients': sh_base.std(axis=0),
        'color_range': {
            'min': sh_base.min(axis=0),
            'max': sh_base.max(axis=0)
        }
    }
    
    def compute_density(points: np.ndarray, resolution: int = 50) -> np.ndarray:
        """Compute point density in 3D space"""
        H, edges = np.histogramdd(points, bins=resolution)
        return H
    
    # Compute point density
    density = compute_density(centroids)
    
    return {
        'n_gaussians': n_gaussians,
        'spatial_stats': spatial_stats,
        'opacity_stats': opacity_stats,
        'scale_stats': scale_stats,
        'rotation_stats': rotation_stats,
        'sh_stats': sh_stats,
        'density': density
    }

def visualize_analysis(analysis_results: Dict) -> None:
    """
    Creates visualizations for the analysis results.
    
    Parameters:
    -----------
    analysis_results: Dict
        Results from analyze_gaussian_splats function
    """
    plt.figure(figsize=(15, 10))
    
    # Plot opacity histogram
    plt.subplot(231)
    plt.bar(range(len(analysis_results['opacity_stats']['histogram'])), 
            analysis_results['opacity_stats']['histogram'])
    plt.title('Opacity Distribution')
    
    # Plot scale distributions
    plt.subplot(232)
    plt.boxplot([
        analysis_results['scale_stats']['aspect_ratios']
    ])
    plt.title('Scale Aspect Ratios')
    
    # Plot density slice
    plt.subplot(233)
    plt.imshow(analysis_results['density'].sum(axis=2))
    plt.title('Point Density (XY Projection)')
    
    plt.tight_layout()
    plt.show()

# Example usage with your data:
analysis = analyze_gaussian_splats(
    centroids=centroids,
    opacity=opacity,
    scales=scales,
    rotations=rots,
    sh_base=sh_base
)
visualize_analysis(analysis)

def print_gaussian_splat_data(centroids, opacity, scales, rots, sh_base):
    """
    Prints a formatted summary of Gaussian Splat data
    
    Parameters:
    -----------
    centroids: np.ndarray - (N, 3) array of x,y,z coordinates
    opacity: np.ndarray - (N, 1) array of opacity values
    scales: np.ndarray - (N, 3) array of scale values
    rots: np.ndarray - (N, 4) array of quaternion rotations
    sh_base: np.ndarray - (N, 3) array of spherical harmonic bases
    """
    num_gaussians = len(centroids)
    
    print("=== Gaussian Splat Data Summary ===\n")
    print(f"Total number of Gaussians: {num_gaussians}\n")
    
    print("=== Centroids Statistics ===")
    print(f"Shape: {centroids.shape}")
    print(f"Min XYZ: {centroids.min(axis=0)}")
    print(f"Max XYZ: {centroids.max(axis=0)}")
    print(f"Mean XYZ: {centroids.mean(axis=0)}")
    print("\nSample of first 5 centroids:")
    print(centroids[:5])
    print()
    
    print("=== Opacity Statistics ===")
    print(f"Shape: {opacity.shape}")
    print(f"Min: {opacity.min()}")
    print(f"Max: {opacity.max()}")
    print(f"Mean: {opacity.mean()}")
    print("\nSample of first 5 opacity values:")
    print(opacity[:5])
    print()
    
    print("=== Scales Statistics ===")
    print(f"Shape: {scales.shape}")
    print(f"Min scales: {scales.min(axis=0)}")
    print(f"Max scales: {scales.max(axis=0)}")
    print(f"Mean scales: {scales.mean(axis=0)}")
    print("\nSample of first 5 scales:")
    print(scales[:5])
    print()
    
    print("=== Rotation Quaternions Statistics ===")
    print(f"Shape: {rots.shape}")
    print(f"Min quaternions: {rots.min(axis=0)}")
    print(f"Max quaternions: {rots.max(axis=0)}")
    print(f"Mean quaternions: {rots.mean(axis=0)}")
    print("\nSample of first 5 quaternions:")
    print(rots[:5])
    print()
    
    print("=== Spherical Harmonics Base Statistics ===")
    print(f"Shape: {sh_base.shape}")
    print(f"Min SH coefficients: {sh_base.min(axis=0)}")
    print(f"Max SH coefficients: {sh_base.max(axis=0)}")
    print(f"Mean SH coefficients: {sh_base.mean(axis=0)}")
    print("\nSample of first 5 SH bases:")
    print(sh_base[:5])
    
    # Print memory usage
    total_bytes = (
        centroids.nbytes + 
        opacity.nbytes + 
        scales.nbytes + 
        rots.nbytes + 
        sh_base.nbytes
    )
    print(f"\nTotal memory usage: {total_bytes / 1024 / 1024:.2f} MB")

# Print the data
print_gaussian_splat_data(centroids, opacity, scales, rots, sh_base)
