import numpy as np
from plyfile import PlyData, PlyElement
import glob
import os

def combine_ply_files(input_directory: str, output_path: str, spacing: float = 4.0):
    """
    Combines multiple PLY files into a single PLY file, offsetting each model along the x-axis.
    
    Parameters:
        input_directory: Directory containing the PLY files to combine
        output_path: Path where to save the combined PLY file
        spacing: Distance between consecutive models along x-axis
    """
    # Get all PLY files in the directory
    ply_files = glob.glob(os.path.join(input_directory, "*.ply"))
    print(f"Found {len(ply_files)} PLY files")
    
    # List to store all vertices from all files
    all_vertices = []
    
    # Process each PLY file
    for file_idx, ply_file in enumerate(ply_files):
        try:
            # Read PLY file
            plydata = PlyData.read(ply_file)
            vertex = plydata['vertex']
            
            # Calculate offset for this model
            offset = np.array([0, 0, 0])
            
            # Get number of vertices in this file
            num_vertices = len(vertex)
            print(f"Processing {ply_file}: {num_vertices} vertices")
            
            # Process each vertex
            for i in range(num_vertices):
                # Get original position and apply offset
                pos = np.array([vertex['x'][i], vertex['y'][i], vertex['z'][i]]) + offset
                
                # Create vertex tuple with all attributes
                vertex_tuple = (
                    float(pos[0]), float(pos[1]), float(pos[2]),  # Offset position
                    float(vertex['scale_0'][i]), float(vertex['scale_1'][i]), 
                    float(vertex['scale_2'][i]),  # Scales
                    float(vertex['rot_0'][i]), float(vertex['rot_1'][i]),
                    float(vertex['rot_2'][i]), float(vertex['rot_3'][i]),  # Rotation
                    float(vertex['opacity'][i]),  # Opacity
                    # Color coefficients (assuming they exist, otherwise adjust accordingly)
                    float(vertex['f_dc_0'][i]) if 'f_dc_0' in vertex else 1.0,
                    float(vertex['f_dc_1'][i]) if 'f_dc_1' in vertex else 1.0,
                    float(vertex['f_dc_2'][i]) if 'f_dc_2' in vertex else 1.0
                )
                all_vertices.append(vertex_tuple)
                
        except Exception as e:
            print(f"Error processing {ply_file}: {str(e)}")
            continue
    
    # Define vertex structure
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    # Create combined vertex element
    vertex_array = np.array(all_vertices, dtype=vertex_dtype)
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    
    # Save combined PLY file
    PlyData([vertex_element], text=True).write(output_path)
    
    print(f"\nSuccessfully combined {len(ply_files)} files into {output_path}")
    print(f"Total number of vertices: {len(all_vertices)}")
    
    # Calculate and print statistics
    if len(all_vertices) > 0:
        vertex_array = np.array(all_vertices)
        print(f"Position range: {vertex_array[:, :3].min():.3f} to {vertex_array[:, :3].max():.3f}")
        print(f"Scale range: {vertex_array[:, 3:6].min():.3f} to {vertex_array[:, 3:6].max():.3f}")
        print(f"Opacity range: {vertex_array[:, 10].min():.3f} to {vertex_array[:, 10].max():.3f}")

# Usage example
if __name__ == "__main__":
    input_dir = "/home/abhimanyu/Diffusion/GEN_bags_VQVAE/"  # Directory containing your PLY files
    output_file = "combined_samples.ply"   # Where to save the combined file
    spacing = 4.0                          # Adjust this value to change spacing between models
    
    combine_ply_files(input_dir, output_file, spacing)
