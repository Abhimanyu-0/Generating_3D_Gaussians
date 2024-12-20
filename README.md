
# Gen_3DGS

## VQ-VAE


This Python script demonstrates the implementation of a Vector Quantized Variational Autoencoder (VQ-VAE) specifically tailored for learning and generating 3D Gaussian distributions. Here's what you'll find in this file:

- **Dataset Handling (`ShapeNetGaussianDataset`)**: 
  - Loads 3D Gaussian parameters from PLY files, which are commonly used to store 3D point data with additional attributes like position, scale, rotation, and opacity. 
  - Normalizes and reshapes the data into a consistent grid format, making it suitable for batch processing in deep learning models. 
  - Includes error handling and checks for data integrity, which is crucial when dealing with real-world datasets.

- **Model Architecture**:
  - **VQVAEGaussian3D**: A VQ-VAE model designed to encode and decode 3D Gaussian splats. 
    - The encoder uses 3D convolutions to understand the spatial structure of Gaussian distributions.
    - Vector Quantization (`VectorQuantizer`) is implemented to discretize the latent space, allowing the model to learn a compact and meaningful representation of 3D shapes.
    - The decoder reconstructs the Gaussian parameters from the quantized latent space.
  - **Normalization (`NormalizationLayer`)**: Ensures input data is within a manageable range for neural network layers.

- **Training and Generation**:
  - `train_vqvae`: A function to train the VQ-VAE model with detailed logging for each epoch and batch, including checks for numerical stability (NaN values).
  - `generate_samples`: Generates new 3D Gaussian samples from the trained model, with options to save outputs in both text and PLY formats for further analysis or visualization.

- **Utility Functions**:
  - `save_gaussians_to_ply`: Converts the generated Gaussian samples into PLY files, which can be viewed in various 3D visualization tools, aiding in understanding the 3D structure and attributes of the generated Gaussians.

This script is an excellent starting point if you're interested in:

- Understanding how 3D data can be represented and processed using Gaussian distributions.
- Learning about VQ-VAE for 3D data, which is a powerful technique for both compressing and generating complex 3D structures.
- Exploring data normalization and preparation for machine learning models in 3D contexts.
- Visualizing and working with 3D point clouds or splats, which are foundational in computer vision and graphics applications like scene reconstruction, VR/AR, and more.

**Usage**: 
- Ensure you have PyTorch, numpy, and plyfile installed. 
- Update the `directory` path in the configuration to point to your dataset of PLY files.
- Run the script to train the model and generate samples.
