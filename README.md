
# Gen_3DGS

## VQ-VAE


# Gen_3DGS

## VQ-VAE and VAE

This Python script demonstrates the implementation of both a Vector Quantized Variational Autoencoder (VQ-VAE) and a Variational Autoencoder (VAE) for learning and generating 3D Gaussian distributions. Below is an overview of the functionalities:

### Dataset Handling (`ShapeNetGaussianDataset`)
- Loads 3D Gaussian parameters from PLY files, which contain 3D point data including position, scale, rotation, and opacity.
- Normalizes and reshapes the data into a grid format for batch processing.
- Implements error handling and data integrity checks to manage real-world datasets.

### Model Architectures

#### **VQVAEGaussian3D**
- **VQ-VAE Model**:
  - **Encoder**: Uses 3D convolutions to encode spatial Gaussian distribution information into a latent space.
  - **Vector Quantization**: Discretizes the latent space with `VectorQuantizer`, allowing for learning discrete representations that can be useful for compression or generation tasks.
  - **Decoder**: Reconstructs 3D Gaussian parameters from the quantized latent space.
  - **Normalization**: Employed with `NormalizationLayer` to manage input data range for neural network stability.

#### **VAEGaussian3D**
- **VAE Model**:
  - **Encoder**: Similar to VQ-VAE, but outputs mean (`mu`) and log variance (`logvar`) for latent space sampling via the reparameterization trick.
  - **Reparameterization**: Allows for stochastic sampling in the latent space, which helps in generating new data points with inherent variability.
  - **Decoder**: Transforms latent samples back to the original space, reconstructing the 3D Gaussian parameters.
  - **Loss**: Incorporates both reconstruction loss (MSE) and KL divergence for regularization, balancing model's fit to data with the distribution of latent variables.

### Training and Generation

- **Training Functions**:
  - `train_vqvae`: Trains the VQ-VAE with detailed logging, ensuring numerical stability.
  - `train_vae`: Trains the VAE, managing reconstruction and KL divergence losses.

- **Generation**:
  - `generate_samples`: Available for both models, allowing you to generate new 3D Gaussian distributions. Options include saving in text or PLY format for visualization.

### Utility Functions

- `save_gaussians_to_ply`: Converts generated Gaussian samples into PLY files for 3D visualization, aiding in the analysis of the generated 3D structures.

### Usage
- Ensure you have PyTorch, numpy, and plyfile installed.
- Update the `directory` path in the configuration to point to your dataset of PLY files.
- Run the script to train either or both models and generate samples.

### Why Both Models?
- **VQ-VAE**: Offers discrete latent representations, which can be beneficial for tasks requiring clear separation of concepts or when aiming for data compression while maintaining generative capabilities.
- **VAE**: Provides a continuous latent space, which is excellent for capturing nuanced variations in data, offering flexibility in generative tasks where smooth transitions between different data points are desirable.

This script serves as an educational and practical resource for understanding how different autoencoding techniques can handle complex 3D data, particularly in applications like scene reconstruction, VR/AR, and beyond.
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
