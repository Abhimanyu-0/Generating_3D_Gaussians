
# Gen_3DGS
This repository serves as my ongoing exploration of different generative models. 
I have written the below so that it can be useful to someone new to these architectures and wants to start experimenting with them. 
Here is a paper that aims to explain the concepts succintly and also references various papers that have been useful 

[add link here]


## VQ-VAE and VAE


This repo demonstrates the implementation of both a Vector Quantized Variational Autoencoder (VQ-VAE) and a Variational Autoencoder (VAE) for learning and generating 3D Gaussian distributions. Below is an overview of the functionalities:

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
 
#### **TransformerDiffusion**
- **Diffusion Model with Transformer Architecture**:
  - **Gaussian Parameters Network**: Predicts position (xyz), scale, rotation (quaternions), and opacity
  - **Transformer Encoder**: Uses self-attention to capture relationships between Gaussians
  - **Time Embedding**: Sinusoidal embeddings for diffusion timesteps
  - **Parameter-Specific Processing**: 
    - Position: Direct prediction in xyz space
    - Scale: Positive values ensured through softplus activation
    - Rotation: Quaternion normalization for valid rotations
    - Opacity: Logits converted to probabilities through sigmoid

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

This repository is an excellent starting point if you're interested in:

- Understanding how 3D data can be represented and processed using Gaussian distributions.
- Learning about VQ-VAE and VAE for 3D data, which is a powerful technique for both compressing and generating complex 3D structures.
- Exploring data normalization and preparation for machine learning models in 3D contexts.
- Visualizing and working with 3D point clouds or splats, which are foundational in computer vision and graphics applications like scene reconstruction, VR/AR, and more.

**Usage**: 
- Ensure you have PyTorch, numpy, and plyfile installed. 
- Update the `directory` path in the configuration to point to your dataset of PLY files.
- Run the script to train the model and generate samples.

### Dataset
ShapeSplatsV1 - [https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1/tree/main](url)
