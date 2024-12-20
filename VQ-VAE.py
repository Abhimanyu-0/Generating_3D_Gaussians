import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plyfile import PlyData, PlyElement
import os
from typing import Dict, Tuple, Optional

class ResBlock(nn.Module):
    """
    Residual block for 3D convolutional neural networks.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        h = self.conv1(self.norm1(x))
        h = F.gelu(h)  # Using GELU activation instead of ReLU for better convergence
        h = self.conv2(self.norm2(h))
        return x + h  # Residual connection

class NormalizationLayer(nn.Module):
    """
    Layer to normalize input to the range [-1, 1].
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Clamp to avoid extreme values, then scale to [-1, 1]
        x = torch.clamp(x, -10, 10)
        x = x / (torch.max(torch.abs(x)) + 1e-8)
        return x

class VectorQuantizer(nn.Module):
    """
    Implements vector quantization for discrete latent representation.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings with smaller range for stability
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, inputs):
        # Normalize inputs to ensure they are within a reasonable range
        inputs = torch.clamp(inputs, -1.0, 1.0)
        
        # Flatten input for distance calculation
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)
        
        # Compute Euclidean distance for quantization
        distances = torch.sum(flat_input ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight ** 2, dim=1) - \
                   2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # Quantize by finding the nearest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(inputs.shape)
        
        # Compute loss for quantization
        loss = F.mse_loss(quantized.detach(), inputs) + \
               self.commitment_cost * F.mse_loss(quantized, inputs.detach())
        
        # Straight-through estimator for gradient flow during backpropagation
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss

class VQVAEGaussian3D(nn.Module):
    """
    VQ-VAE model for encoding and decoding 3D Gaussian distributions.
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
        
        self.input_norm = NormalizationLayer()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(11, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            nn.Conv3d(64, embedding_dim, 3, padding=1),
            nn.GroupNorm(8, embedding_dim)
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            nn.Conv3d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            
            nn.Conv3d(32, 11, 3, padding=1)
        )
    
    def encode(self, x):
        z = self.encoder(x)
        z_q, _ = self.vq_layer(z)
        return z_q
    
    def decode(self, z_q):
        return self.decoder(z_q)
    
    def forward(self, x):
        x = self.input_norm(x)
        z = self.encoder(x)
        z_q, vq_loss = self.vq_layer(z)
        x_recon = self.decoder(z_q)

        # Split into components for post-processing
        positions, scales, rotations, opacity = torch.split(x_recon, [3, 3, 4, 1], dim=1)
        opacity = torch.sigmoid(opacity)  # Ensure opacity is in [0, 1]
        x_recon = torch.cat([positions, scales, rotations, opacity], dim=1)
        return x_recon, vq_loss

class ShapeNetGaussianDataset(Dataset):
    """
    Dataset for loading and processing Gaussian parameters from PLY files.
    """
    def __init__(self, directory: str, points_per_batch: int = 1000):
        super().__init__()
        
        self.ply_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                          if f.endswith('.ply')]
        
        if not self.ply_files:
            raise ValueError(f"No PLY files found in {directory}")
            
        print(f"Loading PLY files from {directory}")
        
        self.data = []
        for ply_file in self.ply_files:
            try:
                plydata = PlyData.read(ply_file)
                vertex = plydata['vertex']
                
                # Extract and process Gaussian parameters
                positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
                scales = np.stack([vertex[f'scale_{i}'] for i in range(3)], axis=-1)
                rotations = np.stack([vertex[f'rot_{i}'] for i in range(4)], axis=-1)
                opacities = vertex['opacity'].reshape(-1, 1)
                
                # Normalize scales in log space for stability
                scales = np.clip(scales, 1e-6, 1e6)
                scales = np.log(scales + 1e-8)
                scales = (scales - np.mean(scales, axis=0, keepdims=True)) / (np.std(scales, axis=0, keepdims=True) + 1e-8)
                
                # Normalize rotations
                rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
                
                # Clip opacities
                opacities = np.clip(opacities, 0, 1)
                
                gaussian_params = np.concatenate([positions, scales, rotations, opacities], axis=-1)
                
                # Check for non-finite values
                if not np.isfinite(gaussian_params).all():
                    print(f"Warning: Non-finite values found in {ply_file}, skipping")
                    continue
                
                # Reshape to fit a 3D grid
                grid_size = int(np.ceil(np.cbrt(points_per_batch)))
                pad_size = grid_size ** 3 - gaussian_params.shape[0]
                if pad_size > 0:
                    padding = np.zeros((pad_size, gaussian_params.shape[1]))
                    gaussian_params = np.concatenate([gaussian_params, padding], axis=0)
                
                gaussian_params = gaussian_params[:grid_size**3].reshape(grid_size, grid_size, grid_size, -1)
                gaussian_params = gaussian_params.transpose(3, 0, 1, 2)  # CDHW format
                
                # Final check for data integrity
                if np.isfinite(gaussian_params).all():
                    self.data.append(gaussian_params)
                    print(f"Successfully processed {ply_file}")
                else:
                    print(f"Warning: Non-finite values after processing {ply_file}, skipping")
                
            except Exception as e:
                print(f"Error processing {ply_file}: {str(e)}")
                continue

        if not self.data:
            raise ValueError("No valid data was loaded")
            
        self.data = np.stack(self.data, axis=0)
        print(f"Final dataset shape: {self.data.shape}")
        
        if not np.isfinite(self.data).all():
            raise ValueError("Non-finite values found in final dataset")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

def train_vqvae(model, dataloader, epochs, lr=1e-5, device='cuda'):
    """
    Training loop for the VQ-VAE model.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Normalize batch data
            batch_max = torch.max(torch.abs(batch))
            batch = batch / (batch_max + 1e-8)
            
            optimizer.zero_grad()
            
            z = model.encoder(batch)
            z_q, vq_loss = model.vq_layer(z)
            recon_batch = model.decoder(z_q)
            
            # Compute losses
            recon_loss = F.mse_loss(recon_batch, batch)
            loss = recon_loss + 0.1 * vq_loss
            
            # Check for NaN values to prevent training divergence
            if torch.isnan(loss):
                print("NaN detected! Breaking...")
                return
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1} complete")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    return model

def save_gaussians_to_ply(samples: torch.Tensor, filepath: str):
    """
    Convert generated Gaussian samples to PLY format with improved error handling and debugging.
    
    Parameters:
        samples (torch.Tensor): Generated samples tensor of shape [batch_size, 11, D, H, W]
        filepath (str): Path where to save the PLY file
    """
    # Move to CPU and convert to numpy
    samples_np = samples.cpu().numpy()
    
    # Print initial stats for debugging
    print("\nInitial sample statistics:")
    print(f"Sample shape: {samples_np.shape}")
    print(f"Sample range: {samples_np.min():.3f} to {samples_np.max():.3f}")
    print(f"Opacity channel range: {samples_np[:, 10].min():.3f} to {samples_np[:, 10].max():.3f}")
    
    # Process each sample
    for sample_idx, sample in enumerate(samples_np):
        # Extract parameters
        positions = sample[:3].reshape(-1, 3)  # xyz positions
        scales = sample[3:6].reshape(-1, 3)    # xyz scales
        rotations = sample[6:10].reshape(-1, 4) # quaternion rotation
        opacity = sample[10].reshape(-1, 1)    # opacity
        
        # Print pre-processing stats
        print(f"\nSample {sample_idx} pre-processing statistics:")
        print(f"Positions range: {positions.min():.3f} to {positions.max():.3f}")
        print(f"Scales range: {scales.min():.3f} to {scales.max():.3f}")
        print(f"Rotations range: {rotations.min():.3f} to {rotations.max():.3f}")
        print(f"Opacity range: {opacity.min():.3f} to {opacity.max():.3f}")
        
        # Normalize and scale data for better visualization
        positions = positions * 2.0  # Increase spatial spread
        scales = np.clip(scales, 0.001, 0.1)  # Reasonable scale values
        rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
        opacity = np.clip(opacity, 0.0, 1.0)  # Changed minimum to 0.0 for debugging
        
        # Print post-processing stats
        print(f"\nSample {sample_idx} post-processing statistics:")
        print(f"Positions range: {positions.min():.3f} to {positions.max():.3f}")
        print(f"Scales range: {scales.min():.3f} to {scales.max():.3f}")
        print(f"Rotations range: {rotations.min():.3f} to {rotations.max():.3f}")
        print(f"Opacity range: {opacity.min():.3f} to {opacity.max():.3f}")
        
        # Get points with significant opacity
        valid_mask = opacity.squeeze() > 0.1
        num_valid = np.sum(valid_mask)
        print(f"Number of valid points (opacity > 0.1): {num_valid}")
        
        if num_valid == 0:
            print("Warning: No valid points found. Lowering opacity threshold to 0.05")
            valid_mask = opacity.squeeze() > 0.05
            num_valid = np.sum(valid_mask)
            print(f"Number of valid points with lower threshold: {num_valid}")
            
            if num_valid == 0:
                print("Error: Still no valid points. Saving all points instead.")
                valid_mask = np.ones_like(opacity.squeeze(), dtype=bool)
                num_valid = np.sum(valid_mask)
        
        # Create color coefficients (default white)
        sh_coeffs = np.ones((positions.shape[0], 3), dtype=np.float32)
        
        # Create vertices array with proper dtype
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
        ]
        
        vertices = []
        for i in range(positions.shape[0]):
            if valid_mask[i]:
                vertex = (
                    float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]),
                    float(scales[i, 0]), float(scales[i, 1]), float(scales[i, 2]),
                    float(rotations[i, 0]), float(rotations[i, 1]), 
                    float(rotations[i, 2]), float(rotations[i, 3]),
                    float(opacity[i, 0]),
                    float(sh_coeffs[i, 0]), float(sh_coeffs[i, 1]), float(sh_coeffs[i, 2])
                )
                vertices.append(vertex)
        
        # Create PLY element
        vertex_array = np.array(vertices, dtype=vertex_dtype)
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        
        # Save PLY file
        output_path = f"{filepath[:-4]}_{sample_idx}.ply" if sample_idx > 0 else filepath
        PlyData([vertex_element], text=True).write(output_path)
        
        print(f"\nSaved PLY file: {output_path}")
        print(f"Number of saved points: {len(vertices)}")
        
        # Safe statistics calculation
        if len(vertices) > 0:
            vertex_array = np.array(vertices)
            print(f"Final position range: {vertex_array[:, :3].min():.3f} to {vertex_array[:, :3].max():.3f}")
            print(f"Final scale range: {vertex_array[:, 3:6].min():.3f} to {vertex_array[:, 3:6].max():.3f}")
            print(f"Final opacity range: {vertex_array[:, 10].min():.3f} to {vertex_array[:, 10].max():.3f}")


def generate_samples(model: VQVAEGaussian3D, 
                    num_samples: int = 1,
                    device: str = 'cuda',
                    save_path: Optional[str] = None,
                    save_ply: bool = True) -> torch.Tensor:
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Sample from learned distributions
        vq_embeddings = model.vq_layer.embedding.weight.data
        # Choose random indices for each sample
        indices = torch.randint(0, model.vq_layer.num_embeddings, (num_samples, 5, 5, 5), device=device)
        # Use these indices to select embeddings
        z_q = vq_embeddings[indices.view(-1)].view(num_samples, 64, 5, 5, 5)
        
        samples = model.decoder(z_q)
        
        # Save text format if requested
        if save_path:
            # Original text saving logic...
            with open(save_path, 'w') as f:
                samples_np = samples.cpu().numpy()
                for sample_idx in range(samples_np.shape[0]):
                    f.write(f"Sample {sample_idx+1}:\n")
                    for d in range(samples_np.shape[2]):
                        for h in range(samples_np.shape[3]):
                            for w in range(samples_np.shape[4]):
                                opacity = samples_np[sample_idx, 10, d, h, w]
                                if opacity > 0.1:
                                    pos = samples_np[sample_idx, :3, d, h, w]
                                    scales = samples_np[sample_idx, 3:6, d, h, w]
                                    rots = samples_np[sample_idx, 6:10, d, h, w]
                                    
                                    f.write(f"Point at ({d},{h},{w}):\n")
                                    f.write(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n")
                                    f.write(f"  Scales: [{scales[0]:.4f}, {scales[1]:.4f}, {scales[2]:.4f}]\n")
                                    f.write(f"  Rotation: [{rots[0]:.4f}, {rots[1]:.4f}, {rots[2]:.4f}, {rots[3]:.4f}]\n")
                                    f.write(f"  Opacity: {opacity:.4f}\n\n")
            
            print(f"Saved text samples to {save_path}")
        save_gaussians_to_ply(samples, save_path)
        # Save PLY files if requested
        """
        if save_ply:
            ply_base_path = save_path.rsplit('.', 1)[0] if save_path else "generated"
            save_as_ply(samples, ply_base_path)
        """
        return samples

if __name__ == "__main__":
    # Configuration
    config = {
        'directory': '/home/abhimanyu/Diffusion/bottles_data',  # Update this path
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'points_per_batch': 1000
    }
    
    # Initialize dataset and dataloader
    dataset = ShapeNetGaussianDataset(directory=config['directory'], 
                                    points_per_batch=config['points_per_batch'])
    
    dataloader = DataLoader(dataset, 
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=4)
    
    # Initialize model
    model = VQVAEGaussian3D(num_embeddings=512, embedding_dim=64)
    
    # Train model
    model = train_vqvae(model,
                       dataloader,
                       epochs=config['num_epochs'],
                       lr=config['learning_rate'],
                       device=config['device'])
    
    # Generate samples
    samples = generate_samples(model,
                             num_samples=100,
                             device=config['device'],
                             save_path='VQVAEgenerated_samples.txt')
    
    print("Training and generation completed!")
