"""
3D Gaussian Generation using Diffusion Models
-------------------------------------------
This implementation focuses on generating 3D Gaussian primitives using a transformer-based 
diffusion model. The model learns to generate parameters for 3D Gaussians including position, 
scale, rotation (as quaternions), and opacity.

Key Components:
- TransformerDiffusion: Main model architecture using transformer layers
- GaussianDiffusion: Implementation of the diffusion process
- GaussianParameters: Network for predicting Gaussian parameters
- GaussianSubsetDataset: Dataset loader for PLY files containing Gaussian primitives

Author: Abhimanyu Suthar
License: MIT
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from plyfile import PlyData
from torch.distributions import Normal
from plyfile import PlyData, PlyElement

class GaussianParameters(nn.Module):

      """
    Neural network module for predicting parameters of 3D Gaussians.
    
    Outputs:
    - mean (xyz position): 3D vector
    - scale: 3D vector (positive values)
    - rotation: 4D quaternion (normalized)
    - opacity: scalar in [0,1]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.mean_net = nn.Linear(hidden_dim, 3).float()
        self.scale_net = nn.Linear(hidden_dim, 3).float()
        self.rotation_net = nn.Linear(hidden_dim, 4).float()
        self.opacity_net = nn.Linear(hidden_dim, 1).float()
        
    def forward(self, h):
        mean = self.mean_net(h)
        scale = F.softplus(self.scale_net(h)) + 1e-6
        rotation = F.normalize(self.rotation_net(h), dim=-1)
        opacity = self.opacity_net(h)
        
        return torch.cat([mean, scale, rotation, opacity], dim=-1)

class TransformerDiffusion(nn.Module):

      """
    Transformer-based architecture for diffusion model.
    
    The model processes a sequence of noisy Gaussian parameters and predicts the denoised parameters.
    Uses self-attention to capture relationships between different Gaussians in the scene.
    """

    def __init__(self, num_gaussians, hidden_dim=64, num_layers=6, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input embeddings with explicit float type
        self.input_norm = nn.BatchNorm1d(11).float()
        self.gaussian_embed = nn.Linear(11, hidden_dim).float()
        self.embed_norm = nn.BatchNorm1d(hidden_dim).float()
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim).float(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim).float()
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            batch_first=True,
            dropout=0.1
        ).float()
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).float()
        
        # Output
        self.gaussian_predictor = GaussianParameters(hidden_dim)
        
    def get_timestep_embedding(self, timesteps):
        timesteps = timesteps.float()
        half_dim = self.hidden_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant', value=0)
        return emb

    def forward(self, x, t):
        # Ensure input is float32
        x = x.float()
        t = t.float()
        
        batch_size, num_points, _ = x.shape
        x_flat = x.view(-1, x.shape[-1])
        x_norm = self.input_norm(x_flat).view(batch_size, num_points, -1)
        
        h = self.gaussian_embed(x_norm)
        h = self.embed_norm(h.view(-1, self.hidden_dim)).view(batch_size, num_points, -1)
        
        t_emb = self.time_embed(self.get_timestep_embedding(t))
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)
        pred = self.gaussian_predictor(h)
        return pred

class GaussianDiffusion:

      """
    Implements the diffusion process for 3D Gaussian parameters.
    
    Features:
    - Cosine noise schedule for stable training
    - Parameter-specific noise addition considering the manifold constraints
    - Separate loss terms for different parameter types
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-5, beta_end=0.01, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Use cosine schedule for betas
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - np.cos((t2) * np.pi / 2) / np.cos((t1) * np.pi / 2), 0.999))
        self.betas = torch.tensor(betas, device=device, dtype=torch.float32)
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32)
            
        # Split parameters and noise
        mean, scale, rot, opacity = torch.split(x_start, [3, 3, 4, 1], dim=-1)
        noise_mean, noise_scale, noise_rot, noise_opacity = torch.split(noise, [3, 3, 4, 1], dim=-1)
        
        # Sample timestep for each item in batch
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Add noise appropriately to each parameter
        noised_mean = sqrt_alpha * mean + sqrt_one_minus_alpha * noise_mean
        noised_scale = torch.exp(sqrt_alpha * torch.log(scale.clamp(min=1e-8)) + sqrt_one_minus_alpha * noise_scale)
        noised_rot = F.normalize(sqrt_alpha * rot + sqrt_one_minus_alpha * noise_rot, dim=-1)
        noised_opacity = sqrt_alpha * opacity + sqrt_one_minus_alpha * noise_opacity
        
        return torch.cat([noised_mean, noised_scale, noised_rot, noised_opacity], dim=-1)

    def p_losses(self, denoise_fn, x_start, t):
        noise = torch.randn_like(x_start, dtype=torch.float32)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = denoise_fn(x_noisy, t)
        
        # Split predictions and targets
        pred_mean, pred_scale, pred_rot, pred_opacity = torch.split(
            predicted, [3, 3, 4, 1], dim=-1
        )
        target_mean, target_scale, target_rot, target_opacity = torch.split(
            x_start, [3, 3, 4, 1], dim=-1
        )
        
        # Add epsilon to avoid log(0)
        eps = 1e-8
        pred_scale = pred_scale.clamp(min=eps)
        target_scale = target_scale.clamp(min=eps)
        
        # Calculate losses with stability improvements
        loss_mean = F.mse_loss(pred_mean, target_mean)
        loss_scale = F.mse_loss(torch.log(pred_scale), torch.log(target_scale))
        loss_rot = 1 - torch.sum(pred_rot * target_rot, dim=-1).mean()
        loss_opacity = F.binary_cross_entropy_with_logits(
            pred_opacity, 
            target_opacity,
            reduction='mean'
        )
        
        # Scale losses for better balance
        total_loss = (
            loss_mean * 0.1 + 
            loss_scale * 0.1 + 
            loss_rot * 0.1 + 
            loss_opacity * 0.01
        )
        
        # Check for NaN
        if torch.isnan(total_loss):
            print(f"NaN detected: mean={loss_mean}, scale={loss_scale}, rot={loss_rot}, opacity={loss_opacity}")
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)
            
        return total_loss

    def p_sample(self, model, x, t):
        t_tensor = torch.tensor([t], device=x.device, dtype=torch.float32)
        pred = model(x, t_tensor)
        
        alpha = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, dtype=torch.float32)
        sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha)) * torch.sqrt(1 - alpha/alpha_prev)
        
        pred_mean, pred_scale, pred_rot, pred_opacity = torch.split(
            pred, [3, 3, 4, 1], dim=-1
        )
        
        if t > 0:
            noise = torch.randn_like(x, dtype=torch.float32)
            noise_mean, noise_scale, noise_rot, noise_opacity = torch.split(
                noise, [3, 3, 4, 1], dim=-1
            )
            
            mean = pred_mean + sigma * noise_mean
            scale = torch.exp(torch.log(pred_scale) + sigma * noise_scale)
            rot = F.normalize(pred_rot + sigma * noise_rot, dim=-1)
            opacity = pred_opacity + sigma * noise_opacity
        else:
            mean = pred_mean
            scale = pred_scale
            rot = pred_rot
            opacity = pred_opacity
            
        return torch.cat([mean, scale, rot, opacity], dim=-1)
    
    @torch.no_grad()
    def sample(self, model, batch_size, num_gaussians, device="cuda"):
        x = torch.randn(batch_size, num_gaussians, 11, device=device, dtype=torch.float32)
        
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t)
        
        # Convert opacity logits to probabilities in final output
        mean, scale, rot, opacity = torch.split(x, [3, 3, 4, 1], dim=-1)
        opacity = torch.sigmoid(opacity)
        x = torch.cat([mean, scale, rot, opacity], dim=-1)
        
        return x

import os

class GaussianSubsetDataset(Dataset):

      """
    Dataset class for loading and processing PLY files containing 3D Gaussian parameters.
    
    Features:
    - Loads multiple PLY files from a directory
    - Normalizes rotation quaternions
    - Supports random sampling of points for batch creation
    """

    def __init__(self, directory, points_per_batch=100, num_batches=1000):
        # Collect all PLY files from the specified directory
        ply_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ply')]
        
        if not ply_files:
            raise ValueError("No PLY files found in the specified directory.")
        
        all_centroids = []
        all_opacity = []
        all_scales = []
        all_rotations = []
        
        print(f"Loading PLY files from {directory}")
        
        for ply_file in ply_files:
            print(f"Loading {ply_file}")
            gs_vertex = PlyData.read(ply_file)['vertex']
            
            # Extract positions
            x = gs_vertex['x'].astype(np.float32)
            y = gs_vertex['y'].astype(np.float32)
            z = gs_vertex['z'].astype(np.float32)
            centroids = np.stack((x, y, z), axis=-1)
            all_centroids.append(centroids)
            
            # Extract opacity
            opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)
            all_opacity.append(opacity)
            
            # Extract scales
            scale_names = sorted([p.name for p in gs_vertex.properties if p.name.startswith("scale_")],
                                 key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((len(x), len(scale_names)), dtype=np.float32)
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = gs_vertex[attr_name].astype(np.float32)
            all_scales.append(scales)
            
            # Extract rotations
            rot_names = sorted([p.name for p in gs_vertex.properties if p.name.startswith("rot")],
                             key=lambda x: int(x.split("_")[-1]))
            rotations = np.zeros((len(x), len(rot_names)), dtype=np.float32)
            for idx, attr_name in enumerate(rot_names):
                rotations[:, idx] = gs_vertex[attr_name].astype(np.float32)
            all_rotations.append(rotations)
            
            print(f"Loaded {len(x)} points from {ply_file}")

        # Stack all data together
        self.centroids = np.concatenate(all_centroids, axis=0)
        self.opacity = np.concatenate(all_opacity, axis=0)
        self.scales = np.concatenate(all_scales, axis=0)
        self.rotations = np.concatenate(all_rotations, axis=0)
        
        # Normalize rotations for the entire dataset
        self.rotations = self.rotations / np.linalg.norm(self.rotations, axis=1, keepdims=True)
        
        self.points_per_batch = points_per_batch
        self.num_batches = num_batches
        self.total_points = len(self.centroids)
        
        print(f"Total points loaded: {self.total_points}")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        indices = np.random.choice(self.total_points, self.points_per_batch, replace=False)
        sample = np.concatenate([
            self.centroids[indices],
            self.scales[indices],
            self.rotations[indices],
            self.opacity[indices]
        ], axis=-1)
        return torch.from_numpy(sample).float()

def train_diffusion_model(
    directory,
    points_per_batch=100,
    batch_size=16,
    num_epochs=20,
    batches_per_epoch=1000,
    device="cuda",
    learning_rate=1e-5
):

      """
    Trains the diffusion model on 3D Gaussian data.
    
    Features:
    - Gradient clipping for stability
    - Learning rate scheduling
    - Loss monitoring and early stopping
    """
    dataset = GaussianSubsetDataset(
        directory, 
        points_per_batch=points_per_batch,
        num_batches=batches_per_epoch
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = TransformerDiffusion(num_gaussians=points_per_batch).to(device).float()
    diffusion = GaussianDiffusion(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # purely training 
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        model.train()
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps for each item in the batch
            t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=device)
            
            # Calculate loss
            loss = diffusion.p_losses(model, batch, t)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
        else:
            print(f"Epoch {epoch}: No valid batches")

    return model
"""
def generate_gaussians(model, diffusion, batch_size=1, num_gaussians=100, device="cuda"):
    model.eval()
    gaussians = diffusion.sample(model, batch_size, num_gaussians, device)
    
    mean, scale, rotation, opacity = torch.split(
        gaussians, [3, 3, 4, 1], dim=-1
    )
    results = {
        'mean': mean.cpu().numpy(),
        'scale': scale.cpu().numpy(),
        'rotation': rotation.cpu().numpy(),
        'opacity': opacity.cpu().numpy()
    }
    
    # Save to a .npz file for easy loading later
    np.savez('generated_gaussians.npz', **results)
    return {
        'mean': mean,
        'scale': scale,
        'rotation': rotation,
        'opacity': opacity
    }

    """
"""
def generate_gaussians(model, diffusion, batch_size=1, num_gaussians=100, device="cuda"):
    model.eval()
    gaussians = diffusion.sample(model, batch_size, num_gaussians, device)
    
    mean, scale, rotation, opacity = torch.split(
        gaussians, [3, 3, 4, 1], dim=-1
    )
    
    mean = mean.cpu().numpy()
    scale = scale.cpu().numpy()
    rotation = rotation.cpu().numpy()
    opacity = opacity.cpu().numpy()

    # Define the structure for the PLY file
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
                    ('opacity', 'f4')]

    # Combine all data into a single array for the PLY file
    vertices = []
    for i in range(mean.shape[0]):
        vertex = (
            mean[i, 0], mean[i, 1], mean[i, 2],
            scale[i, 0], scale[i, 1], scale[i, 2],
            rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3],
            opacity[i, 0]
        )
        vertices.append(vertex)

    # Create the vertex element
    vertex_element = PlyElement.describe(np.array(vertices, dtype=vertex_dtype), 'vertex')

    # Write to PLY
    PlyData([vertex_element], text=True).write('generated_gaussians.ply')
    
    print("\nGenerated Gaussian parameters saved to 'generated_gaussians.ply':")
    print(f"Number of Gaussians: {mean.shape[0]}")

    return {
        'mean': mean,
        'scale': scale,
        'rotation': rotation,
        'opacity': opacity
    }
"""
def generate_gaussians(model, diffusion, batch_size=1, num_gaussians=100, device="cuda"):
        """
    Generates new 3D Gaussian parameters using the trained model.
    
    Features:
    - Post-processes generated parameters for physical validity
    - Saves results in PLY format compatible with 3D visualization tools
    - Includes additional attributes like spherical harmonics coefficients
    """

    model.eval()
    gaussians = diffusion.sample(model, batch_size, num_gaussians, device)
    
    mean, scale, rotation, opacity = torch.split(
        gaussians, [3, 3, 4, 1], dim=-1
    )
    
    # Convert to numpy and reshape
    mean = mean.cpu().numpy().reshape(-1, 3)
    scale = scale.cpu().numpy().reshape(-1, 3)
    rotation = rotation.cpu().numpy().reshape(-1, 4)
    opacity = opacity.cpu().numpy().reshape(-1, 1)

    # Normalize and scale the data for better visualization
    # Spread out the points more
    mean = mean * 2.0  # Increase spatial spread
    
    # Ensure reasonable scale values (between 0.001 and 0.1)
    scale = np.clip(scale, 0.001, 0.1)
    
    # Normalize rotations
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True)
    
    # Convert opacity to reasonable values (between 0.1 and 1.0)
    opacity = np.clip(opacity, 0.1, 1.0)

    # Add spherical harmonics coefficients (RGB colors)
    sh_coeffs = np.ones((mean.shape[0], 3), dtype=np.float32)  # RGB values
    
    # Define the vertex structure for PLY
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')  # RGB coefficients
    ]

    # Create vertices array
    vertices = []
    for i in range(mean.shape[0]):
        vertex = (
            float(mean[i, 0]), float(mean[i, 1]), float(mean[i, 2]),
            float(scale[i, 0]), float(scale[i, 1]), float(scale[i, 2]),
            float(rotation[i, 0]), float(rotation[i, 1]), 
            float(rotation[i, 2]), float(rotation[i, 3]),
            float(opacity[i, 0]),
            float(sh_coeffs[i, 0]), float(sh_coeffs[i, 1]), float(sh_coeffs[i, 2])
        )
        vertices.append(vertex)

    # Create PLY element and save
    vertex_element = PlyElement.describe(np.array(vertices, dtype=vertex_dtype), 'vertex')
    output_path = 'adjusted_generated_gaussians.ply'
    PlyData([vertex_element], text=True).write(output_path)
    
    print(f"\nGenerated Gaussian parameters saved to '{output_path}'")
    print(f"Number of Gaussians: {mean.shape[0]}")
    print(f"Mean range: {mean.min():.3f} to {mean.max():.3f}")
    print(f"Scale range: {scale.min():.3f} to {scale.max():.3f}")
    print(f"Opacity range: {opacity.min():.3f} to {opacity.max():.3f}")

    return {
        'mean': mean,
        'scale': scale,
        'rotation': rotation,
        'opacity': opacity
    }

if __name__ == "__main__":
    """
    Main training and generation pipeline.
    
    Usage:
    1. Set configuration parameters
    2. Train the model
    3. Save the trained model
    4. Generate new Gaussian parameters
    """
    config = {
        'directory': '/home/abhimanyu/Diffusion/02773838',  # Replace with your directory path
        'points_per_batch': 20000,
        'batch_size': 32,
        'num_epochs': 30,
        'batches_per_epoch': 10,
        'learning_rate': 1e-2,
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print("Starting training...")
    print(f"Using device: {config['device']}")
    
    try:
        model = train_diffusion_model(**config)
        
        # Save the trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, "gaussian_diffusion_model_directory.pt")
        
        print("\nTraining completed. Model saved.")
        
        # Generate new Gaussians
        print("\nGenerating new Gaussians...")
        diffusion = GaussianDiffusion(device=config['device'])
        generated = generate_gaussians(
            model,
            diffusion,
            batch_size=1,
            num_gaussians=1,
            device=config['device']
        )
        
        print("\nGenerated Gaussian parameters:")
        for key, value in generated.items():
            print(f"{key}: {value.shape}")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise
