import torch
import torch.nn as nn
import math
from model.pointnet import *

# ======================
# Time step related parameter planning module
# ======================
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta scheduler, returns beta values for all time steps"""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alphas(betas, device):
    """Calculate alpha and cumulative alpha product based on beta"""
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas.to(device), alphas_cumprod.to(device)

# ======================
# Time step encoding module
# ======================
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position encoder, mapping time step t to a high-dimensional vector"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ======================
# Core Network Model
# ======================
class ConditionedDiffusionModel(nn.Module):
    """Conditional Diffusion Model Body"""
    def __init__(self, 
                 data_dim=6, 
                 data_emb_dim=32,
                 time_emb_dim=32,
                 cond_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(6, data_emb_dim),
            nn.ReLU(True),
            nn.Linear(data_emb_dim, data_emb_dim),
            nn.ReLU(True)
        )
        
        self.cond_encoder1 = Pointnet(out_feature_dim=cond_emb_dim)

        self.cond_encoder2 = Pointnet(out_feature_dim=cond_emb_dim)

        self.fuse_cond = nn.Linear(256, 128)
        
        self.main = nn.Sequential(
            nn.Linear(data_emb_dim + time_emb_dim + cond_emb_dim, 256),
            nn.Mish(),
            nn.Linear(256, data_dim)
        )
        
    def forward(self, x, cond, t):
        # Input Dimension Validation
        # x: [batch, data_dim]
        # cond: [batch, n, 3]
        # t: [batch]
        
        # Get the embedding of each component
        t_emb = self.time_mlp(t)  # [batch, time_emb_dim]
        cond_emb1 = self.cond_encoder1(cond)  # [batch, cond_emb_dim]
        cond_emb2 = self.cond_encoder2(cond)
        cond_emb = F.relu(self.fuse_cond(torch.cat([cond_emb1, cond_emb2], dim=1)))
        x = self.pose_encoder(x)
        
        # concatinate all the features
        combined = torch.cat([x, t_emb, cond_emb], dim=1)
        
        # predict all the noise
        return self.main(combined)

# ======================
# Training process function
# ======================
def train_step(model, x0, cond, alphas_cumprod, device, optimizer, loss_fn):
    """
    Single training step
    Params:
        model: DDPM
        x0: rotation [batch, 6]
        cond: partial points [batch, num_points, 3]
        alphas_cumprod: precomputed alpha cumulative product
        device: compute device
        optimizer: Adam
        loss_fn: MSE_loss
    """
    model.train()
    
    # transfer data
    batch_size = x0.shape[0]
    x0 = x0.to(device)
    cond = cond.to(device)
    
    # Randomly sample time steps
    t = torch.randint(0, len(alphas_cumprod), (batch_size,), device=device).long()
    
    # compute alpha_bar
    alpha_bar = alphas_cumprod[t].unsqueeze(-1)  # [batch, 1]
    
    # generate the noise
    eps = torch.randn_like(x0)  # true noise
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
    
    # predict the noise
    pred_eps = model(xt, cond, t)
    
    # compute loss
    loss = loss_fn(pred_eps, eps)
    
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# ======================
# Sampling Generating Function
# ======================
@torch.no_grad()
def sample(model, cond, alphas, alphas_cumprod, device, num_steps=1000):
    """
    Sampling Generation Process
    Params:
        model: trained model
        cond: conditional data (partial points) [1, num_points, 3]
        alphas: Precomputed alpha values
        alphas_cumprod: Precomputed alpha cumulative product
        device: compute device
        num_steps: total steps
    """
    model.eval()
    
    # Initialize pure noise
    x = torch.randn(1, model.main[-1].out_features).to(device)
    cond = cond.unsqueeze(0).to(device)
    
    # Stepwise Denoising
    for t in reversed(range(num_steps)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        pred_eps = model(x, cond, t_batch)
        
        # Calculate alpha related parameters
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        # Calculate the reverse process parameters
        beta_tilde_t = (1 - alpha_bar_t_prev)/(1 - alpha_bar_t) * (1 - alpha_t)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
        
        # Reverse process sampling
        mean = (x - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x = mean + torch.sqrt(beta_tilde_t) * noise
    
    return pred_x0.squeeze(0)