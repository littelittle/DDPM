import torch
import torch.nn as nn
import math
from model.pointnet import *
from model.pts_encoder.pointnet2 import *

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

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).squeeze(1)

# ======================
# Core Network Model
# ======================
class ConditionedDiffusionModel(nn.Module):
    """Conditional Diffusion Model Body"""
    def __init__(self, 
                 data_dim=6, 
                 data_emb_dim=256,
                 time_emb_dim=128,
                 cond_emb_dim=1024):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(embed_dim = time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(6, data_emb_dim),
            nn.ReLU(),
            nn.Linear(data_emb_dim, data_emb_dim),
            nn.ReLU()
        )
        
        self.cond_encoder1 = Pointnet2ClsMSG(0)

        # self.cond_encoder2 = Pointnet(out_feature_dim=cond_emb_dim)

        # self.fuse_cond = nn.Linear(256, 128)
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
        # set_trace()
        cond_emb = self.cond_encoder1(cond)  # [batch, cond_emb_dim]
        # cond_emb2 = self.cond_encoder2(cond)
        # cond_emb = F.relu(self.fuse_cond(torch.cat([cond_emb1, cond_emb2], dim=1)))
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

    # add grad clip 
    # torch.nn.utils.clip_grad_norm(
    #     model.parameters(),
    #     max_norm = 1
    # )
    
    optimizer.step()
    
    return loss.item()
    
# ======================
# Training VE SDE
# ======================

def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=50):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def train_ve_step(model, x0, cond, marginal_prob_func, device, optimizer, loss_fn, eps=1e-5):
    model.train()

    # transfer data
    batch_size = x0.shape[0]
    x0 = x0.to(device)
    cond = cond.to(device)

    t = torch.rand(batch_size, device=device)*(1.-eps) + eps # [bs,]
    t = t.unsqueeze(-1)
    mu, std = marginal_prob_func(x0, t)
    std = std.view(-1, 1)

    z = torch.rand_like(x0)
    preturbed_x = mu + z * std

    target_score = -z * std/(std ** 2)
    estimated_score = model(preturbed_x, cond, t)

    loss_weighting = std**2
    loss = torch.mean(torch.sum((loss_weighting*(estimated_score-target_score)**2).view(batch_size, -1), dim=-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

