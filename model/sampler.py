
import torch
import numpy as np
from DDPM_model import ve_marginal_prob


# ======================
# Sampling Generating Function
# ======================

def ve_prior(shape, sigma_min=0.01, sigma_max=50, T=1.0):
    _, sigma_max_prior = ve_marginal_prob(None, T, sigma_min=sigma_min, sigma_max=sigma_max)
    return torch.randn(*shape) * sigma_max_prior

def ve_sde(t, sigma_min=0.01, sigma_max=50):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

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



def cond_pc_sampler(
        model, 
        cond,
        prior=ve_prior, # the noise distribution
        sde_coeff=ve_sde, # the forward SDE setting 
        snr=0.16,                
        device='cuda',
        num_steps=500,
        eps=1e-5,
        init_x=None,
    ):
    
    pose_dim = 6
    batch_size = cond.shape[0]
    init_x = prior((batch_size, pose_dim)).to(device) if init_x is None else init_x
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(pose_dim) 
    x = init_x
    poses = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            # data['sampled_pose'] = x
            # data['t'] = batch_time_step
            cond = cond.to(device)
            # import ipdb
            # ipdb.set_trace()
            grad = model(x, cond, batch_time_step)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  

            # # normalisation
            # if pose_mode == 'quat_wxyz' or pose_mode == 'quat_xyzw':
            #     # quat, should be normalised
            #     x[:, :4] /= torch.norm(x[:, :4], dim=-1, keepdim=True)   
            # elif pose_mode == 'euler_xyz':
            #     pass
            # else:
            # rotation(x axis, y axis), should be normalised
            x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
            x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)

            # Predictor step (Euler-Maruyama)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2*grad # R-SDE
            mean_x = x + drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            
            # normalisation
            # x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            # TODO: change the normalization part
            x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
            x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)
            poses.append(x.unsqueeze(0))
    
    xs = torch.cat(poses, dim=0) # (num_steps, bs, pose_dim)
    # xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    # mean_x[:, -3:] += data['pts_center']
    # mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    mean_x[:, :3] /= torch.norm(mean_x[:, :3], dim=-1, keepdim=True)
    mean_x[:, 3:6] /= torch.norm(mean_x[:, 3:6], dim=-1, keepdim=True)
    # The last step does not include any noise
    return xs.permute(1, 0, 2), mean_x # (bs, num_steps, pos_dim)