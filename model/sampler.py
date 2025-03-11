
import torch
import numpy as np
from scipy import integrate


# ======================
# Sampling Generating Function
# ======================
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=50):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_prior(shape, sigma_min=0.01, sigma_max=50, T=1.0):
    _, sigma_max_prior = ve_marginal_prob(None, T, sigma_min=sigma_min, sigma_max=sigma_max)
    return torch.randn(*shape) * sigma_max_prior

def ve_sde(t, sigma_min=0.01, sigma_max=50):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

@torch.no_grad()
def ddpm_sample(model, cond, alphas, alphas_cumprod, device, num_steps=1000):
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
        prior=ve_prior,   # the noise distribution
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
    x = init_x                                                      # the initial is sampled from N(0, sigma_max)
    poses = []
    grads = []
    with torch.no_grad():
        for i, time_step in enumerate(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            # data['sampled_pose'] = x
            # data['t'] = batch_time_step
            cond = cond.to(device)
            grad = model(x, cond, batch_time_step)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2                                      # a scalar, HOW LARGE IT IS?
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)    
            
            x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
            x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)

            # import ipdb
            # if i == num_steps-1:                        # check the last step
            #     ipdb.set_trace()

            # Predictor step (Euler-Maruyama)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2*grad # R-SDE in this case, just the diffusion**2 *grad
            mean_x = x + drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            
            # normalisation
            # x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            # TODO: change the normalization part
            x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
            x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)
            poses.append(x.unsqueeze(0))
            grads.append(grad.unsqueeze(0))
    
    xs = torch.cat(poses, dim=0) # (num_steps, bs, pose_dim)
    grads = torch.cat(grads, dim=0)
    # xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    # mean_x[:, -3:] += data['pts_center']
    # mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    # add normalization again  
    # mean_x[:, :3] /= torch.norm(mean_x[:, :3], dim=-1, keepdim=True)
    # mean_x[:, 3:6] /= torch.norm(mean_x[:, 3:6], dim=-1, keepdim=True)
    # # The last step does not include any noise
    return grads.permute(1, 0, 2), xs.permute(1, 0, 2), mean_x # (bs, num_steps, pos_dim)



def cond_ode_sampler(
        model,            # the score model
        cond,             # the points (condition)
        prior=ve_prior,   # the noise distribution
        sde_coeff=ve_sde, # the forward SDE setting 
        atol=1e-5,        # used to solve ode
        rtol=1e-5,        # same
        device='cuda',      
        num_steps=500,
        eps=1e-5,
        T=1,
        init_x=None,
    ):
    pose_dim = 6
    batch_size=cond.shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape

    def score_eval_wrapper(x, cond, batch_time_step):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = model(x, cond, batch_time_step)
        return score.cpu().numpy().reshape((-1,))

    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(x, cond, time_steps)    
    
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if False:                                           # cannot use currently
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    num_steps = xs.shape[0]
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :3] = xs[:, :3]/xs[:, :3].norm()
    xs[:, 3:] = xs[:, 3:]/xs[:, 3:].norm()
    xs = xs.reshape(num_steps, batch_size, -1)
    return xs.permute(1, 0, 2), x