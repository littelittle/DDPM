import sys
import os 
sys.path.append(os.getcwd())

import argparse
import torch
from model.DDPM_model import *
from datasets.real_time_dataset import *
from configs.load_config import load_config

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
    cond = cond.to(device)
    
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

@torch.no_grad()
def test(experiment_name):
    # hyper params
    timesteps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the config
    config_path = 'configs/add_encoder.yaml'
    config = load_config(config_path)
    config['experiments_name'] = experiment_name

    # load the checkpoint
    save_path  = config["checkpoints_path"]+config['experiments_name']+'/'+f'latest_checkpoint.pth'
    checkpoint = torch.load(save_path)

    # load the state dict
    model = ConditionedDiffusionModel(data_dim=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # load the dataset
    RTDataset = RealTimeDataset(config_path)
    RTDataloader = DataLoader(dataset=RTDataset, batch_size=1)

    #  precomputed alpha and alpah cumprod
    betas = linear_beta_schedule(timesteps)
    alphas, alphas_cumprod = get_alphas(betas, device)

    # create the residual list
    residual_list = []

    # start evaluate
    for i, (partial_points, gt_rotation) in enumerate(RTDataloader):
        print(partial_points.shape)
        pre_rotation = sample(model, partial_points.to(torch.float32), alphas, alphas_cumprod, device, timesteps)
        pre_rotation = pre_rotation.detach().to('cpu')
        pre_x, pre_y = pre_rotation[:3], pre_rotation[3:]
        gt_x, gt_y = gt_rotation[0,0,:].view(-1).to(torch.float32), gt_rotation[0,1,:].view(-1).to(torch.float32)
        theta_x = (torch.dot(pre_x, gt_x)/pre_x.norm()/gt_x.norm()).item()
        theta_y = (torch.dot(pre_y, gt_y)/pre_y.norm()/gt_y.norm()).item()
        theta_orthoal = (torch.dot(pre_x, pre_y)/pre_y.norm()/pre_x.norm()).item()
        theta_orthoal2 = (torch.dot(gt_x, gt_y)/pre_y.norm()/pre_x.norm()).item()
        residual_list.append((theta_x, theta_y, theta_orthoal, theta_orthoal2))
        if i > 10:
            break
    
    print(residual_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="default")
    args = parser.parse_args()
    test(args.experiment_name)