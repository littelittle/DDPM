import sys
import os 
sys.path.append(os.getcwd())

import argparse
import torch
from model.DDPM_model import *
from datasets.real_time_dataset import *
from configs.load_config import load_config
from model.sampler import cond_pc_sampler, sample

@torch.no_grad()
def test(experiment_name, sample_mode="ve_pc"):
    # hyper params
    timesteps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the config
    config_path = 'configs/add_encoder.yaml'
    config = load_config(config_path)
    config['experiments_name'] = experiment_name

    # load the checkpoint
    save_path  = config["checkpoints_path"]+config['experiments_name']+'/'+f'checkpoint_epoch_300.pth'
    checkpoint = torch.load(save_path)

    # load the state dict
    model = ConditionedDiffusionModel(data_dim=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # load the dataset
    RTDataset = RealTimeDataset(config_path)
    RTDataloader = DataLoader(dataset=RTDataset, batch_size=1)

    #  precomputed alpha and alpah cumprod
    if sample_mode == 'sample':
        betas = linear_beta_schedule(timesteps)
        alphas, alphas_cumprod = get_alphas(betas, device)

    # create the residual list
    residual_list = []

    # start evaluate
    for i, (partial_points, gt_rotation) in enumerate(RTDataloader):
        # print(partial_points.shape)
        if sample_mode == "ve_pc":
            grads, pre_rotation_list, pre_rotation = cond_pc_sampler(model, partial_points.to(torch.float32))
        elif sample_mode == "sample":
            pre_rotation = sample(model, partial_points.to(torch.float32), alphas, alphas_cumprod, device, timesteps)
        pre_rotation = pre_rotation.detach().to('cpu')
        pre_rotation = pre_rotation.squeeze(0)
        pre_x, pre_y = pre_rotation[:3], pre_rotation[3:]
        gt_x, gt_y = gt_rotation[0,0,:].view(-1).to(torch.float32), gt_rotation[0,1,:].view(-1).to(torch.float32)
        # import ipdb 
        # ipdb.set_trace()
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