import sys
import os 
import tqdm
sys.path.append(os.getcwd())

import argparse
import torch
from model.DDPM_model import *
from datasets.real_time_dataset import *
from configs.load_config import load_config
from model.sampler import cond_pc_sampler, ddpm_sample, cond_ode_sampler

@torch.no_grad()
def test(experiment_name, T, sample_mode="ve_ode", epoch_num=None, infer_num=10, device=torch.device("cuda")):

    # load the config by the experiment_name
    config_path = 'configs/add_encoder.yaml'
    config = load_config(config_path)
    config['experiments_name'] = experiment_name

    # load the checkpoint
    if epoch_num:
        save_path  = config["checkpoints_path"]+config['experiments_name']+'/'+f'checkpoint_epoch_{epoch_num}.pth'
    else:
        save_path  = config["checkpoints_path"]+config['experiments_name']+'/'+f'latest_checkpoint.pth'
    checkpoint = torch.load(save_path)

    # load the state dict
    model = ConditionedDiffusionModel(data_dim=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # load the dataset
    RTDataset = RealTimeDataset(config_path)
    RTDataloader = DataLoader(dataset=RTDataset, batch_size=1)

    #  precomputed alpha and alpah cumprod only for ddpm sample mode
    if sample_mode == 'sample':
        timesteps=1000
        betas = linear_beta_schedule(timesteps)
        alphas, alphas_cumprod = get_alphas(betas, device)

    # create the residual list
    residual_list = []
 
    # start evaluate ...
    for i, (partial_points, gt_rotation) in tqdm.tqdm(enumerate(RTDataloader)):
        # print(partial_points.shape)
        if sample_mode == "ve_pc":
            grads, pre_rotation_list, pre_rotation = cond_pc_sampler(model, partial_points.to(torch.float32))
        elif sample_mode == "sample":
            pre_rotation = ddpm_sample(model, partial_points.to(torch.float32), alphas, alphas_cumprod, device, timesteps)
        elif sample_mode == "ve_ode":
            _, pre_rotation = cond_ode_sampler(model, partial_points.to(torch.float32), T=T)
        pre_rotation = pre_rotation.detach().to('cpu')
        pre_rotation = pre_rotation.squeeze(0)
        pre_x, pre_y = pre_rotation[:3], pre_rotation[3:]
        gt_x, gt_y = gt_rotation[0,0,:].view(-1).to(torch.float32), gt_rotation[0,1,:].view(-1).to(torch.float32)

        theta_x = (torch.dot(pre_x, gt_x)/pre_x.norm()/gt_x.norm()).item()
        theta_y = (torch.dot(pre_y, gt_y)/pre_y.norm()/gt_y.norm()).item()
        theta_orthoal = (torch.dot(pre_x, pre_y)/pre_y.norm()/pre_x.norm()).item()
        theta_orthoal2 = (torch.dot(gt_x, gt_y)/pre_y.norm()/pre_x.norm()).item()
        residual_list.append((theta_x, theta_y, theta_orthoal, theta_orthoal2))

        # ==== temp ====
        if i > infer_num:
            break

    for result_tuple in residual_list:
        print(f"x, y dotproduct is {result_tuple[0], result_tuple[1]}, orthogonal is {result_tuple[2]}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--T0", type=float, default=1.)
    args = parser.parse_args()
    test(experiment_name=args.experiment_name, T=args.T0, epoch_num=500)