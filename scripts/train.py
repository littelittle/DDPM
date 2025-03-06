import sys
import os 
sys.path.append(os.getcwd())

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model.DDPM_model import *
from datasets.real_time_dataset import *
import argparse


# hyper params
config_path = 'configs/add_encoder.yaml'
timesteps = 1000
data_dim = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load configs
config = load_config(config_path)

# get the args from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='default' )
parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
args = parser.parse_args()
config['experiments_name'] = args.experiment_name
config['resume'] = args.resume

writer = SummaryWriter(f'experiments/{config["experiments_name"]}')

# initialize the model
model = ConditionedDiffusionModel(
    data_dim=data_dim,
)
start_epoch = 0

# Prepare scheduling parameters
betas = linear_beta_schedule(timesteps)
alphas, alphas_cumprod = get_alphas(betas, device)

# set teh optimizer and the loss func
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# check if to resume the checkpoints
if config['resume']:
    save_path = config["checkpoints_path"]+config['experiments_name']+'/'+f'latest_checkpoint.pth'
    try:
        checkpoint = torch.load(save_path)
    except:
        print(f"{save_path} not found!")
        raise IOError
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                # 只把 `exp_avg` 和 `exp_avg_sq` 这些梯度累积量移到 CUDA
                if k in ["exp_avg", "exp_avg_sq"]:  
                    state[k] = v.to(device)
                # `step` 相关的计数变量保持在 CPU
                elif k in ["step", "max_exp_avg_sq"]:
                    state[k] = v.cpu()


model.to(device)

# count the params in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"total number of params is {total_params}")
with open("experiments/model_param.txt", 'a') as f:
    f.write(f"{config['experiments_name']}\n{total_params}\n")

# set the dataset and the dataloader
RTDataset = RealTimeDataset(config_path)
RTDataloader = DataLoader(dataset=RTDataset, batch_size=config['dataloader']['batch_size'])

for epoch in range(start_epoch, start_epoch+1000):
    # rotation_batch: [batch, 3, 3] -> [batch, 6]
    # partial_points_batch: [batch, num_points, 3]

    for i, (partial_points_batch, rotation_batch) in enumerate(RTDataloader):
        # loss = train_step(
        #     model, rotation_batch[:, :2, :].view(-1, 6).to(torch.float32), partial_points_batch.to(torch.float32), 
        #     alphas_cumprod, device, optimizer, loss_fn
        # )
        loss = train_ve_step(
            model, rotation_batch[:, :2, :].view(-1, 6).to(torch.float32), partial_points_batch.to(torch.float32), 
            ve_marginal_prob, device, optimizer, loss_fn
        )
        print(f"Step{i:4.0f} | Loss:{loss:4.4f}")
        writer.add_scalars('Loss/train', {'loss':loss}, i+epoch*RTDataloader.__len__())

    print(f"Epoch {epoch:4.0f} | Loss: {loss:4.4f}")

    if epoch % 50 == 0 and epoch != 0:
        checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        save_path = config["checkpoints_path"]+config['experiments_name']+'/'+f'checkpoint_epoch_{epoch}.pth'
        if not os.path.exists(config["checkpoints_path"]+config['experiments_name']+'/'):
            os.mkdir(config["checkpoints_path"]+config['experiments_name']+'/')
        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved at epoch {epoch}')

    if epoch % 2 == 0 and epoch != 0:
        checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        save_path = config["checkpoints_path"]+config['experiments_name']+'/'+f'latest_checkpoint.pth'
        if not os.path.exists(config["checkpoints_path"]+config['experiments_name']+'/'):
            os.mkdir(config["checkpoints_path"]+config['experiments_name']+'/')
        torch.save(checkpoint, save_path)
        print(f'Checkpoint latest saved at epoch {epoch}')


writer.flush()
writer.close()

# samples
# condition = torch.randn(28, 28)  # 生成条件
# generated_data = sample(model, condition, alphas, alphas_cumprod, device)