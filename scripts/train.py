import sys
import os 
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
from model.DDPM_model import *
from datasets.real_time_dataset import *


# 参数配置
config_path = 'configs/trial.yaml'
timesteps = 1000
data_dim = 6
cond_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load configs
config = load_config(config_path)

# initialize the model
model = ConditionedDiffusionModel(
    data_dim=data_dim,
    cond_size=cond_size
).to(device)

# Prepare scheduling parameters
betas = linear_beta_schedule(timesteps)
alphas, alphas_cumprod = get_alphas(betas)

# set teh optimizer and the loss func
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# set the dataset and the dataloader
RTDataset = RealTimeDataset(config_path)
RTDataloader = DataLoader(dataset=RTDataset, batch_size=config['dataloader']['batch_size'])


for epoch in range(1000):
    # 假设已准备好训练数据
    # rotation_batch: [batch, 3, 3] -> [batch, 6]
    # partial_points_batch: [batch, num_points, 3]

    for i, (partial_points_batch, rotation_batch) in RTDataloader:

        loss = train_step(
            model, rotation_batch[:, :2, :].view(-1, 6), partial_points_batch, 
            alphas_cumprod, device, optimizer, loss_fn
        )
        print(f"Step{i:4.0f} | Loss:{loss:4.4f}")

    print(f"Epoch {epoch:4.0f} | Loss: {loss:4.4f}")

# samples
condition = torch.randn(28, 28)  # 生成条件
generated_data = sample(model, condition, alphas, alphas_cumprod, device)