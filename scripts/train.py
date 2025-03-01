import sys
import os 
sys.path.append(os.getcwd())

import torch
from model.DDPM_model import *
from datasets.real_time_dataset import *


# 参数配置
timesteps = 1000
data_dim = 6
cond_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ConditionedDiffusionModel(
    data_dim=data_dim,
    cond_size=cond_size
).to(device)

# 准备调度参数
betas = linear_beta_schedule(timesteps)
alphas, alphas_cumprod = get_alphas(betas)

# 训练循环示例
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    # 假设已准备好训练数据
    # x0_batch: [batch, 6]
    # cond_batch: [batch, 28, 28]
    loss = train_step(
        model, x0_batch, cond_batch, 
        alphas_cumprod, device, optimizer, loss_fn
    )
    print(f"Epoch {epoch} | Loss: {loss:.4f}")

# 采样示例
condition = torch.randn(28, 28)  # 生成条件
generated_data = sample(model, condition, alphas, alphas_cumprod, device)