import torch
import torch.nn as nn
import math

# ======================
# 时间步相关参数规划模块
# ======================
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """线性beta调度器，返回所有时间步的beta值"""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alphas(betas):
    """根据beta计算alpha和累积alpha乘积"""
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod

# ======================
# 时间步编码模块
# ======================
class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码器，将时间步t映射为高维向量"""
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
# 条件编码模块
# ======================
class ConditionEncoder(nn.Module):
    """条件编码器，处理n*n维的条件输入"""
    def __init__(self, cond_size, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # 假设输入是单通道
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*(cond_size//4)*(cond_size//4), output_dim)
        )

    def forward(self, cond):
        return self.main(cond.unsqueeze(1))  # 添加通道维度

# ======================
# 核心网络模型
# ======================
class ConditionedDiffusionModel(nn.Module):
    """条件扩散模型主体"""
    def __init__(self, 
                 data_dim=6, 
                 cond_size=28,
                 time_emb_dim=32,
                 cond_emb_dim=128):
        super().__init__()
        
        # 时间步编码器
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 条件编码器
        self.cond_encoder = ConditionEncoder(cond_size, cond_emb_dim)
        
        # 噪声预测网络
        self.main = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim + cond_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim)
        )
        
    def forward(self, x, cond, t):
        # 输入维度验证
        # x: [batch, data_dim]
        # cond: [batch, cond_size, cond_size]
        # t: [batch]
        
        # 获取各组件编码
        t_emb = self.time_mlp(t)  # [batch, time_emb_dim]
        cond_emb = self.cond_encoder(cond)  # [batch, cond_emb_dim]
        
        # 拼接所有特征
        combined = torch.cat([x, t_emb, cond_emb], dim=1)
        
        # 预测噪声
        return self.main(combined)

# ======================
# 训练流程函数
# ======================
def train_step(model, x0, cond, alphas_cumprod, device, optimizer, loss_fn):
    """
    单次训练步骤
    参数:
        model: 扩散模型
        x0: 原始数据 [batch, data_dim]
        cond: 条件数据 [batch, cond_size, cond_size]
        alphas_cumprod: 预计算的alpha累积乘积
        device: 计算设备
        optimizer: 优化器
        loss_fn: 损失函数
    """
    model.train()
    
    # 准备数据
    batch_size = x0.shape[0]
    x0 = x0.to(device)
    cond = cond.to(device)
    
    # 随机采样时间步
    t = torch.randint(0, len(alphas_cumprod), (batch_size,), device=device).long()
    
    # 计算alpha_bar
    alpha_bar = alphas_cumprod[t].unsqueeze(-1)  # [batch, 1]
    
    # 生成带噪数据
    eps = torch.randn_like(x0)  # 真实噪声
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * eps
    
    # 预测噪声
    pred_eps = model(xt, cond, t)
    
    # 计算损失
    loss = loss_fn(pred_eps, eps)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# ======================
# 采样生成函数
# ======================
@torch.no_grad()
def sample(model, cond, alphas, alphas_cumprod, device, num_steps=1000):
    """
    采样生成过程
    参数:
        model: 训练好的模型
        cond: 条件数据 [1, cond_size, cond_size]
        alphas: 预计算的alpha值
        alphas_cumprod: 预计算的alpha累积乘积
        device: 计算设备
        num_steps: 总时间步数
    """
    model.eval()
    
    # 初始化纯噪声
    x = torch.randn(1, model.main[-1].out_features).to(device)
    cond = cond.unsqueeze(0).to(device)
    
    # 逐步去噪
    for t in reversed(range(num_steps)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        pred_eps = model(x, cond, t_batch)
        
        # 计算alpha相关参数
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        # 计算反向过程参数
        beta_tilde_t = (1 - alpha_bar_t_prev)/(1 - alpha_bar_t) * (1 - alpha_t)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
        
        # 反向过程采样
        mean = (x - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        x = mean + torch.sqrt(beta_tilde_t) * noise
    
    return pred_x0.squeeze(0)