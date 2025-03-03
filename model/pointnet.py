import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Tnet(nn.Module):
    def __init__(self, input_channel, feature_channel):
        super(Tnet, self).__init__()
        self.i_c = input_channel

        self.conv1 = nn.Conv1d(input_channel, feature_channel, 1)
        self.fc1 = nn.Linear(feature_channel, feature_channel)
        self.fc2 = nn.Linear(feature_channel, input_channel*input_channel)
    
        self.bn1 = nn.BatchNorm1d(feature_channel)
        self.bn2 = nn.BatchNorm1d(feature_channel)
        self.bn3 = nn.BatchNorm1d(input_channel*input_channel)

        self.iden = Variable(torch.from_numpy(np.eye(self.i_c).flatten().astype(np.float32))).view(1, -1)

    def forward(self, x:torch.Tensor):
        # x = x.transpose(1, 2) # (bs, i_c, n_p)
        x = F.relu(self.bn1(self.conv1(x))) # (bs, f_c, n_p)
        x = torch.max(x, dim=-1)[0] # (bs, f_c)

        x = F.relu(self.bn2(self.fc1(x))) # (bs, f_c)
        x = F.relu(self.bn3(self.fc2(x))) # (bs, i_c*i_c)

        x = x + self.iden.to(x.device)
        x = x.view(-1, self.i_c, self.i_c)

        return x

class Pointnet(nn.Module):
    def __init__(self, out_feature_dim=256):
        super(Pointnet, self).__init__()
        self.tnet1 = Tnet(3, 16)
        self.con1 = nn.Conv1d(3, 8, 1)
        self.con2 = nn.Conv1d(8, 32, 1)

        self.tnet2 = Tnet(32, 64)
        self.con3 = nn.Conv1d(32, 128, 1)
        self.con_add = nn.Conv1d(128, 128, 1)
        self.con4 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn_add = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, out_feature_dim)


    def forward(self, x):
        """
        x: shape:(batchsize, numpoints(100), 3)
        """
        x = x.transpose(1, 2)

        x_trans = self.tnet1(x)
        x = torch.bmm(x_trans, x)
        x = F.relu(self.bn1(self.con1(x)))
        x = F.relu(self.bn2(self.con2(x)))

        f_trans = self.tnet2(x)
        x = torch.bmm(f_trans, x)
        x = F.relu(self.bn3(self.con3(x)))
        x = F.relu(self.bn_add(self.con_add(x)))
        x = F.relu(self.bn4(self.con4(x)))

        x = torch.max(x, dim=-1)[0]
        x = F.relu(self.fc1(x))

        return x


if __name__ == "__main__":
    # generate points
    x = torch.randn(18, 100, 3)
    # tnet = Tnet(3,16)
    # y = tnet(x)
    pn = Pointnet()
    y = pn(x)
    print(y.shape)
    
