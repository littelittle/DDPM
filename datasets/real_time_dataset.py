import sys
import os 
sys.path.append(os.getcwd())
from torch.utils.data import Dataset, DataLoader
from datasets.pointcloud_generator import *
from datasets.obscure_render import *
from configs.load_config import load_config

class RealTimeDataset(Dataset):
    '''
    based on the configuration, the Real time dataset randomly generate rendered partial
    points as well as rotation matrix which need to be applied on the right hand side.
    the length of the dataset is fully determined on the configuration, since the data is generated
    real time.

    partial_points: ['num_points', 3]
    rot_matrix: [3, 3]

    '''
    def __init__(self, config_path):
        self.config = load_config(config_path)['dataset']
        self.len = self.config['dataset_size']   
        self.r = self.config['ray_tracing']['r']
        self.num_rays = self.config['ray_tracing']['num_rays']     
        self.num_points = self.config['ray_tracing']['num_points']

    def __len__(self,):
        return self.len
    
    def __getitem__(self, index):
        # partial_points, rot_matrix = ray_tracing(self.r, self.num_rays, self.num_points)
        # NOTE: Use the brand new data generator now!
        points = sample_points_on_cuboid_surface()
        rot_matrix = lie_group(lie_algebra())
        partial_points = torch.from_numpy(points)@rot_matrix # not partial at all hhh
        return partial_points, rot_matrix
    

if __name__ == "__main__":
    import time
    
    RTDataset = RealTimeDataset(config_path='configs/trial.yaml')
    config = load_config(config_path='configs/trial.yaml')['dataloader']
    dataloader = DataLoader(RTDataset, batch_size=config['batch_size'])
    start_time = time.time()
    print(RTDataset.len)
    for i, (pp, rm) in enumerate(dataloader):
        print(pp.shape, rm.shape)
    finish_time = time.time()

    # test the load speed
    print(f"total cost :{finish_time-start_time}s in the {RTDataset.len} samples dataset")