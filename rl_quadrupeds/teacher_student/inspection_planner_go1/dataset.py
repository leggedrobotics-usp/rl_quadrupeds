import torch
from torch.utils.data import Dataset, DataLoader

from simulator import Simulator

class InspectionTeacherStudentDataset(Dataset):
    def __init__(self):
        self.num_simulations_per_epoch = 100
        self.num_envs_per_simulation = 131072
        self.device = "cuda:0"
        self.simulator = Simulator(num_envs=self.num_envs_per_simulation, device=self.device)

    def __len__(self):
        return self.num_simulations_per_epoch

    def __getitem__(self, idx):
        obs, expert_action = self.simulator.step()
        return obs, expert_action.squeeze(1)

if __name__ == "__main__":
    dataset = InspectionTeacherStudentDataset()
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for i, (obs, expert_action) in enumerate(dataloader):
        print(f"Batch {i}: obs shape {obs.shape}, expert action shape {expert_action.shape}")
        if i >= 2:
            break
