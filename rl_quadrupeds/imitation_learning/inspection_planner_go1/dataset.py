import torch
from torch.utils.data import Dataset, DataLoader

from simulator import Simulator

class InspectionTeacherStudentDataset(Dataset):
    def __init__(
        self,
        noise: bool = True,
        correct_axes: bool = True,
        normalized: bool = True,
        num_simulations_per_epoch: int = 100,
        num_envs_per_simulation: int = 65536,
        history_len: int = 3,
        num_steps: int = 1,
        device: str = "cuda:0",
        return_unpushed_action: bool = False,
        cfg_json: str = "cfg.json"
    ):
        self.noise = noise
        self.correct_axes = correct_axes
        self.normalized = normalized
        self.num_simulations_per_epoch = num_simulations_per_epoch
        self.num_envs_per_simulation = num_envs_per_simulation
        self.history_len = history_len
        self.num_steps = num_steps
        self.device = device
        self.return_unpushed_action = return_unpushed_action

        self.simulator = Simulator(
            num_envs=self.num_envs_per_simulation, 
            device=self.device, 
            cfg_json=cfg_json
        )
        self.simulator.build(new_simulation=True)

    def __len__(self):
        return self.num_simulations_per_epoch

    def __getitem__(self, idx):
        if self.num_steps == 1:
            obs, expert_action = self.simulator.step(
                normalized=self.normalized,
                noise=self.noise, 
                correct_axes=self.correct_axes
            )
            if self.return_unpushed_action:   
                unpushed_action = self.simulator.get_unpushed_action(normalized=self.normalized).squeeze(1)
                return obs, expert_action.squeeze(1), unpushed_action
            return obs, expert_action.squeeze(1)

        obs, expert_action = self.simulator.simulate_step(
            num_steps=self.num_steps,
            history_len=self.history_len,
            normalized=self.normalized,
            noise=self.noise,
            correct_axes=self.correct_axes
        )
        # original shapes: [num_steps, num_envs, dim]
        num_steps, num_envs, obs_dim = obs.shape
        _, _, act_dim = expert_action.shape

        # merge steps and envs -> [num_steps*num_envs, dim]
        obs = obs.reshape(num_steps * num_envs, obs_dim)
        expert_action = expert_action.reshape(num_steps * num_envs, act_dim)
        
        return obs, expert_action

if __name__ == "__main__":
    dataset = InspectionTeacherStudentDataset(num_steps=3)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for i, (obs, expert_action) in enumerate(dataloader):
        print(f"Batch {i}: obs shape {obs.shape}, expert action shape {expert_action.shape}")
        if i >= 2:
            break