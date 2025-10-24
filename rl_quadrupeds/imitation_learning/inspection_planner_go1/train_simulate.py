import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import your dataset (now uses simulate_step with num_steps argument)
from dataset import InspectionTeacherStudentDataset

# import the rsl_rl ActorCritic used by the runner
# adjust import path if your repo layout differs
from rsl_rl.modules.actor_critic import ActorCritic

def train_bc_actor_critic(
    num_epochs=10,
    num_envs=65536,
    history_len=3,
    lr=1e-4,
    device="cuda:0",
    checkpoint=None,
    log_dir_root="runs/bc_actor_critic",
    actor_hidden_dims=[512, 256, 128],
    num_steps=1,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # === Create timestamped log/checkpoint directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)

    # dataset and dataloader
    dataset = InspectionTeacherStudentDataset(
        noise=False, 
        num_steps=num_steps, 
        num_envs_per_simulation=num_envs,
        history_len=history_len,
        correct_axes=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # each item is already a batch
        shuffle=True,
        num_workers=0,
    )

    # peek to infer dims
    obs_sample, expert_action_sample = next(iter(dataloader))
    if not torch.is_tensor(obs_sample):
        obs_sample = torch.tensor(obs_sample)
    if not torch.is_tensor(expert_action_sample):
        expert_action_sample = torch.tensor(expert_action_sample)

    # obs_sample shape: [num_steps, num_envs, obs_dim]
    obs_dim = obs_sample.shape[-1]
    action_dim = expert_action_sample.shape[-1]

    # instantiate ActorCritic
    policy = ActorCritic(
        num_actions=action_dim,
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
    ).to(device)

    # Train only the actor parameters (behavior cloning)
    optimizer = optim.Adam(policy.actor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    start_epoch = 0

    # === Load checkpoint if provided ===
    if checkpoint is not None and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            policy.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            policy.load_state_dict(ckpt["state_dict"])
        else:
            policy.load_state_dict(ckpt)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                print("Warning: couldn't load scheduler state dict (ignored).")
        if "iter" in ckpt:
            start_epoch = ckpt["iter"] + 1

    # === Training loop ===
    last_avg_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs-1}", leave=False)

        for i, (obs, expert_action) in enumerate(progress_bar):
            # obs, expert_action: [num_steps, num_envs, dim]
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs)
            if not torch.is_tensor(expert_action):
                expert_action = torch.tensor(expert_action)

            obs = obs.to(device)
            expert_action = expert_action.to(device)

            # flatten steps and envs: [(num_steps*num_envs), dim]
            obs_flat = obs.reshape(-1, obs.shape[-1])
            expert_action_flat = expert_action.reshape(-1, expert_action.shape[-1])

            pred_action = policy.act_inference(obs_flat)
            loss = criterion(pred_action, expert_action_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            writer.add_scalar("Loss/step", loss.item(), epoch * len(dataloader) + i)

        avg_loss = epoch_loss / max(1, len(dataloader))
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.6f} | LR: {current_lr:.6e}")

        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        scheduler.step(avg_loss)

        # Save checkpoint only if loss improved
        if avg_loss < last_avg_loss:
            last_avg_loss = avg_loss
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "iter": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "infos": None,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()

    # Save final model in the run directory
    final_model_path = os.path.join(run_dir, "bc_actor_critic_final.pt")
    torch.save({"model_state_dict": policy.state_dict()}, final_model_path)
    print(f"Training complete. Final model saved as {final_model_path}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--history-len", type=int, default=3)
    args = parser.parse_args()

    trained_policy = train_bc_actor_critic(
        num_epochs=args.epochs,
        num_envs=args.num_envs,
        history_len=args.history_len,
        lr=args.lr,
        device=args.device,
        checkpoint=args.checkpoint,
        actor_hidden_dims=[512, 256, 128],
        num_steps=args.num_steps,
    )