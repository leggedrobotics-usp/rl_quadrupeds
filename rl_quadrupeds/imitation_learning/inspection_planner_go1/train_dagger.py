# train_dagger.py

import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from simulator import Simulator
from rsl_rl.modules.actor_critic import ActorCritic

def train_dagger(
    num_epochs=10,
    rollout_steps=5,
    num_envs=2048,
    device="cuda:0",
    lr=1e-4,
    checkpoint=None,
    log_dir_root="runs/dagger_actor_critic",
    actor_hidden_dims=[512, 256, 128],
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # === Create timestamped log/checkpoint directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # === Simulator ===
    sim = Simulator(num_envs=num_envs, device=device, cfg_json="cfg.json")

    # Peek to infer dims
    obs_sample = sim.get_observation(normalized=True, noise=False)
    expert_action_sample = sim.get_expert_action(normalized=True)
    obs_dim = obs_sample.shape[-1]
    action_dim = expert_action_sample.shape[-1]

    # === Policy ===
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

    optimizer = optim.Adam(policy.actor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0
    )

    start_epoch = 0
    if checkpoint is not None and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "iter" in ckpt:
            start_epoch = ckpt["iter"] + 1

    # === Aggregated dataset (starts empty) ===
    agg_obs = []
    agg_act = []

    last_avg_loss = float("inf")
    for epoch in range(start_epoch, num_epochs):
        print(f"\n=== Epoch {epoch} ===")
        sim_obs_batch, sim_act_batch = [], []

        # === Run rollout ===
        obs = sim.get_observation(normalized=True)
        for step in tqdm(range(rollout_steps), desc="Rollout"):
            obs_tensor = obs.to(device)
            with torch.no_grad():
                policy_action = policy.act_inference(obs_tensor)

            # Always query expert for DAgger
            expert_action = sim.get_expert_action(normalized=True)

            # Collect dataset
            sim_obs_batch.append(obs_tensor.cpu())
            sim_act_batch.append(expert_action.cpu())

            # Move robot using policy prediction (DAgger core idea)
            action = policy_action[:, :3].clone()
            action = sim.expert_policy._detransform_axes(action)
            sim.robot_builder.viewpoint += sim.expert_policy.denormalize(action)
            sim.robot_builder.viewpoint = sim.robot_builder.enforce_viewpoint_limits(
                sim.robot_builder.viewpoint
            )
            sim.robot_builder._create_contour()
            sim.robot_builder.ray_caster_builder._ray_tracing_results = None

            # Advance simulation
            obs, _ = sim.step(normalized=True, rebuild_simulation=False)

        # Aggregate data
        agg_obs.append(torch.cat(sim_obs_batch, dim=0))
        agg_act.append(torch.cat(sim_act_batch, dim=0))
        obs_dataset = torch.cat(agg_obs, dim=0)
        act_dataset = torch.cat(agg_act, dim=0)

        # === Train on aggregated dataset ===
        policy.train()
        epoch_loss = 0.0
        batch_size = 32768
        num_batches = (len(obs_dataset) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Training"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(obs_dataset))
            obs_batch = obs_dataset[start:end].to(device)
            act_batch = act_dataset[start:end].to(device)

            pred_action = policy.act_inference(obs_batch)
            loss = criterion(pred_action, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/step", loss.item(), epoch * num_batches + i)

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.6f} | LR: {current_lr:.6e}")

        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        scheduler.step(avg_loss)

        # Save checkpoint if improved
        if avg_loss < last_avg_loss:
            last_avg_loss = avg_loss
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "iter": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()
    final_model_path = os.path.join(run_dir, "dagger_actor_critic_final.pt")
    torch.save({"model_state_dict": policy.state_dict()}, final_model_path)
    print(f"Training complete. Final model saved as {final_model_path}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    train_dagger(
        num_epochs=args.epochs,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        device=args.device,
        lr=args.lr,
        checkpoint=args.checkpoint,
    )