import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

@torch.no_grad()
def reset_root_state_uniform_if_inspection_done(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    avoid_pose_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset or reuse the asset root state depending on:
    - inspection_done flag
    - level progression (only resample position/orientation if last level and inspection is done).
    """

    # Extract asset
    asset = env.scene[asset_cfg.name]

    # --- Inspection done flag ---
    done_mask = (
        env._inspection_done[env_ids]
        if hasattr(env, "_inspection_done")
        else torch.zeros(len(env_ids), dtype=torch.bool, device=asset.device)
    )

    # --- Last level mask (default to True if env_levels missing) ---
    if hasattr(env, "env_levels") and hasattr(env, "num_levels"):
        last_level_mask = (env.env_levels[env_ids] == env.num_levels)
    else:
        last_level_mask = torch.ones(len(env_ids), dtype=torch.bool, device=asset.device)

    # --- Completion count logic ---
    resample_mask = None
    if hasattr(env, "_level_completion_counts"):
        # Current level for env_ids
        cur_levels = env.env_levels[env_ids]
        # Look up how many times those levels have been completed
        counts = env._level_completion_counts[env_ids, cur_levels]
        # Require: inspection_done & at least 1 completion already
        resample_mask = done_mask & (counts > 0)
    else:
        resample_mask = done_mask & last_level_mask

    # --- Lazy buffer init ---
    attr_name = f"last_sampled_state_{asset_cfg.name}"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, torch.zeros(env.num_envs, 13, device=asset.device, dtype=torch.float32))
        # Force resampling on first reset
        resample_mask = torch.ones(len(env_ids), dtype=torch.bool, device=asset.device)

    last_sampled_state = getattr(env, attr_name)
    new_root_states = last_sampled_state[env_ids].clone()

    # --- Resample logic ---
    if resample_mask.any():
        root_states = asset.data.default_root_state[env_ids[resample_mask]].clone()

        # Sample pose deltas
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        pose_ranges = torch.tensor([pose_range.get(k, (0.0, 0.0)) for k in pose_keys], device=asset.device)
        rand_samples = math_utils.sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (resample_mask.sum(), 6), device=asset.device
        )

        # Apply avoid ranges
        if avoid_pose_range:
            for i, key in enumerate(pose_keys):
                if key in avoid_pose_range:
                    low, high = avoid_pose_range[key]
                    inside_mask = (rand_samples[:, i] > low) & (rand_samples[:, i] < high)
                    lower_dist = (rand_samples[:, i] - low).abs()
                    upper_dist = (rand_samples[:, i] - high).abs()
                    rand_samples[inside_mask, i] = torch.where(
                        lower_dist[inside_mask] < upper_dist[inside_mask],
                        torch.full_like(rand_samples[inside_mask, i], low),
                        torch.full_like(rand_samples[inside_mask, i], high),
                    )

        # Positions
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids[resample_mask]] + rand_samples[:, 0:3]

        # Orientation deltas
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        # Velocities
        vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in vel_keys], device=asset.device)
        rand_samples_vel = math_utils.sample_uniform(
            vel_ranges[:, 0], vel_ranges[:, 1], (resample_mask.sum(), 6), device=asset.device
        )
        velocities = root_states[:, 7:13] + rand_samples_vel

        # Update state buffer
        new_root_states[resample_mask, 0:3] = positions
        new_root_states[resample_mask, 3:7] = orientations
        new_root_states[resample_mask, 7:13] = velocities
        last_sampled_state[env_ids[resample_mask]] = new_root_states[resample_mask]

    # --- Write to simulation ---
    asset.write_root_pose_to_sim(
        torch.cat([new_root_states[:, 0:3], new_root_states[:, 3:7]], dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(new_root_states[:, 7:13], env_ids=env_ids)