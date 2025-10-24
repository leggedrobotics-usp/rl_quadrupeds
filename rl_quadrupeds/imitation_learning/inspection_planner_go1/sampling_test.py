import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import InspectionTeacherStudentDataset


if __name__ == "__main__":
    num_simulations_per_epoch = 100  # smaller since sim is heavier
    num_steps = 15
    num_envs_per_simulation = 4096

    dataset = InspectionTeacherStudentDataset(
        noise=False,
        correct_axes=False,
        normalized=False,
        num_simulations_per_epoch=num_simulations_per_epoch,
        num_envs_per_simulation=num_envs_per_simulation,
        num_steps=num_steps,
        return_unpushed_action=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,  # each __getitem__ already returns full sim
        shuffle=False,
        num_workers=0
    )

    # Storage
    robot_positions = []
    robot_orientations = []
    lidar_values = []
    lidar_labels = []
    coverage_confidences = []
    # unpushed_step_actions = []
    step_actions = []
    inspection_flags = []

    # Collect samples
    for i, (obs, expert_action) in enumerate(
        tqdm(dataloader, desc="Sampling dataset", total=num_simulations_per_epoch)
    ):
        num_objects=(obs.shape[1] - 3 - 146) // 28
        viewpoint_idxs = (0, 3)  # x, y, orientation
        lidar_idxs = (3, 146 + 3)
        coverage_idxs = (146 + 3, num_objects*28 + 146 + 3)

        # Robot viewpoint
        viewpoint = obs[:, viewpoint_idxs[0]:viewpoint_idxs[1]]  # [N, 3]
        robot_positions.append(viewpoint[:, :2].cpu().numpy())   # x,y
        robot_orientations.append(viewpoint[:, 2].cpu().numpy()) # orientation

        # Lidar scan (interleaved distances and labels)
        lidar = obs[:, lidar_idxs[0]:lidar_idxs[1]] # [N, num_objects*146]
        lidar_values.append(lidar[:, 0::2].cpu().numpy())  # distances
        lidar_labels.append(lidar[:, 1::2].cpu().numpy())  # labels

        # Coverage confidence
        coverage = obs[:, coverage_idxs[0]:coverage_idxs[1]] # [N, num_objects*28]
        coverage_confidences.append(coverage.cpu().numpy())

        # Inspection flag = last column of expert_action
        inspection_flags.append(expert_action[:, -1].cpu().numpy())

        # Step action (dx, dy, dtheta)
        step_actions.append(expert_action[:, :3].cpu().numpy())
        # unpushed_step_actions.append(unpushed_action.cpu().numpy())

    # Convert to numpy
    robot_positions = np.concatenate(robot_positions, axis=0)
    robot_orientations = np.concatenate(robot_orientations, axis=0)
    lidar_values = np.concatenate(lidar_values, axis=0)
    lidar_labels = np.concatenate(lidar_labels, axis=0)
    coverage_confidences = np.concatenate(coverage_confidences, axis=0)
    # unpushed_step_actions = np.concatenate(unpushed_step_actions, axis=0)
    step_actions = np.concatenate(step_actions, axis=0)
    inspection_flags = np.concatenate(inspection_flags, axis=0)

    # Precompute for plotting
    viewpoints_cpu = np.concatenate([robot_positions, robot_orientations[:, None]], axis=1)
    stepped_cpu = viewpoints_cpu + step_actions
    # unpushed_cpu = viewpoints_cpu + unpushed_step_actions

    num_beams = min(lidar_values.shape[1], lidar_labels.shape[1])
    distances = lidar_values[:, :num_beams]
    labels = lidar_labels[:, :num_beams]

    # ------------------ Plotting ------------------
    os.makedirs("sampling_test_figures", exist_ok=True)

    plotting_steps = [
        "Viewpoint path",
        "Viewpoint heatmap",
        # "Unconstrained viewpoint heatmap",
        "Info flag aggregation",
        "Object coverage histogram",
        "Lidar distances",
        "Lidar label distribution",
        "Inspection flag distribution",
        "Inspection flag heatmap",
    ]

    for step in tqdm(plotting_steps, desc="Generating plots"):
        if step == "Viewpoint path":
            # ======= Vector field (direction only, colored by samples) =======
            grid_res = 0.25     # spacing between arrows
            min_count = 3       # min samples per cell to draw arrow
            arrow_length = 0.2  # fixed arrow length

            positions = viewpoints_cpu[:, :2]
            displacements = step_actions[:, :2]

            if positions.shape[0] == 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.set_title("No viewpoint samples available")
                fig.savefig("sampling_test_figures/viewpoints_flux.png")
                plt.close(fig)
                continue

            # Map positions to grid cells
            gx = np.round(positions[:, 0] / grid_res).astype(np.int64)
            gy = np.round(positions[:, 1] / grid_res).astype(np.int64)
            cells = np.stack([gx, gy], axis=1)

            dtype = np.dtype((np.void, cells.dtype.itemsize * cells.shape[1]))
            cells_view = np.ascontiguousarray(cells).view(dtype)
            uniq_cells, inverse = np.unique(cells_view, return_inverse=True)
            uniq_cells = uniq_cells.view(cells.dtype).reshape(-1, cells.shape[1])

            n_cells = uniq_cells.shape[0]
            sum_dx = np.bincount(inverse, weights=displacements[:, 0], minlength=n_cells)
            sum_dy = np.bincount(inverse, weights=displacements[:, 1], minlength=n_cells)
            counts = np.bincount(inverse, minlength=n_cells)

            # Compute mean displacement vectors
            mean_dx = np.zeros(n_cells)
            mean_dy = np.zeros(n_cells)
            valid = counts >= min_count
            mean_dx[valid] = sum_dx[valid] / counts[valid]
            mean_dy[valid] = sum_dy[valid] / counts[valid]

            # Normalize to fixed length arrows
            mags = np.sqrt(mean_dx**2 + mean_dy**2)
            nonzero = mags > 1e-12
            mean_dx[nonzero] = (mean_dx[nonzero] / mags[nonzero]) * arrow_length
            mean_dy[nonzero] = (mean_dy[nonzero] / mags[nonzero]) * arrow_length

            # Coordinates for plotting
            cell_xs = uniq_cells[:, 0].astype(float) * grid_res
            cell_ys = uniq_cells[:, 1].astype(float) * grid_res

            # Keep valid cells
            cell_xs = cell_xs[valid]
            cell_ys = cell_ys[valid]
            U = mean_dx[valid]
            V = mean_dy[valid]
            C = counts[valid]  # color by counts

            # Plot quiver
            fig, ax = plt.subplots(figsize=(8, 8))
            q = ax.quiver(
                cell_xs, cell_ys, U, V, C,
                angles="xy", scale_units="xy", scale=1.0,
                width=0.003, cmap="viridis"
            )

            # Add colorbar outside plot
            cbar = fig.colorbar(q, ax=ax, pad=0.02)
            cbar.set_label("Samples per cell")

            ax.set_xlabel("X (down)")
            ax.set_ylabel("Y (right)")
            ax.set_title(f"Viewpoint Movement Directions (grid_res={grid_res})")
            ax.set_aspect("equal", adjustable="box")

            # Fit tightly to data (no huge margins)
            ax.set_xlim(cell_xs.min(), cell_xs.max())
            ax.set_ylim(cell_ys.min(), cell_ys.max())

            fig.savefig("sampling_test_figures/viewpoints_flux.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        elif step == "Viewpoint heatmap":
            from scipy.interpolate import griddata

            grid_res = 0.05
            grid = {}

            for x, y in viewpoints_cpu[:, :2]:
                gx, gy = round(x / grid_res), round(y / grid_res)
                grid[(gx, gy)] = grid.get((gx, gy), 0) - 1

            for x, y in stepped_cpu[:, :2]:
                gx, gy = round(x / grid_res), round(y / grid_res)
                grid[(gx, gy)] = grid.get((gx, gy), 0) + 1

            xs = np.array([gx * grid_res for gx, gy in grid.keys()])
            ys = np.array([gy * grid_res for gx, gy in grid.keys()])
            vals = np.array(list(grid.values()))

            xi = np.linspace(xs.min(), xs.max(), 200)
            yi = np.linspace(ys.min(), ys.max(), 200)
            Xi, Yi = np.meshgrid(xi, yi)

            try:
                Zi = griddata((xs, ys), vals, (Xi, Yi), method="cubic", fill_value=0)
            except Exception:
                Zi = griddata((xs, ys), vals, (Xi, Yi), method="linear", fill_value=0)

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(Zi, extent=(xs.min(), xs.max(), ys.min(), ys.max()),
                           origin="lower", cmap="bwr", aspect="auto")
            plt.colorbar(im, ax=ax, label="Step Transition (-1=Start, +1=End)")
            ax.set_xlabel("X (down)")
            ax.set_ylabel("Y (right)")
            ax.set_title("Viewpoint Step Heatmap (Interpolated)\n-1=Start, +1=End")
            fig.savefig("sampling_test_figures/viewpoints_step_colormap.png")
            plt.close(fig)

        elif step == "Unconstrained viewpoint heatmap":
            from scipy.interpolate import griddata

            grid_res = 0.05
            grid = {}

            for x, y in viewpoints_cpu[:, :2]:
                gx, gy = round(x / grid_res), round(y / grid_res)
                grid[(gx, gy)] = grid.get((gx, gy), 0) - 1

            for x, y in unpushed_cpu[:, :2]:
                gx, gy = round(x / grid_res), round(y / grid_res)
                grid[(gx, gy)] = grid.get((gx, gy), 0) + 1

            xs = np.array([gx * grid_res for gx, gy in grid.keys()])
            ys = np.array([gy * grid_res for gx, gy in grid.keys()])
            vals = np.array(list(grid.values()))

            xi = np.linspace(xs.min(), xs.max(), 200)
            yi = np.linspace(ys.min(), ys.max(), 200)
            Xi, Yi = np.meshgrid(xi, yi)

            try:
                Zi = griddata((xs, ys), vals, (Xi, Yi), method="cubic", fill_value=0)
            except Exception:
                Zi = griddata((xs, ys), vals, (Xi, Yi), method="linear", fill_value=0)

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(Zi, extent=(xs.min(), xs.max(), ys.min(), ys.max()),
                           origin="lower", cmap="bwr", aspect="auto")
            plt.colorbar(im, ax=ax, label="Step Transition (-1=Start, +1=End)")
            ax.set_xlabel("X (down)")
            ax.set_ylabel("Y (right)")
            ax.set_title("Unpushed Viewpoint Step Heatmap (Interpolated)\n-1=Start, +1=End")
            fig.savefig("sampling_test_figures/unpushed_viewpoints_step_colormap.png")
            plt.close(fig)

        elif step == "Info flag aggregation":
            radius = 0.05
            grid = {}
            for (x, y), f in zip(viewpoints_cpu[:, :2], inspection_flags):
                key = (round(x / radius), round(y / radius))
                if key not in grid:
                    grid[key] = set()
                grid[key].add(int(f))

            colors = []
            xs, ys = [], []
            for (gx, gy), flags in grid.items():
                xs.append(gx * radius)
                ys.append(gy * radius)
                if flags == {0}:
                    colors.append("blue")
                elif flags == {1}:
                    colors.append("green")
                else:
                    colors.append("purple")

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(xs, ys, c=colors, s=10, alpha=0.7)
            ax.set_xlabel("X (down)")
            ax.set_ylabel("Y (right)")
            ax.set_title("Viewpoints Aggregated by Info Flag\nBlue=0, Green=1, Purple=Both")
            fig.savefig("sampling_test_figures/viewpoints_infoflag.png")
            plt.close(fig)

        elif step == "Object coverage histogram":
            coverage_confidences = coverage_confidences.astype(int)
            if coverage_confidences.size > 0:
                num_objects = coverage_confidences.shape[1]
                freq_ones = coverage_confidences.mean(axis=0)
                freq_zeros = 1 - freq_ones

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(np.arange(num_objects), freq_zeros, label="0", color="red")
                ax.bar(np.arange(num_objects), freq_ones, bottom=freq_zeros, label="1", color="green")
                ax.set_xlabel("Object index")
                ax.set_ylabel("Frequency (fraction)")
                ax.set_title("Object Coverage Confidence Distribution")
                ax.legend()
                fig.savefig("sampling_test_figures/object_coverage.png")
                plt.close(fig)

        elif step == "Lidar distances":
            fig, ax = plt.subplots(figsize=(10, 5))
            for beam in range(num_beams):
                beam_dist = distances[:, beam]
                beam_label = labels[:, beam]
                hits = beam_dist[beam_label == 0]
                if len(hits) > 0:
                    ax.boxplot(
                        hits,
                        positions=[beam],
                        widths=0.6,
                        showfliers=False
                    )

            ax.set_xlabel("Beam index")
            ax.set_ylabel("Distance")
            ax.set_title(f"Lidar Scan ({num_beams} beams)\nBlue=Hit Object")
            ax.set_xticks(np.arange(0, num_beams, max(1, num_beams // 20)))
            fig.savefig("sampling_test_figures/lidar_scan.png")
            plt.close(fig)

        elif step == "Lidar label distribution":
            label_freqs = [(labels[:, b] == 0).mean() for b in range(num_beams)]
            miss_freqs = [(labels[:, b] == -1).mean() for b in range(num_beams)]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(np.arange(num_beams), miss_freqs, label="Miss (-1)", color="red")
            ax.bar(np.arange(num_beams), label_freqs, bottom=miss_freqs, label="Hit (0)", color="green")
            ax.set_xlabel("Beam index")
            ax.set_ylabel("Frequency")
            ax.set_title("Lidar Label Distribution per Beam")
            ax.set_xticks(np.arange(0, num_beams, max(1, num_beams // 20)))
            ax.legend()
            fig.savefig("sampling_test_figures/lidar_labels.png")
            plt.close(fig)

        elif step == "Inspection flag distribution":
            inspection_flags = inspection_flags.astype(int)
            if inspection_flags.size > 0:
                freq_pos = (inspection_flags == 1).mean()
                freq_neg = (inspection_flags == -1).mean()

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.bar([-1], [freq_neg], label="-1 (no inspect)", color="red")
                ax.bar([1], [freq_pos], label="1 (inspect)", color="green")
                ax.set_xticks([-1, 1])
                ax.set_xticklabels(["-1", "1"])
                ax.set_ylabel("Frequency")
                ax.set_ylim(0, 1)
                ax.set_title("Inspection Flag Distribution")
                ax.legend()
                fig.savefig("sampling_test_figures/inspection_flag.png")
                plt.close(fig)

        elif step == "Inspection flag heatmap":
            from scipy.interpolate import griddata

            grid_res = 0.05
            grid = {}
            for (x, y), f in zip(viewpoints_cpu[:, :2], inspection_flags):
                gx, gy = round(x / grid_res), round(y / grid_res)
                grid[(gx, gy)] = grid.get((gx, gy), 0) + f  # f is already ±1

            xs = np.array([gx * grid_res for gx, gy in grid.keys()])
            ys = np.array([gy * grid_res for gx, gy in grid.keys()])
            vals = np.array(list(grid.values()))

            xi = np.linspace(xs.min(), xs.max(), 200)
            yi = np.linspace(ys.min(), ys.max(), 200)
            Xi, Yi = np.meshgrid(xi, yi)

            try:
                Zi = griddata((xs, ys), vals, (Xi, Yi), method="cubic", fill_value=0)
            except Exception:
                Zi = griddata((xs, ys), vals, (Xi, Yi), method="linear", fill_value=0)

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(Zi, extent=(xs.min(), xs.max(), ys.min(), ys.max()),
                           origin="lower", cmap="bwr", aspect="auto")
            plt.colorbar(im, ax=ax, label="Inspection Flag Sum\n(-1=No Inspect, +1=Inspect)")
            ax.set_xlabel("X (down)")
            ax.set_ylabel("Y (right)")
            ax.set_title("Inspection Flag Heatmap")
            fig.savefig("sampling_test_figures/inspection_flag_heatmap.png")
            plt.close(fig)