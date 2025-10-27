import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import warnings

def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x)
    return None

def collect_numeric(d, parent_key=""):
    """Recursively collect all numeric tensors/arrays from nested dicts."""
    metrics = {}
    for k, v in d.items():
        full_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            metrics.update(collect_numeric(v, full_key))
        else:
            arr = tensor_to_numpy(v)
            if arr is not None:
                metrics[full_key] = arr
    return metrics

def center_crop(arr, num_steps, key=""):
    """Extract the central `num_steps` from an array along axis 0."""
    length = arr.shape[0]
    if length < num_steps:
        warnings.warn(f"[{key}] Episode shorter ({length}) than requested num_steps ({num_steps}). Using full data.")
        return arr
    start = (length - num_steps) // 2
    end = start + num_steps
    return arr[start:end]

def plot_debug_recorders(file_path, output_dir="plots", last_n=3, num_steps=None):
    """
    Reads the last N episodes from an HDF5 dataset and organizes
    data into a dictionary of {recorder_term: {episode_name: metrics}}.

    Each episode can be trimmed to the central `num_steps` steps.
    """
    os.makedirs(output_dir, exist_ok=True)

    handler = HDF5DatasetFileHandler()
    handler.open(file_path, "r")

    # Get only the N most recent episodes
    episode_names_rev_sorted = sorted(list(handler.get_episode_names()), reverse=True)

    # --- Collect all metrics grouped by recorder term ---
    term_episode_data = {}
    for ep_name in episode_names_rev_sorted:
        ep_data = handler.load_episode(ep_name, "cpu")._data
        metrics = collect_numeric(ep_data)
        if metrics[list(metrics.keys())[0]].shape[0] < 2:
            continue  # skip episodes with less than 2 data points

        # Apply central cropping if requested
        cropped_metrics = {}
        for key, arr in metrics.items():
            if num_steps is not None:
                arr = center_crop(arr, num_steps, key)
            cropped_metrics[key] = arr

        for key, arr in cropped_metrics.items():
            # Recorder term = first part of the key (e.g. "debug/rewards/foot_dev" → "debug")
            term = key.split("/")[0]
            term_episode_data.setdefault(term, {}).setdefault(ep_name, {})[key] = arr

        if len(list(term_episode_data[term].keys())) == last_n:
            break

    # Print summary
    for term, ep_dict in term_episode_data.items():
        print(f"Recorder Term: {term}")
        for ep_name, data in ep_dict.items():
            print(f"  Episode: {ep_name}")
            for data_key, values in data.items():
                print(f"    {data_key}: shape {values.shape}")
            break

    data_per_recorder = {}
    for term, ep_dict in term_episode_data.items():
        for ep_name, data in ep_dict.items():
            for metric_name, values in data.items():
                parts = metric_name.split("/")
                if len(parts) < 4:
                    print(f"Skipping malformed key: {metric_name}")
                    continue
                recorder_type, recorder_name, property_name = parts[1], parts[2], parts[3]

                if recorder_type not in data_per_recorder:
                    data_per_recorder[recorder_type] = {}
                    os.makedirs(os.path.join(output_dir, recorder_type), exist_ok=True)

                if recorder_name not in data_per_recorder[recorder_type]:
                    data_per_recorder[recorder_type][recorder_name] = {}
                    os.makedirs(os.path.join(output_dir, recorder_type, recorder_name), exist_ok=True)

                if property_name not in data_per_recorder[recorder_type][recorder_name]:
                    data_per_recorder[recorder_type][recorder_name][property_name] = []
                data_per_recorder[recorder_type][recorder_name][property_name].append((ep_name, values))

    # --- Plotting ---
    for recorder_type, recorder_dict in data_per_recorder.items():
        for recorder_name, property_dict in recorder_dict.items():
            for property_name, ep_data_list in property_dict.items():
                sample_shape = ep_data_list[0][1].shape

                if len(sample_shape) == 1:
                    # --- Scalar time series ---
                    plt.figure(figsize=(10, 6))
                    for ep_name, values in ep_data_list:
                        plt.plot(values, label=ep_name)
                    plt.title(f"{recorder_type} - {recorder_name} - {property_name}")
                    plt.xlabel("Time Step")
                    plt.ylabel(property_name)
                    plt.legend()
                    plt.grid(True)

                    plot_path = os.path.join(output_dir, recorder_type, recorder_name, f"{property_name}.png")
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved plot: {plot_path}")

                elif len(sample_shape) == 2:
                    num_cols = sample_shape[1]

                    # --- Feet data (4 legs) ---
                    if num_cols == 4:
                        for foot_idx in range(4):
                            plt.figure(figsize=(10, 6))
                            for ep_name, values in ep_data_list:
                                plt.plot(values[:, foot_idx], label=ep_name)
                            plt.title(f"{recorder_type} - {recorder_name} - {property_name} (Foot {foot_idx})")
                            plt.xlabel("Time Step")
                            plt.ylabel(property_name)
                            plt.legend()
                            plt.grid(True)

                            plot_path = os.path.join(output_dir, recorder_type, recorder_name, f"{property_name}_foot{foot_idx}.png")
                            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                            plt.savefig(plot_path)
                            plt.close()
                            print(f"Saved plot: {plot_path}")

                    # --- Joint data (12 joints) ---
                    elif num_cols == 12:
                        for joint_idx in range(12):
                            plt.figure(figsize=(10, 6))
                            for ep_name, values in ep_data_list:
                                plt.plot(values[:, joint_idx], label=ep_name)
                            plt.title(f"{recorder_type} - {recorder_name} - {property_name} (Joint {joint_idx})")
                            plt.xlabel("Time Step")
                            plt.ylabel(property_name)
                            plt.legend()
                            plt.grid(True)

                            plot_path = os.path.join(output_dir, recorder_type, recorder_name, f"{property_name}_joint{joint_idx}.png")
                            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                            plt.savefig(plot_path)
                            plt.close()
                            print(f"Saved plot: {plot_path}")

                    # --- Other multi-dimensional data ---
                    else:
                        print(f"Skipping plotting for {recorder_type} - {recorder_name} - {property_name} with shape {sample_shape}")
    return term_episode_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--out", type=str, default="plots", help="Output folder")
    parser.add_argument("--last_n", type=int, default=3, help="Number of last episodes to load")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of central steps per episode")
    args = parser.parse_args()

    term_episode_data = plot_debug_recorders(args.file, output_dir=args.out, last_n=args.last_n, num_steps=args.num_steps)

    # Example access:
    # data = term_episode_data["debug/rewards/foot_dev"]["episode_001"]
    # print(list(data.keys()))  # show metrics available for this recorder