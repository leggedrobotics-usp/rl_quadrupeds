import itertools
import os
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalar_data(event_file, tag_prefix="Info / Episode_Reward/"):
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    reward_tags = [tag for tag in ea.Tags()["scalars"] if tag.startswith(tag_prefix)]
    
    data = {}
    for tag in reward_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        tag = tag.replace(tag_prefix, "").replace("_", " ").strip()
        data[tag] = (steps, values)
    
    return data

def plot_reward_data(data):
    plt.figure(figsize=(14, 7))
    
    # Define some line styles and markers to cycle through
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    style_cycler = itertools.cycle(line_styles)
    marker_cycler = itertools.cycle(markers)
    color_cycler = itertools.cycle(color_cycle)

    plt.yscale('log')

    for tag, (steps, values) in data.items():
        values_abs = [abs(v) for v in values]
        
        label = tag.split('/')[-1]
        ls = next(style_cycler)
        marker = next(marker_cycler)
        color = next(color_cycler)
        
        plt.plot(steps, values_abs, label=label, linestyle=ls, marker=marker, color=color, markevery=max(len(steps)//20, 1))
    
    plt.title("Episode Rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward (log scale)")
    plt.legend(ncol=2, fontsize='small', loc='best')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_tensorboard_rewards.py <event_file>")
        sys.exit(1)

    event_file = sys.argv[1]

    if not os.path.exists(event_file):
        print(f"Error: File not found: {event_file}")
        sys.exit(1)

    ea = EventAccumulator(event_file)
    ea.Reload()

    print("Available scalar tags:")
    for tag in ea.Tags()["scalars"]:
        print(tag)


    reward_data = load_scalar_data(event_file)
    if not reward_data:
        print("No 'Info/ Episode_Reward/*' tags found in the event file.")
    else:
        plot_reward_data(reward_data)