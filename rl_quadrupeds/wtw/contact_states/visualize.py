import pandas as pd
import matplotlib.pyplot as plt

csv_file = "/home/ltoschi/Documents/code/wtw/contact_states/wtw_contact_states.csv"
df = pd.read_csv(csv_file)

# Clean and prepare the data
df['global_step'] = pd.to_numeric(df['global_step'], errors='coerce')
df = df.dropna(subset=['global_step'])
df['global_step'] = df['global_step'].astype(int)
df.set_index('global_step', inplace=True)

# Define a helper to plot with different styles
def plot_with_styles(x, y_dict, title, ylabel, save_path):
    plt.figure(figsize=(12, 8), dpi=300)
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    for i, (label, y) in enumerate(y_dict.items()):
        plt.plot(x, y, label=label,
                 linestyle=styles[i % len(styles)],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 markersize=4, linewidth=1.5)

    plt.title(title)
    plt.xlabel('Global Step')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot 1: Foot Indices
plot_with_styles(
    df.index,
    {
        'FR Foot Indices': df['desired_contact_states/foot_indices/FR'],
        'FL Foot Indices': df['desired_contact_states/foot_indices/FL'],
        'RR Foot Indices': df['desired_contact_states/foot_indices/RR'],
        'RL Foot Indices': df['desired_contact_states/foot_indices/RL'],
    },
    'Foot Indices Over Time',
    'Foot Indices',
    '/home/ltoschi/Documents/code/wtw/contact_states/wtw_contact_states_plot.png'
)

# Plot 2: Clock Inputs
plot_with_styles(
    df.index,
    {
        'FR Clock Inputs': df['desired_contact_states/clock_inputs/FR'],
        'FL Clock Inputs': df['desired_contact_states/clock_inputs/FL'],
        'RR Clock Inputs': df['desired_contact_states/clock_inputs/RR'],
        'RL Clock Inputs': df['desired_contact_states/clock_inputs/RL'],
    },
    'Clock Inputs Over Time',
    'Clock Inputs',
    '/home/ltoschi/Documents/code/wtw/contact_states/wtw_contact_states_plot_clock_inputs.png'
)

# Plot 3: Desired Contact States
plot_with_styles(
    df.index,
    {
        'FR Desired Contact States': df['desired_contact_states/desired_contact_states/FR'],
        'FL Desired Contact States': df['desired_contact_states/desired_contact_states/FL'],
        'RR Desired Contact States': df['desired_contact_states/desired_contact_states/RR'],
        'RL Desired Contact States': df['desired_contact_states/desired_contact_states/RL'],
    },
    'Desired Contact States Over Time',
    'Desired Contact States',
    '/home/ltoschi/Documents/code/wtw/contact_states/wtw_contact_states_plot_desired_contact_states.png'
)

print("Plots saved successfully.")
