import os
import csv
from tensorboard.backend.event_processing import event_accumulator

# === CONFIG ===
log_dir = "/home/ltoschi/Documents/code/rl_quadrupeds/logs/skrl/go1_locomotion/test"        # Folder containing .tfevents file
output_csv = "/home/ltoschi/Documents/code/wtw/contact_states/wtw_contact_states.csv"  # Output CSV file
scalar_tags = [                         # Tags you want to extract
    "desired_contact_states/foot_indices/FR",
    "desired_contact_states/foot_indices/FL",
    "desired_contact_states/foot_indices/RR",
    "desired_contact_states/foot_indices/RL",
    "desired_contact_states/clock_inputs/FL",
    "desired_contact_states/clock_inputs/RR",
    "desired_contact_states/clock_inputs/RL",
    "desired_contact_states/clock_inputs/FR",
    "desired_contact_states/desired_contact_states/FR",
    "desired_contact_states/desired_contact_states/FL",
    "desired_contact_states/desired_contact_states/RR",
    "desired_contact_states/desired_contact_states/RL",
]

# === Load TensorBoard log ===
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# === Extract time series for each tag ===
data = {}
for tag in scalar_tags:
    events = ea.Scalars(tag)
    data[tag] = [(e.step, e.value) for e in events]

# === Check all tags have the same steps ===
steps = [step for step, _ in data[scalar_tags[0]]]
for tag in scalar_tags[1:]:
    tag_steps = [step for step, _ in data[tag]]
    if tag_steps != steps:
        raise ValueError(f"Step mismatch in tag: {tag}")

# === Write to CSV ===
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(["global_step"] + scalar_tags)
    
    # Rows
    for i, step in enumerate(steps):
        row = [step] + [data[tag][i][1] for tag in scalar_tags]
        writer.writerow(row)

print(f"✅ Exported {len(steps)} steps to {output_csv}")