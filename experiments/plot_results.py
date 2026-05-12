import json
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# -----------------------------------
# LOAD Q-LEARNING RESULTS
# -----------------------------------

with open(
    "results/results_qlearning_v2_explored.json",
    "r"
) as f:

    qlearning_results = json.load(f)

# -----------------------------------
# LOAD PPO RESULTS
# -----------------------------------

with open(
    "results/results_ppo.json",
    "r"
) as f:

    ppo_results = json.load(f)

# -----------------------------------
# EXTRACT METRICS
# -----------------------------------

baseline_wait_time = 2.14

qlearning_wait_time = float(
    qlearning_results[
        "average_waiting_time_last_50"
    ]
)

ppo_wait_time = float(
    ppo_results[
        "average_waiting_time_last_50"
    ]
)

# -----------------------------------
# IMPROVEMENTS
# -----------------------------------

qlearning_improvement = (
    (
        baseline_wait_time
        - qlearning_wait_time
    )
    / baseline_wait_time
) * 100

ppo_improvement = (
    (
        baseline_wait_time
        - ppo_wait_time
    )
    / baseline_wait_time
) * 100

# -----------------------------------
# PLOT DATA
# -----------------------------------

models = [
    "Nearest-Taxi\nBaseline",
    "Q-Learning",
    "PPO"
]

wait_times = [
    baseline_wait_time,
    qlearning_wait_time,
    ppo_wait_time
]

colors = [
    "steelblue",
    "darkorange",
    "green"
]

# -----------------------------------
# CREATE FIGURE
# -----------------------------------

plt.figure(figsize=(10, 6))

bars = plt.bar(
    models,
    wait_times,
    color=colors,
    width=0.6
)

# -----------------------------------
# LABELS
# -----------------------------------

plt.ylabel(
    "Average Waiting Time",
    fontsize=12
)

plt.xlabel(
    "Dispatch Policy",
    fontsize=12
)

plt.title(
    "Baseline vs Q-Learning vs PPO Waiting Time Comparison",
    fontsize=14
)

# -----------------------------------
# VALUE LABELS
# -----------------------------------

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.03,
        f"{height:.2f}",
        ha="center",
        fontsize=11
    )

# -----------------------------------
# GRID
# -----------------------------------

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# -----------------------------------
# IMPROVEMENT TEXT
# -----------------------------------

plt.figtext(
    0.5,
    0.01,
    (
        f"Q-Learning Improvement: "
        f"{qlearning_improvement:.1f}%   |   "
        f"PPO Improvement: "
        f"{ppo_improvement:.1f}%"
    ),
    ha="center",
    fontsize=11
)

# -----------------------------------
# SAVE FIGURE
# -----------------------------------

plt.tight_layout()

plt.savefig(
    "plots/wait_time_comparison.png",
    dpi=300
)

plt.close()

# -----------------------------------
# CONSOLE OUTPUT
# -----------------------------------

print("\n=== Taxi Dispatch Model Comparison ===\n")

print(
    f"Baseline Waiting Time: "
    f"{baseline_wait_time:.2f}"
)

print(
    f"Q-Learning Waiting Time: "
    f"{qlearning_wait_time:.2f}"
)

print(
    f"PPO Waiting Time: "
    f"{ppo_wait_time:.2f}"
)

print(
    f"\nQ-Learning Improvement: "
    f"{qlearning_improvement:.2f}%"
)

print(
    f"PPO Improvement: "
    f"{ppo_improvement:.2f}%"
)

print(
    "\nComparison plot saved to:"
)

print(
    "plots/wait_time_comparison.png"
)