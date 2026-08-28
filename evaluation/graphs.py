import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


BASELINE_LOG = "baseline_log.csv"
ADAPTIVE_LOG = "streaming_log.csv"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(BASELINE_LOG):
    print("ERROR: baseline_log.csv not found.")
    exit()

if not os.path.exists(ADAPTIVE_LOG):
    print("ERROR: streaming_log.csv not found.")
    exit()


# ============================================================
# READ BASELINE LOG
# ============================================================

baseline_rows = []

with open(BASELINE_LOG, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row.get("bytes_sent"):
            baseline_rows.append(row)


# ============================================================
# READ ADAPTIVE LOG
# ============================================================

adaptive_rows = []

with open(ADAPTIVE_LOG, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row.get("bytes_sent"):
            adaptive_rows.append(row)


# ============================================================
# CHECK DATA
# ============================================================

if not baseline_rows:
    print("ERROR: baseline_log.csv contains no data.")
    exit()

if not adaptive_rows:
    print("ERROR: streaming_log.csv contains no data.")
    exit()


# ============================================================
# BASIC CALCULATIONS
# ============================================================

baseline_bytes = sum(
    int(row["bytes_sent"])
    for row in baseline_rows
)

adaptive_bytes = sum(
    int(row["bytes_sent"])
    for row in adaptive_rows
)

baseline_frames = len(baseline_rows)
adaptive_frames = len(adaptive_rows)

baseline_average = (
    baseline_bytes / baseline_frames
)

adaptive_average = (
    adaptive_bytes / adaptive_frames
)

saving_percent = (
    (baseline_average - adaptive_average)
    / baseline_average
    * 100
)


# ============================================================
# CREATE GRAPH 1
# BASELINE VS ADAPTIVE
# ============================================================

methods = [
    "Baseline",
    "Adaptive"
]

average_bytes = [
    baseline_average,
    adaptive_average
]

plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    average_bytes
)

plt.title(
    "Baseline vs Adaptive Streaming"
)

plt.xlabel(
    "Streaming Method"
)

plt.ylabel(
    "Average Bytes per Frame"
)

plt.tight_layout()

plt.savefig(
    "graph_baseline_vs_adaptive.png",
    dpi=300
)

plt.show()


# ============================================================
# ADAPTIVE TIER DATA
# ============================================================

tier_frames = defaultdict(int)

for row in adaptive_rows:

    tier = int(row["tier"])

    tier_frames[tier] += 1


tiers = sorted(tier_frames.keys())

tier_values = [
    tier_frames[tier]
    for tier in tiers
]


# ============================================================
# CREATE GRAPH 2
# FRAMES BY TIER
# ============================================================

tier_names = [
    f"Tier {tier}"
    for tier in tiers
]

plt.figure(figsize=(8, 5))

plt.bar(
    tier_names,
    tier_values
)

plt.title(
    "Adaptive Streaming Frames by Quality Tier"
)

plt.xlabel(
    "Quality Tier"
)

plt.ylabel(
    "Number of Frames"
)

plt.tight_layout()

plt.savefig(
    "graph_frames_by_tier.png",
    dpi=300
)

plt.show()


# ============================================================
# ACTIVITY DATA
# ============================================================

activity_frames = Counter()

for row in adaptive_rows:

    activity = row["activity"]

    activity_frames[activity] += 1


activities = sorted(
    activity_frames.keys()
)

activity_values = [
    activity_frames[activity]
    for activity in activities
]


# ============================================================
# CREATE GRAPH 3
# ACTIVITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    activities,
    activity_values
)

plt.title(
    "Adaptive Streaming Activity Distribution"
)

plt.xlabel(
    "Activity"
)

plt.ylabel(
    "Number of Frames"
)

plt.tight_layout()

plt.savefig(
    "graph_activity_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# RTT DATA
# ============================================================

rtt_values = []

for row in adaptive_rows:

    try:

        rtt = float(
            row.get("rtt_ms", 0)
        )

        if rtt > 0:
            rtt_values.append(rtt)

    except ValueError:

        pass


# ============================================================
# CREATE GRAPH 4
# RTT
# ============================================================

if rtt_values:

    frame_numbers = list(
        range(
            1,
            len(rtt_values) + 1
        )
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        frame_numbers,
        rtt_values
    )

    plt.title(
        "Network RTT During Adaptive Streaming"
    )

    plt.xlabel(
        "Frame Number"
    )

    plt.ylabel(
        "RTT (ms)"
    )

    plt.tight_layout()

    plt.savefig(
        "graph_rtt.png",
        dpi=300
    )

    plt.show()


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==============================")
print("GRAPHS CREATED")
print("==============================")

print(
    f"Baseline average : "
    f"{baseline_average:,.0f} bytes/frame"
)

print(
    f"Adaptive average : "
    f"{adaptive_average:,.0f} bytes/frame"
)

print(
    f"Bandwidth saving : "
    f"{saving_percent:.2f}%"
)

print()
print("Created files:")

print(
    "graph_baseline_vs_adaptive.png"
)

print(
    "graph_frames_by_tier.png"
)

print(
    "graph_activity_distribution.png"
)

if rtt_values:

    print(
        "graph_rtt.png"
    )

print()