import csv
import os
from collections import defaultdict


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

if len(baseline_rows) == 0:
    print("ERROR: baseline_log.csv contains no data.")
    exit()

if len(adaptive_rows) == 0:
    print("ERROR: streaming_log.csv contains no data.")
    exit()


# ============================================================
# BASELINE CALCULATION
# ============================================================

baseline_frames = len(baseline_rows)

baseline_bytes = 0

for row in baseline_rows:
    baseline_bytes += int(row["bytes_sent"])

baseline_average = baseline_bytes / baseline_frames


# ============================================================
# ADAPTIVE CALCULATION
# ============================================================

adaptive_frames = len(adaptive_rows)

adaptive_bytes = 0

for row in adaptive_rows:
    adaptive_bytes += int(row["bytes_sent"])

adaptive_average = adaptive_bytes / adaptive_frames


# ============================================================
# BANDWIDTH SAVING
# ============================================================

saving_per_frame = (
    baseline_average
    - adaptive_average
)

saving_percent = (
    saving_per_frame
    / baseline_average
    * 100
)


estimated_baseline = (
    baseline_average
    * adaptive_frames
)

estimated_saving = (
    estimated_baseline
    - adaptive_bytes
)

estimated_saving_percent = (
    estimated_saving
    / estimated_baseline
    * 100
)


# ============================================================
# ADAPTIVE DATA BY TIER
# ============================================================

tier_data = defaultdict(
    lambda: {
        "frames": 0,
        "bytes": 0
    }
)


for row in adaptive_rows:

    tier = int(row["tier"])

    bytes_sent = int(
        row["bytes_sent"]
    )

    tier_data[tier]["frames"] += 1

    tier_data[tier]["bytes"] += bytes_sent


# ============================================================
# ADAPTIVE DATA BY ACTIVITY
# ============================================================

activity_data = defaultdict(
    lambda: {
        "frames": 0,
        "bytes": 0
    }
)


for row in adaptive_rows:

    activity = row["activity"]

    bytes_sent = int(
        row["bytes_sent"]
    )

    activity_data[activity]["frames"] += 1

    activity_data[activity]["bytes"] += bytes_sent


# ============================================================
# NETWORK DATA
# ============================================================

rtt_values = []

network_data = defaultdict(int)


for row in adaptive_rows:

    try:

        rtt = float(
            row.get(
                "rtt_ms",
                ""
            )
        )

        if rtt > 0:
            rtt_values.append(rtt)

    except ValueError:
        pass


    quality = row.get(
        "network_quality",
        "UNKNOWN"
    )

    if not quality:
        quality = "UNKNOWN"

    network_data[quality] += 1


if len(rtt_values) > 0:

    average_rtt = (
        sum(rtt_values)
        / len(rtt_values)
    )

    minimum_rtt = min(
        rtt_values
    )

    maximum_rtt = max(
        rtt_values
    )

else:

    average_rtt = 0

    minimum_rtt = 0

    maximum_rtt = 0


# ============================================================
# REPORT
# ============================================================

print()

print("==============================")

print(
    "BASELINE VS ADAPTIVE EVALUATION"
)

print("==============================")


# ============================================================
# BASELINE
# ============================================================

print()

print("BASELINE")

print("------------------------------")

print(
    f"Frames recorded       : "
    f"{baseline_frames}"
)

print(
    f"Total bytes sent      : "
    f"{baseline_bytes:,}"
)

print(
    f"Total MB sent         : "
    f"{baseline_bytes / (1024 * 1024):.2f}"
)

print(
    f"Average bytes/frame   : "
    f"{baseline_average:,.0f}"
)


# ============================================================
# ADAPTIVE
# ============================================================

print()

print("ADAPTIVE")

print("------------------------------")

print(
    f"Frames recorded       : "
    f"{adaptive_frames}"
)

print(
    f"Total bytes sent      : "
    f"{adaptive_bytes:,}"
)

print(
    f"Total MB sent         : "
    f"{adaptive_bytes / (1024 * 1024):.2f}"
)

print(
    f"Average bytes/frame   : "
    f"{adaptive_average:,.0f}"
)


# ============================================================
# TIER DATA
# ============================================================

print()

print("ADAPTIVE DATA BY TIER")

print("------------------------------")


for tier in sorted(tier_data):

    frames = (
        tier_data[tier]["frames"]
    )

    bytes_sent = (
        tier_data[tier]["bytes"]
    )

    average = (
        bytes_sent
        / frames
    )

    print(
        f"Tier {tier}: "
        f"{frames} frames, "
        f"{bytes_sent:,} bytes, "
        f"{average:,.0f} bytes/frame"
    )


# ============================================================
# ACTIVITY DATA
# ============================================================

print()

print("ADAPTIVE DATA BY ACTIVITY")

print("------------------------------")


for activity in sorted(
    activity_data
):

    frames = (
        activity_data[activity]["frames"]
    )

    bytes_sent = (
        activity_data[activity]["bytes"]
    )

    print(
        f"{activity}: "
        f"{frames} frames, "
        f"{bytes_sent:,} bytes"
    )


# ============================================================
# NETWORK PERFORMANCE
# ============================================================

print()

print("NETWORK PERFORMANCE")

print("------------------------------")

print(
    f"Average RTT : "
    f"{average_rtt:.2f} ms"
)

print(
    f"Minimum RTT : "
    f"{minimum_rtt:.2f} ms"
)

print(
    f"Maximum RTT : "
    f"{maximum_rtt:.2f} ms"
)


for quality in sorted(
    network_data
):

    count = network_data[
        quality
    ]

    percentage = (
        count
        / adaptive_frames
        * 100
    )

    print(
        f"{quality:<9}: "
        f"{count} frames "
        f"({percentage:.2f}%)"
    )


# ============================================================
# BANDWIDTH COMPARISON
# ============================================================

print()

print("BANDWIDTH COMPARISON")

print("------------------------------")

print(
    f"Baseline average/frame : "
    f"{baseline_average:,.0f} bytes"
)

print(
    f"Adaptive average/frame : "
    f"{adaptive_average:,.0f} bytes"
)

print(
    f"Saving per frame       : "
    f"{saving_per_frame:,.0f} bytes"
)

print(
    f"Bandwidth reduction    : "
    f"{saving_percent:.2f}%"
)


# ============================================================
# ADAPTIVE RUN ESTIMATE
# ============================================================

print()

print("ADAPTIVE-RUN ESTIMATE")

print("------------------------------")

print(
    f"Baseline-equivalent data : "
    f"{estimated_baseline:,.0f} bytes"
)

print(
    f"Adaptive data            : "
    f"{adaptive_bytes:,} bytes"
)

print(
    f"Estimated saving         : "
    f"{estimated_saving:,.0f} bytes"
)

print(
    f"Estimated saving         : "
    f"{estimated_saving_percent:.2f}%"
)


# ============================================================
# COMPLETE
# ============================================================

print()

print("==============================")

print("EVALUATION COMPLETE")

print("==============================")

print()