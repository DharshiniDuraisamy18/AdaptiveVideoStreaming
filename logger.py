import csv
import os
from datetime import datetime


# ============================================================
# DEFAULT LOG FILE
# ============================================================

LOG_FILE = "streaming_log.csv"


# ============================================================
# INITIALIZE LOG
# ============================================================

def initialize_log(log_file=None):

    global LOG_FILE

    if log_file is not None:
        LOG_FILE = log_file

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "timestamp",
                "activity",
                "tier",
                "width",
                "height",
                "fps",
                "motion",
                "bytes_sent",
                "rtt_ms",
                "network_quality"
            ])


# ============================================================
# LOG EVENT
# ============================================================

def log_event(
    activity,
    tier,
    width,
    height,
    fps,
    motion,
    bytes_sent,
    rtt_ms=0,
    network_quality="UNKNOWN"
):

    with open(
        LOG_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().isoformat(
                timespec="seconds"
            ),
            activity,
            tier,
            width,
            height,
            fps,
            f"{motion:.2f}",
            bytes_sent,
            f"{rtt_ms:.2f}",
            network_quality
        ])