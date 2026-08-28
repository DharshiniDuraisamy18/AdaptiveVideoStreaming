import cv2
import time
import csv
import os
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0

CONFIDENCE_THRESHOLD = 0.60

# Test value only
LOITERING_TIME = 10

CSV_FILE = "stage1_results.csv"


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Camera not opened.")
    exit()

print("Camera opened.")
print("Press Q to stop.")


# ============================================================
# RESTRICTED ZONE
# ============================================================

ZONE_X1_PERCENT = 0.25
ZONE_Y1_PERCENT = 0.20

ZONE_X2_PERCENT = 0.75
ZONE_Y2_PERCENT = 0.80


# ============================================================
# TIMER
# ============================================================

zone_start_time = None
active_track_id = None


# ============================================================
# CSV FILE
# ============================================================

file_exists = os.path.exists(CSV_FILE)

csv_file = open(
    CSV_FILE,
    "a",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

if not file_exists:

    writer.writerow([
        "timestamp",
        "track_id",
        "confidence",
        "inside_zone",
        "zone_time_seconds",
        "activity"
    ])


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    height, width = frame.shape[:2]


    # ========================================================
    # RESTRICTED ZONE
    # ========================================================

    zx1 = int(width * ZONE_X1_PERCENT)
    zy1 = int(height * ZONE_Y1_PERCENT)

    zx2 = int(width * ZONE_X2_PERCENT)
    zy2 = int(height * ZONE_Y2_PERCENT)


    cv2.rectangle(
        frame,
        (zx1, zy1),
        (zx2, zy2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (zx1, zy1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    person_found = False
    inside_zone = False

    current_id = None
    current_confidence = 0.0
    zone_time = 0.0
    activity = "NO PERSON"


    # ========================================================
    # PROCESS PERSON
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            if box.id is None:
                continue


            # -----------------------------------------------
            # TRACK ID
            # -----------------------------------------------

            track_id = int(box.id[0])

            current_id = track_id


            # -----------------------------------------------
            # CONFIDENCE
            # -----------------------------------------------

            confidence = float(box.conf[0])

            current_confidence = confidence


            # -----------------------------------------------
            # BOUNDING BOX
            # -----------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # -----------------------------------------------
            # CENTER
            # -----------------------------------------------

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)


            person_found = True


            # -----------------------------------------------
            # CHECK ZONE
            # -----------------------------------------------

            inside_zone = (
                zx1 <= cx <= zx2
                and
                zy1 <= cy <= zy2
            )


            # -----------------------------------------------
            # DRAW PERSON
            # -----------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                6,
                (255, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                f"ID {track_id} Person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # =================================================
            # INSIDE ZONE
            # =================================================

            if inside_zone:

                # Start timer
                if zone_start_time is None:

                    zone_start_time = time.time()
                    active_track_id = track_id


                # If same person
                if active_track_id == track_id:

                    zone_time = (
                        time.time()
                        - zone_start_time
                    )


                if zone_time >= LOITERING_TIME:

                    activity = "LOITERING"

                else:

                    activity = "IN RESTRICTED ZONE"


            # =================================================
            # OUTSIDE ZONE
            # =================================================

            else:

                activity = "PERSON DETECTED"


            break


    # ========================================================
    # NO PERSON
    # ========================================================

    if not person_found:

        activity = "NO PERSON"

        inside_zone = False

        zone_time = 0.0


    # ========================================================
    # DISPLAY ACTIVITY
    # ========================================================

    cv2.putText(
        frame,
        f"Activity: {activity}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY TIMER
    # ========================================================

    if inside_zone:

        cv2.putText(
            frame,
            f"ZONE TIME: {zone_time:.1f} s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )


    # ========================================================
    # WRITE CSV
    # ========================================================

    writer.writerow([
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        current_id,
        round(current_confidence, 3),
        inside_zone,
        round(zone_time, 2),
        activity
    ])

    csv_file.flush()


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Stage 1 Logging",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

csv_file.close()

print("Program stopped.")
print(f"Results saved to: {CSV_FILE}")