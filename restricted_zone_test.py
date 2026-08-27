import cv2
import time
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------

CONFIDENCE = 0.60
LOITERING_TIME = 10

# -----------------------------
# YOLO
# -----------------------------

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO loaded.")

# -----------------------------
# CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opened")
    exit()

print("Camera opened.")
print("Put your PERSON CENTER inside the RED BOX.")
print("Press Q to quit.")

# -----------------------------
# TIMER
# -----------------------------

zone_enter_time = None
current_id = None

# -----------------------------
# LOOP
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # ---------------------------------
    # Restricted zone
    # ---------------------------------

    zx1 = int(width * 0.25)
    zy1 = int(height * 0.20)

    zx2 = int(width * 0.75)
    zy2 = int(height * 0.80)

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

    # ---------------------------------
    # YOLO TRACKING
    # ---------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    person_found = False
    person_inside = False

    # ---------------------------------
    # PROCESS PERSON
    # ---------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Person center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            person_found = True

            # ---------------------------------
            # CHECK ZONE
            # ---------------------------------

            inside = (
                zx1 <= cx <= zx2
                and
                zy1 <= cy <= zy2
            )

            # ---------------------------------
            # DRAW PERSON
            # ---------------------------------

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
                7,
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

            # ---------------------------------
            # INSIDE ZONE
            # ---------------------------------

            if inside:

                person_inside = True

                # Start timer
                if zone_enter_time is None:

                    zone_enter_time = time.time()
                    current_id = track_id

                    print(
                        f"ID {track_id} entered zone"
                    )

                # Calculate timer
                elapsed = time.time() - zone_enter_time

                # ---------------------------------
                # SHOW TIMER
                # ---------------------------------

                cv2.putText(
                    frame,
                    f"ZONE TIME: {elapsed:.1f} s",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # ---------------------------------
                # LOITERING
                # ---------------------------------

                if elapsed >= LOITERING_TIME:

                    cv2.putText(
                        frame,
                        "LOITERING",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "Activity: LOITERING",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                else:

                    cv2.putText(
                        frame,
                        "Activity: IN RESTRICTED ZONE",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

            # ---------------------------------
            # OUTSIDE ZONE
            # ---------------------------------

            else:

                cv2.putText(
                    frame,
                    "Activity: PERSON DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    # ---------------------------------
    # NO PERSON
    # ---------------------------------

    if not person_found:

        cv2.putText(
            frame,
            "Activity: NO PERSON",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # ---------------------------------
    # SHOW
    # ---------------------------------

    cv2.imshow(
        "Restricted Zone Test",
        frame
    )

    # ---------------------------------
    # QUIT
    # ---------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")