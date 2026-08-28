import cv2
import time
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.60

# For testing, use 10 seconds.
# Later we can change this to the project's final loitering
# duration after we finalize the specification.
LOITERING_TIME = 10


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO loaded.")


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened.")
print("Stand in front of the camera.")
print("Press Q to quit.")


# ============================================================
# TRACK START TIMES
# ============================================================

track_start_times = {}


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break


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


    # ========================================================
    # PROCESS PERSONS
    # ========================================================

    current_track_ids = set()

    activity = "IDLE"


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            if box.id is None:
                continue


            # ------------------------------------------------
            # Tracking ID
            # ------------------------------------------------

            track_id = int(box.id[0])

            current_track_ids.add(track_id)


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(box.conf[0])


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # Start timer for new ID
            # ------------------------------------------------

            if track_id not in track_start_times:

                track_start_times[track_id] = time.time()

                print(
                    f"Person ID {track_id} entered."
                )


            # ------------------------------------------------
            # Calculate duration
            # ------------------------------------------------

            elapsed = (
                time.time()
                - track_start_times[track_id]
            )


            # ------------------------------------------------
            # Activity classification
            # ------------------------------------------------

            if elapsed >= LOITERING_TIME:

                activity = "LOITERING"

            else:

                activity = "PERSON DETECTED"


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # Person label
            # ------------------------------------------------

            label = (
                f"ID {track_id} "
                f"Person {confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # Timer
            # ------------------------------------------------

            timer_text = (
                f"Time: {elapsed:.1f}s"
            )

            cv2.putText(
                frame,
                timer_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # Loitering label
            # ------------------------------------------------

            if elapsed >= LOITERING_TIME:

                cv2.putText(
                    frame,
                    "LOITERING",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


    # ========================================================
    # DISPLAY ACTIVITY
    # ========================================================

    cv2.putText(
        frame,
        f"Activity: {activity}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Loitering threshold: {LOITERING_TIME}s",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "Loitering Test",
        frame
    )


    # ========================================================
    # REMOVE OLD TRACKS
    # ========================================================

    # We don't immediately delete an ID here because
    # ByteTrack can temporarily lose a person.
    #
    # We'll make this more robust later.


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

print("Loitering test stopped.")