import cv2
import time
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0

# Motion threshold
MOTION_THRESHOLD = 1000

# YOLO confidence
YOLO_CONFIDENCE = 0.60


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO loaded.")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


# ============================================================
# VARIABLES
# ============================================================

previous_frame = None

# Stores start time for each tracked person
track_start_times = {}


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read webcam
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read webcam frame.")
        break


    # ========================================================
    # MOTION DETECTION
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    if previous_frame is None:

        previous_frame = gray

        continue


    difference = cv2.absdiff(
        previous_frame,
        gray
    )

    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    motion_area = cv2.countNonZero(
        threshold
    )


    if motion_area >= MOTION_THRESHOLD:

        motion_detected = True

    else:

        motion_detected = False


    # ========================================================
    # ACTIVITY
    # ========================================================

    activity = "IDLE"


    if motion_detected:

        activity = "MOTION"


    # ========================================================
    # YOLO TRACKING
    #
    # IMPORTANT:
    # We run tracking continuously so that a person can
    # remain tracked even when they stop moving.
    # ========================================================

    results = model.track(
        frame,
        conf=YOLO_CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],          # Person only
        verbose=False
    )


    person_detected = False


    # ========================================================
    # PROCESS TRACKING RESULTS
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # Tracking ID
            # ------------------------------------------------

            if box.id is None:
                continue


            track_id = int(
                box.id[0]
            )


            # ------------------------------------------------
            # Person detected
            # ------------------------------------------------

            person_detected = True


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # Start timer for new track
            # ------------------------------------------------

            if track_id not in track_start_times:

                track_start_times[track_id] = time.time()


            # ------------------------------------------------
            # Calculate tracking duration
            # ------------------------------------------------

            elapsed = (
                time.time()
                - track_start_times[track_id]
            )


            # ------------------------------------------------
            # Activity
            # ------------------------------------------------

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
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # TIMER
            # ------------------------------------------------

            timer_text = (
                f"Tracked: {elapsed:.1f}s"
            )


            cv2.putText(
                frame,
                timer_text,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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


    # ========================================================
    # DISPLAY MOTION AREA
    # ========================================================

    cv2.putText(
        frame,
        f"Motion Area: {motion_area}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Stage 1 - Motion + YOLO + Tracking",
        frame
    )


    # ========================================================
    # UPDATE PREVIOUS FRAME
    # ========================================================

    previous_frame = gray


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

print("Program stopped.")