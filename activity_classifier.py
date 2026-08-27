import cv2
import time
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.50

# Restricted zone
ZONE_X1 = 150
ZONE_Y1 = 100
ZONE_X2 = 500
ZONE_Y2 = 400

# Testing value
# Later we can change this to the project's final value.
LOITERING_TIME = 10.0

# Motion threshold
MOTION_THRESHOLD = 1.5


# ============================================================
# TIER CONTROLLER
# ============================================================

def get_tier(activity):

    if activity == "IDLE":

        return {
            "tier": 0,
            "width": 320,
            "height": 240,
            "fps": 5,
            "jpeg_quality": 40
        }

    elif activity == "MOTION":

        return {
            "tier": 1,
            "width": 480,
            "height": 360,
            "fps": 8,
            "jpeg_quality": 50
        }

    elif activity == "PERSON":

        return {
            "tier": 2,
            "width": 640,
            "height": 480,
            "fps": 10,
            "jpeg_quality": 60
        }

    elif activity == "LOITERING":

        return {
            "tier": 3,
            "width": 640,
            "height": 480,
            "fps": 15,
            "jpeg_quality": 70
        }

    else:

        return {
            "tier": 0,
            "width": 320,
            "height": 240,
            "fps": 5,
            "jpeg_quality": 40
        }


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Cannot open webcam.")

    exit()


print("Webcam opened.")
print("Move around to test activity.")
print("Stay inside the red zone for 10 seconds to test loitering.")
print("Press Q to quit.")


# ============================================================
# TRACK TIMERS
# ============================================================

zone_start_times = {}


# ============================================================
# MOTION VARIABLES
# ============================================================

previous_gray = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame.")

        break


    # ========================================================
    # MOTION DETECTION
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    small_gray = cv2.resize(
        gray,
        (160, 120)
    )

    small_gray = cv2.GaussianBlur(
        small_gray,
        (5, 5),
        0
    )


    if previous_gray is None:

        previous_gray = small_gray.copy()

        motion_value = 0.0

    else:

        difference = cv2.absdiff(
            previous_gray,
            small_gray
        )

        motion_value = float(
            difference.mean()
        )

        previous_gray = small_gray.copy()


    # ========================================================
    # DEFAULT ACTIVITY
    # ========================================================

    activity = "IDLE"


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        verbose=False
    )


    person_found = False

    loitering_found = False


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if results and results[0].boxes is not None:

        boxes = results[0].boxes


        if boxes.xyxy is not None:

            xyxy = boxes.xyxy.cpu().numpy()


            # ------------------------------------------------
            # TRACK IDs
            # ------------------------------------------------

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = [
                    None
                ] * len(xyxy)


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            if boxes.conf is not None:

                confidences = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

            else:

                confidences = [
                    0.0
                ] * len(xyxy)


            # =================================================
            # PROCESS EACH PERSON
            # =================================================

            for box, track_id, confidence in zip(
                xyxy,
                track_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # ------------------------------------------------
                # PERSON CENTER
                # ------------------------------------------------

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                person_found = True


                # ------------------------------------------------
                # CHECK RESTRICTED ZONE
                # ------------------------------------------------

                inside_zone = (

                    ZONE_X1 <= center_x <= ZONE_X2

                    and

                    ZONE_Y1 <= center_y <= ZONE_Y2

                )


                zone_time = 0.0


                # ------------------------------------------------
                # START / UPDATE TIMER
                # ------------------------------------------------

                if inside_zone and track_id is not None:

                    if track_id not in zone_start_times:

                        zone_start_times[
                            track_id
                        ] = time.time()


                    zone_time = (
                        time.time()
                        -
                        zone_start_times[track_id]
                    )


                    # --------------------------------------------
                    # LOITERING
                    # --------------------------------------------

                    if zone_time >= LOITERING_TIME:

                        loitering_found = True


                else:

                    # Person left zone
                    if track_id in zone_start_times:

                        del zone_start_times[
                            track_id
                        ]


                # =================================================
                # DRAW PERSON BOX
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # =================================================
                # PERSON LABEL
                # =================================================

                label = (
                    f"ID:{track_id} "
                    f"Conf:{confidence:.2f}"
                )


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


                # =================================================
                # ZONE TIMER
                # =================================================

                if inside_zone and track_id is not None:

                    timer_text = (
                        f"Zone: {zone_time:.1f}s"
                    )


                    # Put timer above person box
                    timer_y = max(
                        y1 - 35,
                        45
                    )


                    cv2.putText(
                        frame,
                        timer_text,
                        (x1, timer_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )


    # ========================================================
    # ACTIVITY CLASSIFICATION
    # ========================================================

    if loitering_found:

        activity = "LOITERING"

    elif person_found:

        activity = "PERSON"

    elif motion_value > MOTION_THRESHOLD:

        activity = "MOTION"

    else:

        activity = "IDLE"


    # ========================================================
    # GET STREAMING TIER
    # ========================================================

    tier_settings = get_tier(
        activity
    )


    tier = tier_settings["tier"]

    target_width = tier_settings["width"]

    target_height = tier_settings["height"]

    target_fps = tier_settings["fps"]

    jpeg_quality = tier_settings["jpeg_quality"]


    # ========================================================
    # DRAW RESTRICTED ZONE
    # ========================================================

    cv2.rectangle(
        frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        2
    )


    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )


    # ========================================================
    # DISPLAY ACTIVITY
    # ========================================================

    cv2.putText(
        frame,
        f"ACTIVITY: {activity}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY MOTION
    # ========================================================

    cv2.putText(
        frame,
        f"Motion: {motion_value:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY TIER
    # ========================================================

    cv2.putText(
        frame,
        f"TIER: {tier}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY STREAM SETTINGS
    # ========================================================

    cv2.putText(
        frame,
        f"STREAM: {target_width}x{target_height} "
        f"{target_fps} FPS",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY LOITERING THRESHOLD
    # ========================================================

    cv2.putText(
        frame,
        "Loitering: 10 sec",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Activity Classifier",
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

print("Activity classifier stopped.")