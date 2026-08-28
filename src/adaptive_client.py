import socket
import struct
import pickle
import cv2
import time
from ultralytics import YOLO
from logger import initialize_log, log_event
initialize_log(
    os.path.join(
        PROJECT_ROOT,
        "results",
        "streaming_log.csv"
    )
)
# ============================================================
# NETWORK
# ============================================================

SERVER_IP = "127.0.0.1"
PORT = 5000

# ============================================================
# NETWORK QUALITY MONITOR
# ============================================================

NETWORK_CHECK_INTERVAL = 5.0

GOOD_RTT = 50.0
MODERATE_RTT = 150.0

last_network_check = 0
network_rtt = 0.0
network_quality = "GOOD"


def measure_network_quality(sock):
    """
    Measure RTT using a small message over the existing TCP connection.
    Returns:
        rtt_ms, quality
    """

    try:
        start = time.perf_counter()

        sock.sendall(b"PING")

        response = sock.recv(4)

        end = time.perf_counter()

        if response != b"PONG":
            return network_rtt, network_quality

        rtt_ms = (end - start) * 1000.0

        if rtt_ms <= GOOD_RTT:
            quality = "GOOD"

        elif rtt_ms <= MODERATE_RTT:
            quality = "MODERATE"

        else:
            quality = "POOR"

        return rtt_ms, quality

    except Exception:
        return 999.0, "POOR"
# ============================================================
# YOLO
# ============================================================

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "yolo11n.pt"
)
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# RESTRICTED ZONE
# ============================================================

ZONE_X1 = 150
ZONE_Y1 = 100
ZONE_X2 = 500
ZONE_Y2 = 400

LOITERING_TIME = 10.0


# ============================================================
# MOTION
# ============================================================

MOTION_THRESHOLD = 1.5


# ============================================================
# TIER SETTINGS
# ============================================================

TIERS = {

    0: {
        "width": 320,
        "height": 240,
        "fps": 5,
        "quality": 40
    },

    1: {
        "width": 480,
        "height": 360,
        "fps": 8,
        "quality": 50
    },

    2: {
        "width": 640,
        "height": 480,
        "fps": 10,
        "quality": 60
    },

    3: {
        "width": 640,
        "height": 480,
        "fps": 15,
        "quality": 70
    }
}


# ============================================================
# ACTIVITY → TIER
# ============================================================

def get_tier(activity):

    if activity == "IDLE":
        return 0

    if activity == "MOTION":
        return 1

    if activity == "PERSON":
        return 2

    if activity == "LOITERING":
        return 3

    return 0


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# CONNECT SERVER
# ============================================================

client_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

client_socket.connect(
    (SERVER_IP, PORT)
)

print("Connected to server.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam cannot be opened.")

    client_socket.close()

    exit()


print("Webcam opened.")
print("Press Q to quit.")


# ============================================================
# VARIABLES
# ============================================================

previous_gray = None

zone_start_times = {}

last_send_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ========================================================
    # MOTION
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
    # PROCESS PERSONS
    # ========================================================

    if results and results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.xyxy is not None:

            xyxy = boxes.xyxy.cpu().numpy()


            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = [None] * len(xyxy)


            if boxes.conf is not None:

                confidences = (
                    boxes.conf
                    .cpu()
                    .numpy()
                )

            else:

                confidences = [0.0] * len(xyxy)


            for box, track_id, confidence in zip(
                xyxy,
                track_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                person_found = True


                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # ------------------------------------------------
                # RESTRICTED ZONE
                # ------------------------------------------------

                inside_zone = (

                    ZONE_X1 <= center_x <= ZONE_X2

                    and

                    ZONE_Y1 <= center_y <= ZONE_Y2

                )


                zone_time = 0.0


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


                    if zone_time >= LOITERING_TIME:

                        loitering_found = True


                else:

                    if track_id in zone_start_times:

                        del zone_start_times[
                            track_id
                        ]


                # ------------------------------------------------
                # DRAW PERSON
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"ID:{track_id} Conf:{confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


                if inside_zone and track_id is not None:

                    cv2.putText(
                        frame,
                        f"Zone: {zone_time:.1f}s",
                        (x1, max(y1 - 35, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )


    # ========================================================
    # ACTIVITY
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
    # SELECT TIER
    # ========================================================

    tier = get_tier(activity)
    if network_quality == "POOR":
        tier = max(0, tier - 2)

    elif network_quality == "MODERATE":
        tier = max(0, tier - 1)
    settings = TIERS[tier]

    target_width = settings["width"]
    target_height = settings["height"]
    target_fps = settings["fps"]
    quality = settings["quality"]


    # ========================================================
    # SEND FRAME
    # ========================================================

    current_time = time.time()

    interval = 1.0 / target_fps


    if current_time - last_send_time >= interval:

        last_send_time = current_time


        stream_frame = cv2.resize(
            frame,
            (
                target_width,
                target_height
            )
        )


        success, encoded = cv2.imencode(
            ".jpg",
            stream_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality
            ]
        )
        if success:

            data = pickle.dumps(
                encoded,
                protocol=pickle.HIGHEST_PROTOCOL
            )

            bytes_sent = len(data)

            

            message = (
                struct.pack("Q", len(data))
                + data
            )
               
            try:
                send_start = time.perf_counter()

                client_socket.sendall(message)

                # Wait for server acknowledgement
                ack = client_socket.recv(3)

                if ack == b"ACK":
                    rtt = (time.perf_counter() - send_start) * 1000
                else:
                    rtt = 999.0

                if rtt <= 50:
                    network_quality = "GOOD"
                elif rtt <= 150:
                    network_quality = "MODERATE"
                else:
                    network_quality = "POOR"

                print(
                    f"RTT: {rtt:.2f} ms | "
                    f"Network: {network_quality}"
                )

            except ConnectionError:
                print("Server disconnected.")
                break    
            log_event(
                activity=activity,
                tier=tier,
                width=target_width,
                height=target_height,
                fps=target_fps,
                motion=motion_value,
                bytes_sent=bytes_sent,
                rtt_ms=rtt,
                network_quality=network_quality
            )     
    # ========================================================
    # DISPLAY
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
        f"ACTIVITY: {activity}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"TIER: {tier}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"MOTION: {motion_value:.2f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


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


    cv2.imshow(
        "Adaptive Video Client",
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

client_socket.close()

cv2.destroyAllWindows()

print("Adaptive client stopped.")