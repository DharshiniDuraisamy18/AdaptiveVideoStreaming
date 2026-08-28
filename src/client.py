import socket
import struct
import pickle
import cv2
import time


# ============================================================
# NETWORK
# ============================================================

SERVER_IP = "127.0.0.1"
PORT = 5000


# ============================================================
# STREAM SETTINGS
# ============================================================

IDLE_WIDTH = 320
IDLE_HEIGHT = 240
IDLE_FPS = 5
IDLE_JPEG_QUALITY = 40

HIGH_WIDTH = 640
HIGH_HEIGHT = 480
HIGH_FPS = 15
HIGH_JPEG_QUALITY = 70


# ============================================================
# MOTION SETTINGS
# ============================================================

MOTION_THRESHOLD = 1.5


# ============================================================
# CONNECT TO SERVER
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

    print("ERROR: Cannot open webcam.")

    client_socket.close()

    exit()


print("Webcam opened.")
print("Move your hand/body to test motion.")
print("Press Q to quit.")


# ============================================================
# VARIABLES
# ============================================================

previous_gray = None

mode = "IDLE"

last_send_time = 0


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
    # MOTION DECISION
    # ========================================================

    if motion_value > MOTION_THRESHOLD:

        mode = "HIGH_TRAFFIC"

    else:

        mode = "IDLE"


    # ========================================================
    # SELECT STREAM SETTINGS
    # ========================================================

    if mode == "HIGH_TRAFFIC":

        target_width = HIGH_WIDTH
        target_height = HIGH_HEIGHT
        target_fps = HIGH_FPS
        jpeg_quality = HIGH_JPEG_QUALITY

    else:

        target_width = IDLE_WIDTH
        target_height = IDLE_HEIGHT
        target_fps = IDLE_FPS
        jpeg_quality = IDLE_JPEG_QUALITY


    # ========================================================
    # SEND FRAME AT TARGET FPS
    # ========================================================

    current_time = time.time()

    frame_interval = 1.0 / target_fps


    if current_time - last_send_time >= frame_interval:

        last_send_time = current_time


        # Resize
        stream_frame = cv2.resize(
            frame,
            (
                target_width,
                target_height
            )
        )


        # JPEG encode
        success, encoded = cv2.imencode(
            ".jpg",
            stream_frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality
            ]
        )


        if success:

            data = pickle.dumps(
                encoded,
                protocol=pickle.HIGHEST_PROTOCOL
            )


            message = (
                struct.pack(
                    "Q",
                    len(data)
                )
                +
                data
            )


            try:

                client_socket.sendall(
                    message
                )

            except ConnectionError:

                print("ERROR: Server disconnected.")

                break


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"MOTION VALUE: {motion_value:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"THRESHOLD: {MOTION_THRESHOLD:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"STREAM: {target_width}x{target_height} "
        f"{target_fps} FPS",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "CLIENT - Webcam",
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

print("Client stopped.")