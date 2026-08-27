import socket
import struct
import pickle
import cv2


# ============================================================
# SERVER CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = 5000


# ============================================================
# CREATE SERVER SOCKET
# ============================================================

server_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

server_socket.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1
)

server_socket.bind((HOST, PORT))
server_socket.listen(1)


print(f"Server listening on port {PORT}...")
print("Waiting for client...")


# ============================================================
# ACCEPT CLIENT
# ============================================================

conn, addr = server_socket.accept()

print("Client connected:", addr)


# ============================================================
# VARIABLES
# ============================================================

data = b""

payload_size = struct.calcsize("Q")

frame_count = 0


# ============================================================
# RECEIVE LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # RECEIVE MESSAGE SIZE
    # --------------------------------------------------------

    try:

        while len(data) < payload_size:

            packet = conn.recv(4096)

            if not packet:
                raise ConnectionError("Client disconnected.")

            data += packet

    except (ConnectionError, ConnectionResetError):

        print("Client disconnected.")
        break


    # --------------------------------------------------------
    # EXTRACT MESSAGE SIZE
    # --------------------------------------------------------

    packed_msg_size = data[:payload_size]

    data = data[payload_size:]


    msg_size = struct.unpack(
        "Q",
        packed_msg_size
    )[0]


    # --------------------------------------------------------
    # RECEIVE FRAME DATA
    # --------------------------------------------------------

    try:

        while len(data) < msg_size:

            packet = conn.recv(4096)

            if not packet:
                raise ConnectionError("Client disconnected.")

            data += packet

    except (ConnectionError, ConnectionResetError):

        print("Client disconnected.")
        break


    # --------------------------------------------------------
    # EXTRACT FRAME
    # --------------------------------------------------------

    frame_data = data[:msg_size]

    data = data[msg_size:]


    # --------------------------------------------------------
    # DECODE FRAME
    # --------------------------------------------------------

    try:

        frame = pickle.loads(frame_data)

    except Exception as e:

        print("Could not decode frame:", e)
        continue


    # --------------------------------------------------------
    # COUNT FRAME
    # --------------------------------------------------------

    frame_count += 1


    # --------------------------------------------------------
    # DISPLAY VIDEO
    # --------------------------------------------------------

    cv2.imshow(
        "SERVER - Received Video",
        frame
    )


    # --------------------------------------------------------
    # SEND ACK TO CLIENT
    # --------------------------------------------------------

    try:

        conn.sendall(b"ACK")

    except (ConnectionError, ConnectionResetError):

        print("Client disconnected while sending ACK.")
        break


    # --------------------------------------------------------
    # PRINT FRAME COUNT
    # --------------------------------------------------------

    if frame_count % 10 == 0:

        print(
            f"Frames received: {frame_count}"
        )


    # --------------------------------------------------------
    # CHECK FOR Q
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("Q pressed. Stopping server.")
        break


# ============================================================
# CLEANUP
# ============================================================

try:
    conn.close()
except:
    pass

try:
    server_socket.close()
except:
    pass

cv2.destroyAllWindows()


print()
print("==============================")
print("SERVER STOPPED")
print("==============================")
print(
    f"Total frames received: {frame_count}"
)