import socket
import time
import statistics


SERVER_IP = "127.0.0.1"
PORT = 6000

PACKET_SIZE = 1024
PACKET_COUNT = 20


def run_probe():

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    sock.connect(
        (SERVER_IP, PORT)
    )

    print("Connected to network probe server.")

    results = []

    for i in range(PACKET_COUNT):

        data = b"x" * PACKET_SIZE

        start = time.perf_counter()

        sock.sendall(data)

        received = 0

        while received < PACKET_SIZE:

            chunk = sock.recv(
                PACKET_SIZE - received
            )

            if not chunk:
                print("Server disconnected.")
                sock.close()
                return

            received += len(chunk)

        end = time.perf_counter()

        rtt_ms = (
            end - start
        ) * 1000

        results.append(rtt_ms)

        print(
            f"Packet {i + 1}/{PACKET_COUNT} "
            f"RTT: {rtt_ms:.2f} ms"
        )


    sock.close()


    # ========================================================
    # RESULTS
    # ========================================================

    average_rtt = statistics.mean(
        results
    )

    minimum_rtt = min(results)

    maximum_rtt = max(results)


    print()
    print("==============================")
    print("NETWORK PROBE RESULTS")
    print("==============================")

    print(
        f"Average RTT : {average_rtt:.2f} ms"
    )

    print(
        f"Minimum RTT : {minimum_rtt:.2f} ms"
    )

    print(
        f"Maximum RTT : {maximum_rtt:.2f} ms"
    )


    # ========================================================
    # TEMPORARY CLASSIFICATION
    # ========================================================

    if average_rtt <= 100:

        network_quality = "GOOD"

    else:

        network_quality = "POOR"


    print(
        f"Network Quality: {network_quality}"
    )


if __name__ == "__main__":

    run_probe()