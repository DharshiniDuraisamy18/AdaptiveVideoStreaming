import socket


HOST = "0.0.0.0"
PORT = 6000


server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1
)

server.bind(
    (HOST, PORT)
)

server.listen(1)

print(
    f"Network probe server listening on port {PORT}"
)


conn, address = server.accept()

print(
    "Probe connected from:",
    address
)


while True:

    data = conn.recv(4096)

    if not data:

        break

    conn.sendall(data)


conn.close()
server.close()

print("Probe server stopped.")