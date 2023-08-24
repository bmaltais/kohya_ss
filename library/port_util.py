import socket

USED_PORTS = set()


def get_available_port(port: int, max_retries=100) -> int:
    tried_ports = []
    while port < 65535:
        if port in USED_PORTS:
            port += 1
            continue
        try:
            sock = socket.socket()
            sock.bind(("127.0.0.1", port)) # only bind to localhost
            sock.close()
            return port
        except (OSError, socket.error):
            print(f"Port {port} is already in use.")
            tried_ports.append(port)
            port += 1
        if len(tried_ports) > max_retries:
            break
    raise OSError(f"No available ports found, searched from {port} to {port + max_retries}")

def mark_port_as_used(port: int):
    USED_PORTS.add(port)
  
def remove_port_from_used(port: int):
    if port in USED_PORTS:
        USED_PORTS.remove(port)