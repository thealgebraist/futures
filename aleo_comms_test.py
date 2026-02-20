import socket
import ssl
import json
import time
import sys

# --- CONFIGURATION ---
ADDR = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_audit"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 40000
TIMEOUT = 3.0

POOLS = [
    {"name": "F2Pool SSL",  "host": "aleo-asia.f2pool.com", "port": 4420,  "proto": "ssl"},
    {"name": "Apool HK IP", "host": "172.65.162.169",       "port": 9090,  "proto": "tcp"},
    {"name": "ZkWork HK",   "host": "aleo.hk.zk.work",      "port": 10003, "proto": "tcp"},
    {"name": "Oula WSS",    "host": "aleo.oula.network",    "port": 6666,  "proto": "wss"},
    {"name": "WhalePool",   "host": "aleo.asia1.whalepool.com", "port": 42343, "proto": "tcp"}
]

def manual_socks5_connect(dest_host, dest_port):
    """Pure Standard Library SOCKS5h Handshake."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT)
    try:
        s.connect((PROXY_HOST, PROXY_PORT))
    except:
        # Fallback to direct if WARP proxy isn't listening
        print(f"  [WARN] WARP Proxy not found on {PROXY_PORT}. Trying direct...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.connect((dest_host, dest_port))
        return s

    # SOCKS5 Greeting
    s.sendall(b"\x05\x01\x00")
    if s.recv(2) != b"\x05\x00": raise Exception("SOCKS5 Auth Failed")
    
    # SOCKS5 Connect (Mode 0x03 = Domain Name for remote DNS)
    host_bytes = dest_host.encode()
    request = b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + dest_port.to_bytes(2, 'big')
    s.sendall(request)
    
    resp = s.recv(10)
    if not resp or resp[1] != 0: raise Exception("SOCKS5 Target Unreachable")
    return s

def test_pool(pool):
    print(f"\n[TEST] {pool['name']} ({pool['host']}:{pool['port']})")
    conn = None
    try:
        raw_sock = manual_socks5_connect(pool['host'], pool['port'])
        
        if pool['proto'] == "tcp":
            conn = raw_sock
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = ctx.wrap_socket(raw_sock, server_hostname=pool['host'])
            
            if pool['proto'] == "wss":
                handshake = (f"GET / HTTP/1.1\r\nHost: {pool['host']}\r\n"
                             "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                             "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                             "Sec-WebSocket-Version: 13\r\n\r\n")
                conn.sendall(handshake.encode())
                if b"101" not in conn.recv(1024): raise Exception("WSS Upgrade Failed")

        # Stratum Handshake
        auth = {"id": 1, "method": "mining.authorize", "params": [ADDR, "x"]}
        conn.sendall((json.dumps(auth) + "\n").encode())
        
        resp = conn.recv(4096).decode()
        print(f"  [RECV] {resp[:80].strip()}")
        return ("result" in resp or "null" in resp)

    except Exception as e:
        print(f"  [FAIL] {str(e)}")
        return False
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    print("=========================================")
    print("   ALEO PURE-PYTHON AUDITOR (NO LIBS)    ")
    print("=========================================")
    results = []
    for p in POOLS: results.append((p['name'], test_pool(p)))
    
    print("\n" + "="*40)
    print("   SUMMARY")
    print("="*40)
    for name, ok in results:
        print(f"{name:<15} : {'ONLINE' if ok else 'BLOCKED'}")
