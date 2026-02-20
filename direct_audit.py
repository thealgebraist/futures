import socket
import ssl
import json
import time

# --- CONFIGURATION ---
ADDR = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_audit"
TIMEOUT = 2.0

POOLS = [
    {"name": "F2Pool US (SSL)",  "host": "aleo-us.f2pool.com",   "port": 4420,  "proto": "ssl"},
    {"name": "F2Pool AS (SSL)",  "host": "aleo-asia.f2pool.com", "port": 4420,  "proto": "ssl"},
    {"name": "Apool US",         "host": "aleo1.us.apool.io",    "port": 9090,  "proto": "tcp"},
    {"name": "Apool HK",         "host": "aleo1.hk.apool.io",    "port": 9090,  "proto": "tcp"},
    {"name": "ZkWork US",        "host": "aleo.us.zk.work",      "port": 10003, "proto": "tcp"},
    {"name": "ZkWork HK",        "host": "aleo.hk.zk.work",      "port": 10003, "proto": "tcp"},
    {"name": "Oula WSS",         "host": "aleo.oula.network",    "port": 6666,  "proto": "wss"},
    {"name": "WhalePool US",     "host": "aleo.us1.whalepool.com","port": 42343, "proto": "tcp"},
    {"name": "AntPool",          "host": "aleo.antpool.com",     "port": 9038,  "proto": "tcp"},
    {"name": "Hpool",            "host": "aleo.hpool.io",        "port": 9090,  "proto": "tcp"},
    {"name": "F2Pool Port 443",  "host": "aleo.f2pool.com",      "port": 443,   "proto": "ssl"},
    {"name": "AleoPool",         "host": "aleo.aleopool.io",     "port": 9090,  "proto": "tcp"},
    {"name": "Uniplus",          "host": "pool.uniplus.pro",     "port": 9090,  "proto": "tcp"},
    {"name": "6Pool",            "host": "aleo.6pool.com",       "port": 9090,  "proto": "tcp"}
]

def test_pool(pool):
    print(f"[TEST] {pool['name']:<15} ({pool['host']}:{pool['port']})", end=" ... ", flush=True)
    conn = None
    try:
        # Direct Connection
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        s.connect((pool['host'], pool['port']))
        
        if pool['proto'] in ["ssl", "wss"]:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = ctx.wrap_socket(s, server_hostname=pool['host'])
            
            if pool['proto'] == "wss":
                handshake = (f"GET / HTTP/1.1
Host: {pool['host']}
"
                             "Upgrade: websocket
Connection: Upgrade
"
                             "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
"
                             "Sec-WebSocket-Version: 13

")
                conn.sendall(handshake.encode())
                if b"101" not in conn.recv(1024): raise Exception("WSS Upgrade Failed")
        else:
            conn = s

        # Stratum Handshake
        auth = {"id": 1, "method": "mining.authorize", "params": [ADDR, "x"]}
        conn.sendall((json.dumps(auth) + "
").encode())
        
        resp = conn.recv(4096).decode()
        if "result" in resp or "null" in resp or "jsonrpc" in resp:
            print("\033[1;32mONLINE\033[0m")
            return True
        else:
            print(f"\033[1;31mREJECTED\033[0m ({resp[:20].strip()})")
            return False

    except Exception as e:
        print(f"\033[1;31mFAIL\033[0m ({type(e).__name__})")
        return False
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    print("=========================================")
    print("   ALEO DIRECT-PATH AUDITOR (NO PROXY)   ")
    print("=========================================")
    results = []
    for p in POOLS: results.append((p['name'], test_pool(p)))
    
    print("
" + "="*40)
    print("   FINAL SUMMARY (DIRECT CONNECT)")
    print("="*40)
    for name, ok in results:
        if ok: print(f"{name:<15} : \033[1;32mREADY TO MINE\033[0m")
    print("="*40)
