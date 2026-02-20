import socket
import socks
import ssl
import json
import time
from websocket import create_connection

# --- CONFIGURATION ---
ADDR = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_audit"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 40000
TIMEOUT = 3.0

POOLS = [
    {"name": "ZkWork HK",   "host": "aleo.hk.zk.work",      "port": 10003, "proto": "tcp"},
    {"name": "Apool HK IP", "host": "172.65.162.169",       "port": 9090,  "proto": "tcp"},
    {"name": "F2Pool SSL",  "host": "aleo-asia.f2pool.com", "port": 4420,  "proto": "ssl"},
    {"name": "Oula WSS",    "host": "aleo.oula.network",    "port": 6666,  "proto": "wss"},
    {"name": "Uniplus",     "host": "pool.uniplus.pro",     "port": 9090,  "proto": "tcp"},
    {"name": "WhalePool",   "host": "aleo.asia1.whalepool.com", "port": 42343, "proto": "tcp"}
]

def test_pool(pool):
    print(f"\n[TESTING] {pool['name']} ({pool['host']}:{pool['port']}) via {pool['proto'].upper()}")
    
    try:
        if pool['proto'] == "wss":
            # Native WebSocket through SOCKS5
            ws = create_connection(
                f"wss://{pool['host']}:{pool['port']}",
                proxy_type="socks5",
                http_proxy_host=PROXY_HOST,
                http_proxy_port=PROXY_PORT,
                timeout=TIMEOUT,
                sslopt={"cert_reqs": ssl.CERT_NONE}
            )
            auth_msg = {"id": 1, "method": "mining.authorize", "params": [ADDR, "x"]}
            ws.send(json.dumps(auth_msg))
            resp = ws.recv()
            print(f"  [RECV] {resp[:100]}")
            ws.close()
            return True

        # Stratum TCP/SSL through SOCKS5
        s = socks.socksocket()
        s.set_proxy(socks.SOCKS5, PROXY_HOST, PROXY_PORT, rdns=True)
        s.settimeout(TIMEOUT)
        
        if pool['proto'] == "ssl":
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            raw_s = s
            s = ctx.wrap_socket(raw_s, server_hostname=pool['host'])
            s.connect((pool['host'], pool['port']))
        else:
            s.connect((pool['host'], pool['port']))

        auth_msg = {"id": 1, "method": "mining.authorize", "params": [ADDR, "x"]}
        s.sendall((json.dumps(auth_msg) + "\n").encode())
        
        data = s.recv(4096).decode()
        print(f"  [RECV] {data[:100].strip()}")
        s.close()
        
        if "result\":true" in data or "result\": null" in data or "null" in data or "jsonrpc" in data:
            return True
        return False

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    print("=================================================")
    print("   ALEO NATIVE PROTOCOL AUDITOR (UV POWERED)     ")
    print("   Routing: Cloudflare WARP Tunnel               ")
    print("=================================================")
    
    results = []
    for p in POOLS:
        results.append((p['name'], test_pool(p)))

    print("\n\n" + "="*40)
    print("   FINAL AUDIT SUMMARY")
    print("="*40)
    for name, success in results:
        status = "\033[1;32mONLINE\033[0m" if success else "\033[1;31mOFFLINE/BLOCKED\033[0m"
        print(f"{name:<15} : {status}")
    print("="*40)
