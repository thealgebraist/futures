import socket
import json
import time

EXTENDED_POOLS = [
    {"name": "Apool SG", "host": "aleo1.sg.apool.io", "port": 9090},
    {"name": "ZKRush US", "host": "aleo.us.zk.work", "port": 10003},
    {"name": "Hpool Global", "host": "aleo.hpool.io", "port": 9090},
    {"name": "AleoPool.io", "host": "aleo.aleopool.io", "port": 9090}
]

ADDR = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_audit"

def test_pool(pool):
    print(f"\n--- AUDITING: {pool['name']} ({pool['host']}) ---")
    try:
        try:
            ip = socket.gethostbyname(pool['host'])
            print(f"[DNS] Resolved: {ip}")
        except:
            print("[DNS] FAILED to resolve.")
            return False

        s = socket.create_connection((ip, pool['port']), timeout=5)
        print(f"[TCP] CONNECTED to {ip}:{pool['port']}")

        auth = {"id": 1, "method": "mining.authorize", "params": [ADDR, "x"]}
        s.sendall((json.dumps(auth) + "\n").encode())
        
        s.settimeout(5)
        resp = s.recv(4096).decode()
        if "result\":true" in resp or "result\": null" in resp or "null" in resp:
            print("[AUTH] SUCCESS")
            return True
        else:
            print(f"[AUTH] REJECTED: {resp.strip()}")
            return False

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

if __name__ == "__main__":
    print("=========================================")
    print("   ALEO EXTENDED POOL AUDIT (4 NEW)      ")
    print("=========================================")
    results = []
    for p in EXTENDED_POOLS:
        results.append((p['name'], test_pool(p)))
    
    print("\n\nFINAL SUMMARY:")
    for name, success in results:
        status = "ONLINE" if success else "BLOCKED/ERROR"
        print(f"{name:<15}: {status}")
