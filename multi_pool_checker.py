import socket
import json
import time

POOLS = [
    {"name": "Apool HK", "host": "aleo1.hk.apool.io", "ip": "172.65.162.169", "port": 9090},
    {"name": "Apool US", "host": "aleo1.us.apool.io", "ip": "172.65.230.151", "port": 9090},
    {"name": "ZKRush HK", "host": "aleo.hk.zk.work", "ip": "47.243.163.37", "port": 10003},
    {"name": "ZKRush SG", "host": "aleo.sg.zk.work", "ip": "161.117.82.155", "port": 10003},
    {"name": "F2Pool Asia", "host": "aleo-asia.f2pool.com", "ip": "47.52.166.182", "port": 4400}
]

ADDR = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_audit"

def test_pool(pool):
    print(f"\n--- TESTING: {pool['name']} ({pool['host']}) ---")
    
    # 1. DNS Check
    try:
        resolved = socket.gethostbyname(pool['host'])
        print(f"[DNS] Resolved to: {resolved}")
    except:
        print(f"[DNS] FAILED. Using static IP: {pool['ip']}")
        resolved = pool['ip']

    # 2. Connection
    try:
        s = socket.create_connection((resolved, pool['port']), timeout=5)
        print(f"[TCP] CONNECTED to {resolved}:{pool['port']}")
        
        # 3. Auth
        auth_msg = {"id": 1, "method": "mining.authorize", "params": [ADDR, "x"]}
        s.sendall((json.dumps(auth_msg) + "\n").encode())
        
        response = s.recv(4096).decode()
        if "result\":true" in response or "result\": null" in response:
            print(f"[AUTH] SUCCESS")
        else:
            print(f"[AUTH] REJECTED: {response.strip()}")
            s.close()
            return False

        # 4. Work Check
        print("[WORK] Waiting for mining.notify...", end="", flush=True)
        s.settimeout(10)
        start = time.time()
        while time.time() - start < 10:
            try:
                data = s.recv(4096).decode()
                if "mining.notify" in data:
                    print(" RECEIVED")
                    s.close()
                    return True
            except:
                break
        print(" TIMEOUT")
        s.close()
        return False

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

if __name__ == "__main__":
    results = []
    for p in POOLS:
        success = test_pool(p)
        results.append((p['name'], success))
    
    print("\n\n=========================================")
    print("   FINAL ALEO POOL AUDIT RESULTS")
    print("=========================================")
    for name, success in results:
        status = "ONLINE" if success else "OFFLINE/BLOCKED"
        print(f"{name:<15} : {status}")
    print("=========================================")
