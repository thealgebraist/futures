import socket
import json
import time
import sys

def audit_pool(ip, port, address):
    print(f"--- APOOL HK AUDIT: {ip}:{port} ---")
    print(f"Testing Address: {address}")
    
    try:
        # 1. TCP Handshake
        print(f"[1/3] Attempting TCP Connection...", end="", flush=True)
        s = socket.create_connection((ip, port), timeout=5)
        print(" OK")
        
        # 2. Authorization
        print(f"[2/3] Sending mining.authorize...", end="", flush=True)
        auth_msg = {
            "id": 1,
            "method": "mining.authorize",
            "params": [address, "x"]
        }
        s.sendall((json.dumps(auth_msg) + "\n").encode())
        
        response = s.recv(4096).decode()
        print(" OK")
        print(f"      RAW RESPONSE: {response.strip()}")
        
        if "result\":true" in response or "result\": null" in response:
            print("\033[1;32m[PASS] Pool is accepting our credentials.\033[0m")
        else:
            print("\033[1;31m[FAIL] Pool rejected authorization.\033[0m")
            return
            
        # 3. Work Subscription
        print(f"[3/3] Waiting for mining.notify (Work)...", end="", flush=True)
        s.settimeout(10)
        work = s.recv(4096).decode()
        if "mining.notify" in work:
            print(" OK")
            print(f"      WORK RECEIVED: {work[:100]}...")
            print("\033[1;32m[SUCCESS] Full Stratum Pipeline Verified.\033[0m")
        else:
            print(" TIMEOUT/INVALID")
            print(f"      RESPONSE: {work.strip()}")
            
        s.close()
        
    except Exception as e:
        print(f"\n\033[1;31m[CRITICAL ERROR] {str(e)}\033[0m")
        print("Check if port 9090 is blocked by your firewall or if the IP is correct.")

if __name__ == "__main__":
    addr = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_audit"
    pool_ip = "172.65.162.169"
    pool_port = 9090
    
    if len(sys.argv) > 1:
        addr = sys.argv[1]
    
    audit_pool(pool_ip, pool_port, addr)
