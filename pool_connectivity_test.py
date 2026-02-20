import socket
import ssl
import json

pools = [
    {"name": "zkWork (HK)", "host": "aleo.hk.zk.work", "ip": "47.243.163.37", "port": 10003, "type": "tcp"},
    {"name": "ZKRush", "host": "aleo.zkrush.com", "ip": "47.243.163.37", "port": 3333, "type": "tcp"}, # Note: ZKRush IP might be different, but let's test port
    {"name": "f2pool (Global)", "host": "aleo.f2pool.com", "ip": "172.65.186.4", "port": 4420, "type": "ssl"},
    {"name": "Antpool", "host": "aleo.antpool.com", "ip": "172.65.186.4", "port": 9038, "type": "tcp"},
    {"name": "Apool (HK)", "host": "aleo1.hk.apool.io", "ip": "172.65.162.169", "port": 9090, "type": "tcp"},
]

def test_connection(name, host, ip, port, conn_type):
    print(f"Testing {name} ({host}:{port})...")
    
    # Try Hostname first
    try:
        s = socket.create_connection((host, port), timeout=5)
        if conn_type == "ssl":
            context = ssl.create_default_context()
            s = context.wrap_socket(s, server_hostname=host)
        print(f"  [SUCCESS] Connected to {host}")
        s.close()
        return True
    except Exception as e:
        print(f"  [FAILED] Hostname {host}: {e}")

    # Try IP if hostname failed
    if ip:
        print(f"  Trying direct IP {ip}...")
        try:
            s = socket.create_connection((ip, port), timeout=5)
            if conn_type == "ssl":
                context = ssl.create_default_context()
                # For SSL, we still need the hostname for SNI, but we connect to the IP
                s = context.wrap_socket(s, server_hostname=host)
            print(f"  [SUCCESS] Connected to {ip}")
            s.close()
            return True
        except Exception as e:
            print(f"  [FAILED] IP {ip}: {e}")
            
    return False

if __name__ == "__main__":
    results = []
    for pool in pools:
        success = test_connection(pool["name"], pool["host"], pool["ip"], pool["port"], pool["type"])
        results.append({"name": pool["name"], "success": success})
    
    print("
--- Summary ---")
    for res in results:
        status = "PASSED" if res["success"] else "FAILED"
        print(f"{res['name']}: {status}")
