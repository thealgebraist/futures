import socket
import ssl
import json
import asyncio
import websockets

# Direct IPs to bypass local DNS blocks
POOL_CONFIG = [
    {"name": "zkWork (HK)", "host": "aleo.hk.zk.work", "ip": "47.243.163.37", "port": 10003, "type": "tcp"},
    {"name": "f2pool (SSL)", "host": "aleo.f2pool.com", "ip": "172.65.186.4", "port": 4420, "type": "ssl"},
    {"name": "Antpool", "host": "aleo.antpool.com", "ip": "172.65.186.4", "port": 9038, "type": "tcp"},
    {"name": "Apool (HK)", "host": "aleo1.hk.apool.io", "ip": "172.65.162.169", "port": 9090, "type": "tcp"},
    {"name": "ZKRush (WSS)", "url": "wss://aleo.zkrush.com:3333", "type": "wss"}
]

def test_tcp_ssl(config):
    target = config["ip"]
    port = config["port"]
    try:
        sock = socket.create_connection((target, port), timeout=5)
        if config["type"] == "ssl":
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=config["host"])
        
        payload = json.dumps({"id": 1, "method": "mining.subscribe", "params": []}) + "\n"
        sock.sendall(payload.encode())
        sock.close()
        return True
    except:
        return False

async def test_wss(config):
    try:
        # Use a shorter timeout for the handshake
        async with websockets.connect(config["url"], open_timeout=5) as ws:
            return True
    except:
        return False

async def main():
    print("--- Full Aleo Pool Connectivity Report (uv/websockets) ---")
    for pool in POOL_CONFIG:
        if pool["type"] == "wss":
            success = await test_wss(pool)
        else:
            success = test_tcp_ssl(pool)
        
        status = "REACHABLE" if success else "BLOCKED"
        color = "\033[1;32m" if success else "\033[1;31m"
        print(f"{pool['name']:<15} : {color}{status}\033[0m")

if __name__ == "__main__":
    asyncio.run(main())
