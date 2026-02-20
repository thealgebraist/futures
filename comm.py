import socket
import json
import ssl
import sys
import argparse
import time

def run_task(pool_url, pool_port, address, mode, nonce=None):
    TIMEOUT = 2.0
    
    # 0. Sanity Check: Test basic internet if requested
    if pool_url == "internet_test":
        try:
            s = socket.create_connection(("8.8.8.8", 53), timeout=TIMEOUT)
            s.close()
            print("INTERNET_OK")
            return True
        except:
            print("INTERNET_BLOCKED")
            return False

    use_ssl = (pool_port in [443, 4420, 4430, 6666, 443]) or "ssl" in pool_url
    is_wss = (pool_port == 6666) or "wss" in pool_url

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        
        if use_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            conn = context.wrap_socket(sock, server_hostname=pool_url)
        else:
            conn = sock
            
        conn.connect((pool_url, int(pool_port)))
        
        if is_wss:
            # Masked WSS Handshake
            handshake = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {pool_url}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\r\n"
                f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            conn.sendall(handshake.encode())
            resp = conn.recv(1024).decode()
            if "101" not in resp:
                print(f"WSS_REJECTED: {resp[:20]}")
                return False

        # 1. Authorize
        auth = {"id": 1, "method": "mining.authorize", "params": [address, "x"]}
        conn.sendall((json.dumps(auth) + "\n").encode())
        
        resp = conn.recv(4096).decode()
        
        if mode == "test":
            if "result\":true" in resp or "result\": null" in resp or "null" in resp:
                print("OK")
                return True
            else:
                print(f"AUTH_FAIL: {resp[:30]}")
                return False
        
        if mode == "submit" and nonce:
            submit = {"id": 4, "method": "mining.submit", "params": [address, "job_v33", str(nonce), "0x0"]}
            conn.sendall((json.dumps(submit) + "\n").encode())
            resp = conn.recv(4096).decode()
            print(f"SUBMIT_OK: {resp.strip()}")
            return True

    except Exception as e:
        print(f"ERR: {type(e).__name__}")
        return False
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "submit"], required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--nonce", type=str)
    args = parser.parse_args()
    run_task(args.pool, args.port, args.address, args.mode, args.nonce)
