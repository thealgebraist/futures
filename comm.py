import socket
import json
import ssl
import sys
import argparse
import time

def run_task(pool_url, pool_port, address, mode, nonce=None):
    TIMEOUT = 2.0
    
    use_ssl = (pool_port in [443, 4420, 4430, 6666]) or "ssl" in pool_url
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
        
        # LOG RAW CONNECTION SUCCESS
        print(f"DEBUG_RAW_CONNECT: {pool_url}:{pool_port}")

        if is_wss:
            handshake = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {pool_url}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            conn.sendall(handshake.encode())
            resp = conn.recv(1024)
            print(f"DEBUG_RAW_WSS_RESP: {resp[:200]!r}")
            if b"101" not in resp: return False

        # 1. Authorize
        auth = {"id": 1, "method": "mining.authorize", "params": [address, "x"]}
        conn.sendall((json.dumps(auth) + "\n").encode())
        
        resp = conn.recv(4096)
        print(f"DEBUG_RAW_AUTH_RESP: {resp[:500]!r}")
        
        if mode == "test":
            if b"result\":true" in resp or b"result\": null" in resp or b"null" in resp:
                print("OK")
                return True
            else:
                return False
        
        if mode == "submit" and nonce:
            submit = {"id": 4, "method": "mining.submit", "params": [address, "job_v34", str(nonce), "0x0"]}
            conn.sendall((json.dumps(submit) + "\n").encode())
            resp = conn.recv(4096)
            print(f"SUBMIT_RAW: {resp!r}")
            return True

    except Exception as e:
        print(f"ERR: {type(e).__name__} - {str(e)}")
        return False
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--nonce", type=str)
    args = parser.parse_args()
    run_task(args.pool, args.port, args.address, args.mode, args.nonce)
