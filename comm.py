import socket
import json
import ssl
import sys
import argparse
import time

def run_task(pool_url, pool_port, address, mode, nonce=None):
    # Strict 2s timeout as requested
    TIMEOUT = 2.0
    
    # Identify if we should use SSL
    use_ssl = (pool_port in [4420, 443, 4430, 6666]) or "ssl" in pool_url
    
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
            
        start_time = time.time()
        conn.connect((pool_url, int(pool_port)))
        
        # 1. Handshake (Standard Stratum)
        auth = {"id": 1, "method": "mining.authorize", "params": [address, "x"]}
        conn.sendall((json.dumps(auth) + "\n").encode())
        
        # 2. Wait for response
        resp = conn.recv(4096).decode()
        
        if mode == "test":
            if "result\":true" in resp or "result\": null" in resp:
                print("OK")
                return True
            else:
                print(f"REJECTED: {resp.strip()}")
                return False
        
        if mode == "submit" and nonce:
            submit = {
                "id": 4, 
                "method": "mining.submit", 
                "params": [address, "job_v32", str(nonce), "0x0"]
            }
            conn.sendall((json.dumps(submit) + "\n").encode())
            resp = conn.recv(4096).decode()
            print(f"SUBMIT_OK: {resp.strip()}")
            return True

    except Exception as e:
        if mode == "test":
            print(f"FAIL: {type(e).__name__}")
        else:
            print(f"ERROR: {str(e)}")
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
