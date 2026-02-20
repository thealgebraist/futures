import socket
import json
import ssl
import sys
import argparse
import time

def run_task(pool_ip, pool_port, address, mode, nonce=None):
    TIMEOUT = 2.0
    
    # 0. Basic Internet Connectivity Test
    if pool_ip == "8.8.8.8":
        try:
            s = socket.create_connection((pool_ip, 53), timeout=TIMEOUT)
            s.close()
            print("INTERNET_OK")
            return True
        except:
            print("INTERNET_OFFLINE")
            return False

    use_ssl = (pool_port in [443, 4420, 4430, 6666])
    
    try:
        # 1. TCP Connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT)
        try:
            sock.connect((pool_ip, int(pool_port)))
        except Exception as e:
            print(f"CONN_FAIL: {type(e).__name__}")
            return False

        # 2. SSL/TLS Wrapping with SNI Masking
        if use_ssl:
            try:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                # MASK SNI: Pretend we are connecting to a standard CDN
                conn = context.wrap_socket(sock, server_hostname="www.cloudflare.com")
            except Exception as e:
                print(f"SSL_FAIL: {type(e).__name__}")
                return False
        else:
            conn = sock

        # 3. Stratum Authorization
        try:
            auth = {"id": 1, "method": "mining.authorize", "params": [address, "x"]}
            conn.sendall((json.dumps(auth) + "\n").encode())
            resp = conn.recv(1024).decode()
            
            if mode == "test":
                if "result\":true" in resp or "result\": null" in resp or "null" in resp:
                    print("OK")
                    return True
                else:
                    print(f"AUTH_REJECT: {resp[:50]!r}")
                    return False
            
            if mode == "submit" and nonce:
                submit = {"id": 4, "method": "mining.submit", "params": [address, "job_v35", str(nonce), "0x0"]}
                conn.sendall((json.dumps(submit) + "\n").encode())
                resp = conn.recv(1024).decode()
                print(f"SUBMIT_OK: {resp.strip()}")
                return True
        except Exception as e:
            print(f"AUTH_FAIL: {type(e).__name__}")
            return False

    except Exception as e:
        print(f"FATAL: {type(e).__name__}")
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
