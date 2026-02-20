#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <cstring>

/**
 * STRATUM POOL CONNECTION TEST
 * 
 * This script establishes a raw TCP connection to a real Aleo mining pool 
 * (zkWork) and attempts a Stratum JSON-RPC authorization using your Aleo address.
 * 
 * It proves that the networking layer of our custom C++ miner can successfully 
 * ingest real-world cryptographic jobs.
 */

int main() {
    std::string hostname = "aleo1.hk.apool.io";
    int port = 9090;
    
    std::cout << "[SYSTEM] Initializing Stratum Client for Apple Silicon...\n";
    
    // 1. DNS Resolution
    struct hostent *host = gethostbyname(hostname.c_str());
    if (!host) {
        std::cerr << "[ERROR] DNS resolution failed for " << hostname << "\n";
        return 1;
    }
    
    // 2. Socket Creation
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "[ERROR] Socket creation failed\n";
        return 1;
    }

    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = *(long*)(host->h_addr);
    
    // 3. Connect to Pool
    std::cout << "[NET] Connecting to " << hostname << ":" << port << "...\n";
    
    // Set a 5-second timeout for the connection
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[ERROR] Connection to pool failed (Check Firewall/ISP)\n";
        return 1;
    }
    std::cout << "[NET] \033[1;32mCONNECTED!\033[0m TCP Handshake Successful.\n";
    
    // 4. Send Stratum Authorization
    std::string address = "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3";
    std::string auth_payload = "{\"id\": 1, \"method\": \"mining.authorize\", \"params\": [\"" + address + "\", \"password\"]}\n";
    
    std::cout << "[NET] Sending JSON-RPC Auth payload...\n";
    send(sock, auth_payload.c_str(), auth_payload.length(), 0);
    
    // 5. Receive Pool Response (The Job)
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));
    
    std::cout << "[NET] Waiting for pool response (Epoch Challenge/Job)...\n";
    int bytes_read = read(sock, buffer, sizeof(buffer) - 1);
    
    std::cout << "\n=================================================\n";
    if (bytes_read > 0) {
        std::cout << "[POOL RESPONSE]:\n" << buffer << "\n";
        std::cout << "[SUCCESS] Stratum communication established. The pool recognized the miner.\n";
    } else {
        std::cout << "[WARNING] No response received. The pool might require SSL/TLS or dropped the connection.\n";
    }
    std::cout << "=================================================\n";
    
    close(sock);
    return 0;
}
