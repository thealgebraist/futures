#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

/**
 * PRODUCTION HYBRID MINER (v29 - WSS FRAMING & CRASH FIX)
 */

enum class Proto { TCP, SSL, WSS };

struct MinerState {
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> authorized{false};
    std::atomic<uint64_t> hashes{0};
    std::atomic<uint64_t> shares{0};
    int socket_fd{-1};
    SSL* ssl_handle{nullptr};
    SSL_CTX* ssl_ctx{nullptr};
    Proto current_proto{Proto::TCP};
    char address[256];
    char pool_url[128];
    int pool_port;
};

// ------------------------------------------------------------------
// WSS FRAMING (RFC 6455)
// ------------------------------------------------------------------
void send_framed(MinerState* state, const char* data) {
    size_t len = strlen(data);
    std::vector<uint8_t> frame;
    frame.push_back(0x81); // Text frame
    if (len <= 125) {
        frame.push_back((uint8_t)len | 0x80); // Mask bit set
    } else {
        frame.push_back(126 | 0x80);
        frame.push_back((len >> 8) & 0xFF);
        frame.push_back(len & 0xFF);
    }
    uint8_t mask[4] = {0x12, 0x34, 0x56, 0x78}; // Simple mask
    frame.insert(frame.end(), mask, mask + 4);
    for (size_t i = 0; i < len; ++i) frame.push_back(data[i] ^ mask[i % 4]);

    if (state->ssl_handle) SSL_write(state->ssl_handle, frame.data(), frame.size());
    else send(state->socket_fd, frame.data(), frame.size(), 0);
}

// ------------------------------------------------------------------
// NETWORK ENGINE
// ------------------------------------------------------------------
bool wss_handshake(MinerState* state) {
    char req[1024];
    snprintf(req, 1024, "GET / HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n", state->pool_url);
    SSL_write(state->ssl_handle, req, strlen(req));
    char resp[2048];
    int r = SSL_read(state->ssl_handle, resp, 2047);
    return (r > 0 && strstr(resp, "101 Switching Protocols"));
}

void stratum_listener(MinerState* state) {
    char buf[16384];
    while (state->connected && !state->stop_flag) {
        int r = (state->ssl_handle) ? SSL_read(state->ssl_handle, buf, 16383) : read(state->socket_fd, buf, 16383);
        if (r <= 0) { state->connected = false; break; }
        buf[r] = '\0';
        if (strstr(buf, "mining.notify") || strstr(buf, "authorize") || strstr(buf, "submit")) {
            std::printf("\n\033[1;30m[POOL]\033[0m %s", strstr(buf, "{")); std::fflush(stdout);
            if (strstr(buf, "\"result\":true") || strstr(buf, "null")) {
                if (strstr(buf, "authorize")) state->authorized = true;
                else if (strstr(buf, "submit")) state->shares++;
            }
        }
    }
}

void attempt_connection(MinerState* state) {
    struct hostent* host = gethostbyname(state->pool_url);
    struct sockaddr_in serv{}; serv.sin_family = AF_INET; serv.sin_port = htons(state->pool_port);
    if (host) memcpy(&serv.sin_addr, host->h_addr, host->h_length);
    else inet_pton(AF_INET, "172.65.230.151", &serv.sin_addr);

    state->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(state->socket_fd, (struct sockaddr*)&serv, sizeof(serv)) < 0) return;

    if (state->current_proto == Proto::SSL || state->current_proto == Proto::WSS) {
        state->ssl_handle = SSL_new(state->ssl_ctx);
        SSL_set_fd(state->ssl_handle, state->socket_fd);
        if (SSL_connect(state->ssl_handle) <= 0) return;
        if (state->current_proto == Proto::WSS && !wss_handshake(state)) return;
    }

    state->connected = true;
    std::thread(stratum_listener, state).detach();

    char auth[512]; snprintf(auth, 512, "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"%s\",\"x\"]}\n", state->address);
    if (state->current_proto == Proto::WSS) send_framed(state, auth);
    else if (state->ssl_handle) SSL_write(state->ssl_handle, auth, strlen(auth));
    else send(state->socket_fd, auth, strlen(auth), 0);
}

int main(int argc, char** argv) {
    MinerState state; 
    SSL_library_init(); OpenSSL_add_all_algorithms(); SSL_load_error_strings();
    state.ssl_ctx = SSL_CTX_new(TLS_client_method());
    strcpy(state.address, "aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3.worker_v29");
    strcpy(state.pool_url, "aleo.oula.network"); state.pool_port = 6666; state.current_proto = Proto::WSS;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--address") == 0 && (i + 1 < argc)) strcpy(state.address, argv[++i]);
        else if (strcmp(argv[i], "--pool") == 0 && (i + 1 < argc)) {
            char* url = argv[++i];
            if (strstr(url, "ssl://")) { state.current_proto = Proto::SSL; url += 6; }
            else if (strstr(url, "wss://")) { state.current_proto = Proto::WSS; url += 6; }
            else state.current_proto = Proto::TCP;
            char* p = strchr(url, ':');
            if (p) { *p = '\0'; strcpy(state.pool_url, url); state.pool_port = atoi(p+1); }
            else strcpy(state.pool_url, url);
        }
    }

    std::printf("=================================================\n");
    std::printf("   PRODUCTION HYBRID MINER (v29 - WSS FIXED)     \n");
    std::printf("   Pool: %s | Proto: %s\n", state.pool_url, state.current_proto == Proto::WSS ? "WSS" : "SSL/TCP");
    std::printf("=================================================\n");

    while(!state.stop_flag) {
        if (!state.connected) attempt_connection(&state);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::printf("\r[MINER] Speed: %.2f Mh/s | Acc: %llu | Status: %s", (double)state.hashes.exchange(0)/1e6, state.shares.load(), state.connected ? "ONLINE":"OFFLINE");
        std::fflush(stdout);
    }
    return 0;
}
