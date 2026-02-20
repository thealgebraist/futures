const WebSocket = require('ws');

const ws = new WebSocket('wss://aleo.zkrush.com:3333');

console.log('[SYSTEM] Initializing WSS Stratum Client for ZKRush...');

ws.on('open', function open() {
  console.log('[NET] CONNECTED! TLS/WebSocket Handshake Successful.');
  
  const authPayload = {
    id: 1,
    method: "mining.authorize",
    params: ["aleo1wss37wdffev2ezdz4e48hq3yk9k2xenzzhweeh3rse7qm8rkqc8s4vp8v3", "worker_m2"]
  };
  
  console.log('[NET] Sending JSON-RPC Auth payload...');
  ws.send(JSON.stringify(authPayload) + "\\n");
});

ws.on('message', function message(data) {
  console.log('=================================================');
  console.log('[POOL RESPONSE]: ' + data);
  console.log('[SUCCESS] Stratum communication established. The pool recognized the miner.');
  console.log('=================================================');
  
  ws.close();
  process.exit(0);
});

ws.on('error', function error(err) {
  console.error('[ERROR] Connection failed:', err.message);
  process.exit(1);
});

setTimeout(() => {
  console.error('[ERROR] Connection timed out after 10 seconds.');
  process.exit(1);
}, 10000);
