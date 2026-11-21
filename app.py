import asyncio, base64, json, os, subprocess, time, uuid
import httpx
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

PORT = int(os.environ.get('PORT', '8081'))
MODEL_NAME = 'bonsai-2-27b'
MLKEM_PATH = '/usr/local/bin/mlkem-generator'
MASTER = base64.b64decode(os.environ['HS_MASTER_KEY'])   # 32 bytes, identical on every replica
TOKEN_TTL = 15 * 60
ACTIVE = 0    # in-flight chats; keeps /health stable during a long prefill

api = FastAPI()
client = AsyncOpenAI(base_url=f'http://127.0.0.1:{PORT}/v1', api_key='None')

async def _llama_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            return (await c.get(f'http://127.0.0.1:{PORT}/health')).status_code == 200
    except Exception:
        return False

@api.get('/live')       # liveness: only proves the Python process is alive
async def live(): return {'status': 'ok'}

@api.get('/health')     # startup + readiness; frontend checks status == 'ok'
async def health():
    if ACTIVE == 0 and not await _llama_ready(): raise HTTPException(503, 'model loading')
    return {'status': 'ok', 'models': [MODEL_NAME]}

async def _guarded(action):   # no os.kill any more; entrypoint.sh handles crashes
    try: return await action()
    except HTTPException: raise
    except Exception as e:
        print(f'[error] {e!r}', flush=True)
        raise HTTPException(502, 'backend error')

# _run_mlkem: unchanged

def _seal(dk: str, client_ek: str) -> str:
    nonce = os.urandom(12)
    body = json.dumps({'dk': dk, 'cek': client_ek, 'exp': int(time.time()) + TOKEN_TTL}).encode()
    return base64.b64encode(nonce + ChaCha20Poly1305(MASTER).encrypt(nonce, body, None)).decode()

def _open(token: str):
    try:
        raw = base64.b64decode(token)
        body = json.loads(ChaCha20Poly1305(MASTER).decrypt(raw[:12], raw[12:], None))
    except Exception:
        raise HTTPException(400, 'bad handshake token')
    if body['exp'] < time.time(): raise HTTPException(400, 'handshake token expired')
    return body['dk'], body['cek']

@api.post('/v1/handshake')
async def handshake(request: dict):
    client_ek = request.get('ek')
    if not client_ek: raise HTTPException(400, "Missing 'ek'")
    keygen = await _run_mlkem('keygen')
    return {'ek': keygen['ek'], 'token': _seal(keygen['dk'], client_ek)}

async def with_keepalive(agen, interval: float = 15):
    global ACTIVE
    ACTIVE += 1
    q: asyncio.Queue = asyncio.Queue()
    END = object()
    async def pump():
        try:
            async for item in agen: await q.put(item)
        except Exception as e: await q.put(e)
        finally: await q.put(END)
    task = asyncio.create_task(pump())
    try:
        yield 'data: {"type":"ping"}\n\n'      # first byte immediately, before prefill
        while True:
            try: item = await asyncio.wait_for(q.get(), interval)
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'; continue
            if item is END: break
            if isinstance(item, Exception): raise item
            yield item
    finally:
        ACTIVE -= 1
        task.cancel()   # client gone: upstream stream closes, llama-server aborts and frees the slot

@api.post('/chat')
async def chat(request: dict):
    async def _():
        if not await _llama_ready(): raise HTTPException(503, 'model loading')
        token = request.get('token')
        if not token: raise HTTPException(400, 'Missing token')
        server_dk, client_ek = _open(token)
        ct, em = request.get('ct'), request.get('em')
        if not ct or not em: raise HTTPException(400, 'Missing ct/em.')
        # ... unchanged: decapsulate-and-decrypt, messages, reasoning_effort, artifacts, params, extra_body,
        # ... unchanged: encapsulate-and-encrypt (resp_ct, resp_ss, resp_em).
        # Delete: session.pop('client_ek'), _save_session, all x_session_id use.
        # ... unchanged: async def generate(): ...
        return StreamingResponse(with_keepalive(generate()), media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    return await _guarded(_)