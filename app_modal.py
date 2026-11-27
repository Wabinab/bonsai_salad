"""
TODO: The STATS generation currently only return the final line, but that's not true, because the model may call tools, 
the frontend and backend may break connection and needs resuming, etc that causes the reported statistics to break into 
multiple parts. . The solution is therefore to capture all stats 'after the end of a turn' and sum/average them out 
rather than only 'pick the last tag'. 
"""

import asyncio, difflib, json, os, subprocess, re, time, uuid
import httpx
# from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from typing import Optional

ENABLE_ARTIFACTS = True  # False = pop and ignore the client's `artifacts`, and don't expose artifact tools
SEND_PER_BATCH = 100  # deltas buffered before one encrypted packet is sent

WEB_TOOLS = [
  {"type": "function", "function": {
    "name": "web_search",
    "description": (
      "Search the web for information. Use this when you need current or factual information, "
      "for research comparison, documentation, and troubleshooting questions. "
      "Multiple `search_queries` can be issued in a single call."),
    "parameters": {
      "type": "object",
      "properties": {
        "objective": {"type": "string", "description":
          "One-sentence goal of this search. Keep it to one goal. May name preferred sources or how recent results should be."},
        "search_queries": {"type": "array", "items": {"type": "string"}, "description":
          "Keyword queries of 3-6 words each. At least one required; 2-3 work best. "
          "The `site:` operator takes a bare domain, not a URL path. Do not use quoted phrases."},
      },
      "required": ["objective", "search_queries"],
    },
  }},
  {"type": "function", "function": {
    "name": "web_fetch",
    "description": (
      "Fetch and extract relevant content from specific web URLs. Use only when "
      "web_search excerpts are insufficient, or when the user asked about a specific URL or page, "
      "or you need exact wording or quotes that excerpts may have truncated, or a full-page analysis, "
      "or when web_search excerpts are conflicting or clearly insufficient to answer."),
    "parameters": {
      "type": "object",
      "properties": {
        "objective": {"type": "string", "description":
          "What to look for in the pages. Maximum 200 characters."},
        "urls": {"type": "array", "items": {"type": "string"}, "description":
          "HTTP or HTTPS URLs to read. Maximum 20 per call."},
        "search_queries": {"type": "array", "items": {"type": "string"}, "description":
          "Optional. Keyword queries of 3-6 words each that focus the excerpts. "
          "When fetching results from web_search, pass the same queries."},
      },
      "required": ["objective", "urls"],
    },
  }},
  {"type": "function", "function": {
    "name": "image_search",
    "description": (
      "Find images on the web. Returns direct image URLs, the page each image appears on, "
      "the page title, and pixel dimensions when known. Results link to images but do not "
      "describe them; for information about a subject, use web_search instead. "
      "Use only when the user needs to see or obtain an image."),
    "parameters": {
      "type": "object",
      "properties": {
        "objective": {"type": "string", "description":
          "Natural-language description of the images to find, including subject, style, or context "
          "that should rank them, such as 'official headshot' or 'diagram, not a photo'."},
        "search_queries": {"type": "array", "items": {"type": "string"}, "description":
          "1-3 keyword queries of 2-6 words each, up to 200 characters each."},
      },
      "required": ["objective", "search_queries"],
    },
  }},
]
ARTIFACT_TOOLS = [
  {"type": "function", "function": {
    "name": "artifact_create",
    "description": (
      "Create a new artifact (a standalone file or document the user keeps and iterates on), or fully "
      "overwrite an existing one by reusing its `id`. Use for new files and for rewrites touching most of the "
      "file. For small or local changes use `artifact_edit` instead. `content` is the raw file content; "
      "do not wrap it in code fences."),
    "parameters": {"type": "object", "properties": {
      "id": {"type": "string", "description": "Short stable slug, e.g. 'meet-locs-service'."},
      "title": {"type": "string", "description": "Human-readable title or filename."},
      "type": {"type": "string", "enum": ["code", "markdown", "html", "text"]},
      "language": {"type": "string", "description": "Optional. Programming language when type is code."},
      "content": {"type": "string", "description": "Full raw content of the file."},
    }, "required": ["id", "title", "type", "content"]},
  }},
  {"type": "function", "function": {
    "name": "artifact_edit",
    "description": (
      "Edit an existing artifact by find-and-replace. `find` must match the current content EXACTLY "
      "(whitespace, indentation, newlines) and must occur exactly once, so include enough surrounding lines "
      "to make it unique (unless replace_all is true). Copy `find` verbatim from the artifact; never include "
      "line numbers. To insert, `find` an anchor and put anchor + new text in `replace`. Empty `replace` "
      "deletes. Multiple calls in one turn are applied in order, each against the result of the previous."),
    "parameters": {"type": "object", "properties": {
      "id": {"type": "string"},
      "find": {"type": "string", "description": "Exact existing text to replace."},
      "replace": {"type": "string", "description": "New text. May be empty."},
      "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
    }, "required": ["id", "find", "replace"]},
  }},
  {"type": "function", "function": {
    "name": "artifact_view",
    "description": "Return the current raw content of an artifact. Use only if unsure what it currently contains.",
    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
  }},
]
TOOLS = WEB_TOOLS + (ARTIFACT_TOOLS if ENABLE_ARTIFACTS else [])
ARTIFACT_TOOL_NAMES = {t["function"]["name"] for t in ARTIFACT_TOOLS} if ENABLE_ARTIFACTS else set()

# Sampling: model card, thinking mode. (llama.cpp names it repeat_penalty; see translation in chat().)
OPENAI_PARAMS = {"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty",
  "stop", "n", "seed", "logprobs", "prompt", "best_of", "echo", "logit_bias",
  "stream_options", "suffix", "user"}
INFERENCE_DEFAULTS = {"max_tokens": 32768, "temperature": 1.0, "top_p": 0.95,
  "top_k": 20, "min_p": 0.05, "presence_penalty": 0.0, "repeat_penalty": 1.0}
REASONING_EFFORTS = ("medium", "xhigh")  # "low" is unsupported by the model (behaves like xhigh)

PORT = int(os.environ.get('PORT', '8081'))
MODEL_NAME = 'bonsai-2-27b'
MLKEM_PATH = '/usr/local/bin/mlkem-generator'
# MASTER = base64.b64decode(os.environ['HS_MASTER_KEY'])   # 32 bytes, identical on every replica
# TOKEN_TTL = 15 * 60
SESSIONS: dict = {}  # sid -> {'dk', 'client_ek', 'ts'}; lives only in this process
SESSION_TTL = 15 * 60  # timeout
ACTIVE = 0    # in-flight chats; keeps /health stable during a long prefill

# ---------------- [SALAD_BROKEN_STREAM] cut-recovery helpers ----------------
# Tool results keyed by the tool_call id we send to the client. A resumed request carries the
# tool_calls (with ids) but not the results; we fill them in from here. In-memory only: a
# container restart empties it, and then the model is told the result is gone.
TOOL_RESULTS: dict = {}           # id -> {'name', 'content', 'ts'}
TOOL_RESULTS_MAX = 200
TOOL_RESULTS_TTL = 2 * 60 * 60

def _tool_results_prune():
  """Even though one believe no need TTL because my model seldom calls results so it won't stash too much of them
  in RAM, but anyways, since the AI already does it, then one'll let it be."""
  now = time.time()
  for k in [k for k, v in TOOL_RESULTS.items() if now - v['ts'] > TOOL_RESULTS_TTL]: TOOL_RESULTS.pop(k, None)
  while len(TOOL_RESULTS) > TOOL_RESULTS_MAX: TOOL_RESULTS.pop(next(iter(TOOL_RESULTS)), None)

async def _fill_tool_results(messages: list) -> list:
  '''A resumed request carries assistant message(s) with tool_calls but no tool results. Insert a tool
  message for every tool_call that has none, re-emitting the whole block in CALL order (Qwen-style
  templates pair tool results to calls by position, not id). Normal history never carries tool_calls,
  so for ordinary chats this is a no-op. Never re-runs a tool: a lost result becomes an explicit message.'''
  _tool_results_prune()
  out = []
  i, n = 0, len(messages)
  while i < n:
    m = messages[i]; out.append(m); i += 1
    if not (isinstance(m, dict) and m.get('role') == 'assistant' and m.get('tool_calls')): continue
    j = i
    while j < n and isinstance(messages[j], dict) and messages[j].get('role') == 'tool': j += 1
    existing = {messages[k].get('tool_call_id'): messages[k] for k in range(i, j)}
    tcs = m['tool_calls']
    ids = [(tc or {}).get('id') or '' for tc in tcs]
    if not any(t and t not in existing for t in ids): continue   # nothing to fill: block passes through as-is
    block = []
    for tc, tid in zip(tcs, ids):
      if tid and tid in existing: block.append(existing[tid]); continue
      if not tid: continue   # unkeyable call: nothing to attach (same as before)
      fn = (tc or {}).get('function') or {}
      name = fn.get('name') or ''
      stored = TOOL_RESULTS.get(tid)
      nm = name or 'tool'
      if stored is not None: content = stored['content']
      elif name in ARTIFACT_TOOL_NAMES:   # already applied before the cut; the client re-attaches the live artifacts
        content = json.dumps({'ok': True, 'note': 'already applied before the connection broke; the current artifact state is attached above.'})
      else: content = f'Error: the result of this {nm} call was lost when the connection broke. Call the tool again if you still need it.'
      print(f'[resume] filled tool result {nm} {tid} (stored={stored is not None})', flush=True)
      block.append({'role': 'tool', 'tool_call_id': tid, 'content': content})
    out.extend(block)   # one tool message per call, in CALL order
    out.extend(messages[k] for k in range(i, j) if messages[k].get('tool_call_id') not in set(ids))  # keep unrecognized extras
    i = j
  return out

def _nc(c: str) -> str:
  l = c.lower()
  return l if len(l) == 1 else c

def _norm_idx(s: str) -> list:
  '''[(lowercased alnum char, index in s)]: punctuation, whitespace and quote-style differences vanish.'''
  return [(_nc(c), i) for i, c in enumerate(s) if c.isalnum()]

class _EchoStrip:
  '''llama-server echoes the text we sent as a trailing assistant message at the start of the new stream
  (observed on plain and xyz: the whole sent text plus the first new word arrive in ONE delta). Drop the
  echoed part and keep everything after it, including that first new word, so the client just appends
  the continuation. If the start does not match, emit everything (duplicated text, never lost text).'''
  def __init__(self, sent: str):
    idx = _norm_idx(sent)
    self.ref = ''.join(c for c, _ in idx)
    self.tail = (len(sent) - (idx[-1][1] + 1)) if idx else 0   # trailing non-alnum chars we sent
    self.buf, self.got = '', ''
    self.done = not self.ref

  def feed(self, text: str) -> str:
    if self.done: return text
    self.buf += text
    self.got += ''.join(_nc(c) for c in text if c.isalnum())
    n = len(self.ref)
    if len(self.got) < n:
      return '' if self.ref.startswith(self.got) else self._give_up()   # still consistent with an echo: keep holding
    if self.got[:n] != self.ref: return self._give_up()
    nb = _norm_idx(self.buf)
    j, k = nb[n - 1][1] + 1, 0                      # index just after the last echoed letter or digit
    while j < len(self.buf) and k < self.tail and not self.buf[j].isalnum(): j += 1; k += 1   # echoed trailing punctuation/space
    out, self.buf, self.done = self.buf[j:], '', True
    return out

  def _give_up(self) -> str:
    print('[echo] no/different echo: emitting text as is', flush=True)
    out, self.buf, self.done = self.buf, '', True
    return out

  def flush(self) -> str:
    # out, self.buf, self.done = self.buf, '', True
    held_echo = (not self.done) and self.ref.startswith(self.got)   # everything held so far is still pure echo
    out, self.buf, self.done = ('' if held_echo else self.buf), '', True
    return out
# ---------------- end [SALAD_BROKEN_STREAM] ----------------

# ---------------- [RESUMABLE_STREAM] background generation + replay buffer ----------------
GENS: dict = {}            # gen_id -> _Gen. RAM only: ciphertext frames, never prompts or history
MAX_GENS = 8
PING_EVERY = 5.0           # s of quiet before a connection-local ping (client stall timer is 10 s)
ROTATE_AFTER = 24 * 3600.0  # s; Salad needed 110 (gateway ~180 s sweep). Modal has no such sweep, so effectively off
CONSUMER_TTL = 120.0       # s with no listener: cancel the generation (stops GPU burn from abandoned tabs)
GEN_MAX = 45 * 60          # s; hard cap on one generation
DONE_TTL = 10 * 60         # s; keep a finished buffer for a late reattach
_janitor_task = None

def _sse(obj: dict) -> str: return 'data: ' + json.dumps(obj) + '\n\n'

class _Gen:
  def __init__(self, gid: str):
    self.gid = gid
    self.frames: list = []     # ready-to-send SSE strings, append-only; frame number (seq) = index + 1
    self.seq = 0
    self.done = False
    self.task = None
    self.tailers = 0
    self.started = self.last_consumer = time.monotonic()
    self.done_at = None
    self.cond = asyncio.Condition()

async def _gen_push(g, ev: str):
  line = ev.strip()
  if not line.startswith('data: '): return
  body = line[6:]
  async with g.cond:
    g.seq += 1
    if body != '[DONE]':
      try:
        obj = json.loads(body); obj['seq'] = g.seq; body = json.dumps(obj)
      except Exception: pass
    g.frames.append('data: ' + body + '\n\n')
    g.cond.notify_all()

async def _gen_finish(g, fail=None):
  async with g.cond:
    if g.done: return
    if fail:
      g.seq += 1
      g.frames.append(_sse({'type': 'fail', 'message': fail, 'seq': g.seq}))
    g.done = True; g.done_at = time.monotonic()
    g.cond.notify_all()

async def _gen_worker(g, agen):
  '''Runs outside any request: no client socket can cancel it. agen is your existing generate().'''
  global ACTIVE
  ACTIVE += 1
  t0 = time.monotonic(); why = 'finished'; saw_done = False
  try:
    async for ev in agen:
      if ev.strip() == 'data: [DONE]': saw_done = True
      await _gen_push(g, ev)
    await _gen_finish(g, None if saw_done else 'ended without completion')
  except asyncio.CancelledError:
    why = 'CANCELLED'; await _gen_finish(g, 'cancelled'); raise
  except Exception as e:
    why = f'ERROR {e!r}'; await _gen_finish(g, 'generation failed')
  finally:
    ACTIVE -= 1
    if not g.done: g.done = True; g.done_at = time.monotonic()
    print(f'[gen end] {why} after {time.monotonic() - t0:.1f}s, {g.seq} frames', flush=True)

async def _gen_tail(g, since: int):
  '''The push channel. Replays frames after `since`, then follows live. A client disconnect ends only this
  generator. meta, ping and rotate are connection-local: never buffered, carry no seq.'''
  g.tailers += 1
  t_start = time.monotonic()
  try:
    yield _sse({'type': 'meta', 'gen': g.gid, 'len': len(g.frames), 'done': g.done})
    i = max(0, since)
    while True:
      g.last_consumer = time.monotonic()
      async with g.cond:
        if i >= len(g.frames) and not g.done:
          try: await asyncio.wait_for(g.cond.wait_for(lambda: len(g.frames) > i or g.done), timeout=PING_EVERY)
          except asyncio.TimeoutError: pass
      while i < len(g.frames):       # appends only grow the list, so this drain cannot race the worker
        yield g.frames[i]; i += 1
      if g.done and i >= len(g.frames): return
      if time.monotonic() - t_start > ROTATE_AFTER:
        yield 'data: {"type":"rotate"}\n\n'; return
      if i >= len(g.frames): yield 'data: {"type":"ping"}\n\n'
  finally:
    g.tailers -= 1

async def _janitor():
  while True:
    await asyncio.sleep(15)
    now = time.monotonic()
    for gid, g in list(GENS.items()):
      if not g.done:
        if (g.tailers == 0 and now - g.last_consumer > CONSUMER_TTL) or now - g.started > GEN_MAX:
          print(f'[janitor] cancelling {gid[:8]}: no listener or over the time cap', flush=True)
          if g.task is not None: g.task.cancel()
      elif g.done_at is not None and now - g.done_at > DONE_TTL:
        GENS.pop(gid, None)
# ---------------- end [RESUMABLE_STREAM] ----------------

# ---------------- [STATS] per-generation statistics ----------------
STATS_PROGRESS = True      # ask llama.cpp for prompt_progress chunks (needed for the prefill start/end speed)

class _Round:
  '''One llama-server call. A reply that uses tools is several rounds.'''
  def __init__(self):
    self.t0 = time.monotonic(); self.t_first = None; self.t_end = None
    self.timings = None        # llama.cpp timings object (last chunk), when present
    self.samples = []          # [(processed tokens, ms since the prompt started)] from return_progress
    self.pt = self.ct = 0      # prompt / completion tokens from the usage chunk

  def see(self, chunk):
    '''Call for EVERY chunk, before any filtering.'''
    tm = getattr(chunk, 'timings', None)
    if isinstance(tm, dict): self.timings = tm
    pp = getattr(chunk, 'prompt_progress', None)
    if isinstance(pp, dict):
      try: self.samples.append((int(pp['processed']), float(pp['time_ms'])))
      except Exception: pass
    u = getattr(chunk, 'usage', None)
    if u: self.pt, self.ct = u.prompt_tokens, u.completion_tokens

  def token(self):
    if self.t_first is None: self.t_first = time.monotonic()

def _prefill_curve(samples):
  '''(first_tps, last_tps, curve) from cumulative samples. Speeds come from differences, so a cached prefix cancels out.'''
  pts = []
  for (p1, t1), (p2, t2) in zip(samples, samples[1:]):
    if t2 > t1 and p2 > p1: pts.append((p2, (p2 - p1) / (t2 - t1) * 1000.0))
  if not pts: return None, None, []
  step = max(1, len(pts) // 16)
  return round(pts[0][1], 1), round(pts[-1][1], 1), [[p, round(v, 1)] for p, v in pts[::step]]

def _stats(rounds, t_request):
  pre_tok = pre_ms = dec_tok = dec_ms = cached = drafted = accepted = 0.0
  have_timings = False; ttft = None
  for k, r in enumerate(rounds):
    tm = r.timings or {}
    if tm: have_timings = True
    # prefill: engine numbers when present, else wall clock up to the first generated token
    pre_tok += tm.get('prompt_n', r.pt) or 0
    pre_ms += tm.get('prompt_ms') or ((r.t_first - r.t0) * 1000 if r.t_first else 0)
    cached += tm.get('cache_n', 0) or 0
    dec_tok += tm.get('predicted_n', r.ct) or 0
    dec_ms += tm.get('predicted_ms') or ((r.t_end - r.t_first) * 1000 if (r.t_first and r.t_end) else 0)
    drafted += tm.get('draft_n', 0) or 0
    accepted += tm.get('draft_n_accepted', 0) or 0
    if k == 0 and r.t_first: ttft = (r.t_first - t_request) * 1000
  best = max(rounds, key=lambda r: len(r.samples)) if rounds else None
  first_tps, last_tps, curve = _prefill_curve(best.samples) if best else (None, None, [])
  return {
    'engine_timings': have_timings,
    'ttft_ms': round(ttft) if ttft is not None else None,
    'prefill': {'tokens': int(pre_tok), 'cached': int(cached), 'ms': round(pre_ms),
      'avg_tps': round(pre_tok / pre_ms * 1000, 1) if pre_ms > 0 and pre_tok > 0 else None,   # [STATS] tok>0 guard: an all-tool-call round would otherwise show 0 t/s
      'first_tps': first_tps, 'last_tps': last_tps, 'curve': curve},
    'decode': {'tokens': int(dec_tok), 'ms': round(dec_ms),
      'avg_tps': round(dec_tok / dec_ms * 1000, 1) if dec_ms > 0 and dec_tok > 0 else None},
    'spec': ({'drafted': int(drafted), 'accepted': int(accepted), 'rate': round(accepted / drafted, 3)} if drafted else None),
    'rounds': len(rounds),
  }
# ---------------- end [STATS] ----------------

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

async def _run_mlkem(*args: str, input_data: Optional[str] = None) -> dict:
  proc = await asyncio.to_thread(subprocess.run, [MLKEM_PATH, *args],
    input=input_data, capture_output=True, text=True)
  if proc.returncode != 0:
    try: detail = json.loads(proc.stdout).get("error", proc.stderr.strip())
    except Exception: detail = proc.stderr.strip() or "Unknown MLKem Error"
    raise HTTPException(400, f"[mlkem]: {detail}")
  try: return json.loads(proc.stdout)
  except json.JSONDecodeError: raise HTTPException(500, "Invalid JSON from mlkem")

# def _seal(dk: str, client_ek: str) -> str:
#     nonce = os.urandom(12)
#     body = json.dumps({'dk': dk, 'cek': client_ek, 'exp': int(time.time()) + TOKEN_TTL}).encode()
#     return base64.b64encode(nonce + ChaCha20Poly1305(MASTER).encrypt(nonce, body, None)).decode()

# def _open(token: str):
#     try:
#         raw = base64.b64decode(token)
#         body = json.loads(ChaCha20Poly1305(MASTER).decrypt(raw[:12], raw[12:], None))
#     except Exception:
#         raise HTTPException(400, 'bad handshake token')
#     if body['exp'] < time.time(): raise HTTPException(400, 'handshake token expired')
#     return body['dk'], body['cek']

@api.post('/v1/handshake')
async def handshake(request: dict, x_session_id: Optional[str] = Header(default='default')):
  client_ek = request.get('ek')
  if not client_ek: raise HTTPException(400, "Missing 'ek'")
  keygen = await _run_mlkem('keygen')
  now = time.time()
  for k in [k for k, v in SESSIONS.items() if now - v['ts'] > SESSION_TTL]: del SESSIONS[k]
  SESSIONS[x_session_id] = {'dk': keygen['dk'], 'client_ek': client_ek, 'ts': now}
  return {'ek': keygen['ek']}

@api.post("/chat")
async def chat(request: dict, x_session_id: Optional[str] = Header(default='default'), 
  x_gen_id: Optional[str] = Header(default=None)  # [RESUMABLE_STREAM]
):   
  async def _():
    if not await _llama_ready(): raise HTTPException(503, "model loading")
    # token = request.get("token")
    # if not token: raise HTTPException(400, "Missing token")
    # server_dk, client_ek = _open(token)
    session = SESSIONS.pop(x_session_id, None)  # Single use. 
    if not session: raise HTTPException(400, "No handshake found. Call /v1/handshake first.")
    server_dk, client_ek = session['dk'], session['client_ek']

    ct = request.get("ct")
    em = request.get("em")
    if not ct or not em: raise HTTPException(400, "Missing ct/em.")

    retval = await _run_mlkem("decapsulate-and-decrypt", "--from-stdin",
      input_data=json.dumps({"dk": server_dk, "ciphertext": ct, "encrypted_message": em}))
    payload = json.loads(retval["plaintext"])
    messages = payload.pop("messages", [])
    if not isinstance(messages, list): raise HTTPException(400, "plaintext must be JSON with 'messages' array.")
    reasoning_effort = payload.pop("reasoning_effort", "xhigh")  # default xhigh
    if reasoning_effort not in REASONING_EFFORTS: reasoning_effort = "xhigh"

    # Artifacts: popped BEFORE params is built, so they never leak into sampling params.
    raw_artifacts = payload.pop("artifacts", None) or {}
    if not isinstance(raw_artifacts, dict): raise HTTPException(400, "'artifacts' must be an object keyed by id.")
    artifacts = _norm_artifacts(raw_artifacts) if ENABLE_ARTIFACTS else {}
    if artifacts: messages = inject_artifacts(messages, artifacts)

     # [SALAD_BROKEN_STREAM] resumed request: fill missing tool results, detect a trailing partial assistant message
    try: messages = await _fill_tool_results(messages)   # failsafe: never fail the request over this
    except Exception as e: print(f'[resume] tool fill failed, continuing without: {e!r}', flush=True)
    echo_r = echo_c = ''
    if messages and isinstance(messages[-1], dict) and messages[-1].get('role') == 'assistant' and not messages[-1].get('tool_calls'):
      echo_r = messages[-1].get('reasoning_content') or ''
      echo_c = messages[-1].get('content') or ''
      if not isinstance(echo_r, str): echo_r = ''
      if not isinstance(echo_c, str): echo_c = ''
      print(f'[resume] continuing a partial assistant message: {len(echo_r)} reasoning chars, {len(echo_c)} content chars', flush=True)

    params = {**INFERENCE_DEFAULTS, **payload}  # payload overrides defaults.
    # vLLM/SGLang name -> llama.cpp name
    if "repetition_penalty" in params: params["repeat_penalty"] = params.pop("repetition_penalty")
    # Unknown params could raise in the OpenAI client, so split them into extra_body.
    standard = {k: v for k, v in params.items() if k in OPENAI_PARAMS}
    extra    = {k: v for k, v in params.items() if k not in OPENAI_PARAMS}
    # The model reads reasoning_effort from the chat template kwargs; the top-level field is sent too.
    extra_body = {**extra, "chat_template_kwargs": {"reasoning_effort": reasoning_effort}, "return_progress": STATS_PROGRESS}   # [STATS]

    # Remove dk, keep client_ek until stream pre-key is computed.
    encap_res = await _run_mlkem("encapsulate-and-encrypt", "--from-stdin",
      input_data=json.dumps({"ek": client_ek, "plaintext": ""}))
    resp_ct, resp_ss = encap_res["ciphertext"], encap_res["shared_secret"]
    resp_em = encap_res["encrypted_message"]

    async def generate():
      r_buf, t_buf = [], []
      first_sent = False
      strip_r = _EchoStrip(echo_r) if echo_r else None   # [SALAD_BROKEN_STREAM]
      strip_c = _EchoStrip(echo_c) if echo_c else None

      async def _emit_json(typ: str, obj: dict):
        nonlocal first_sent
        enc = await _run_mlkem("encrypt", "--from-stdin",
          input_data=json.dumps({"key": resp_ss, "plaintext": json.dumps(obj)}))
        pkt = {"type": typ, "content": enc["ciphertext"]}
        if not first_sent:
          pkt["ct"], pkt["em"] = resp_ct, resp_em
          first_sent = True
        yield f"data: {json.dumps(pkt)}\n\n"

      async def _emit(buf: list, typ: str):
        nonlocal first_sent
        if not buf: return
        enc = await _run_mlkem("encrypt", "--from-stdin",
          input_data=json.dumps({"key": resp_ss, "plaintext": "".join(buf)}))
        buf.clear()
        pkt = {"type": typ, "content": enc["ciphertext"]}
        if not first_sent:
          pkt["ct"], pkt["em"] = resp_ct, resp_em
          first_sent = True
        yield f"data: {json.dumps(pkt)}\n\n"

      loop_start = time.monotonic()
      tot_prompt_toks = tot_compl_toks = 0
      stat_rounds = []   # [STATS] one _Round per llama call

      while True:
        acc_tool_calls: dict = {}
        finish_reason = None
        rs = _Round(); stat_rounds.append(rs)   # [STATS]
        stream = await client.chat.completions.create(model=MODEL_NAME, messages=messages,
          tools=TOOLS, tool_choice="auto", stream=True, stream_options={"include_usage": True},
          reasoning_effort=reasoning_effort, extra_body=extra_body, **standard)
        async for chunk in stream:
          rs.see(chunk)   # [STATS] progress/timings/usage chunks may have no choices: look before filtering
          if not getattr(chunk, 'choices', None):
            if chunk.usage:  # usage arrives in a final chunk with empty choices.
              tot_prompt_toks += chunk.usage.prompt_tokens
              tot_compl_toks  += chunk.usage.completion_tokens
            continue
          choice = chunk.choices[0]
          delta = choice.delta
          finish_reason = choice.finish_reason or finish_reason

          # llama-server (deepseek-style reasoning format) uses reasoning_content.
          # `if raw:` (not raw.strip()) so newline-only deltas between paragraphs aren't dropped.
          raw = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
          if raw and strip_r is not None: raw = strip_r.feed(raw)   # [SALAD_BROKEN_STREAM] drop the echoed copy of what we sent
          if raw or delta.content or delta.tool_calls: rs.token()   # [STATS] the first generated token ends the prefill
          if raw:
            async for ev in _emit(t_buf, "text"): yield ev  # flush text if switching back.
            r_buf.append(raw)
            if len(r_buf) >= SEND_PER_BATCH:
              async for ev in _emit(r_buf, "reasoning"): yield ev

          dc = delta.content   # [SALAD_BROKEN_STREAM]
          if dc and strip_r is not None and not strip_r.done:   # content started: nothing more will echo in reasoning
            tail = strip_r.flush()
            if tail: r_buf.append(tail)
          if dc and strip_c is not None: dc = strip_c.feed(dc)
          if dc:
            async for ev in _emit(r_buf, "reasoning"): yield ev  # flush reasoning on switch.
            t_buf.append(dc)
            if len(t_buf) >= SEND_PER_BATCH:
              async for ev in _emit(t_buf, "text"): yield ev

          # Accumulate for tool calls.
          if delta.tool_calls:
            for tc in delta.tool_calls:
              i = tc.index
              if i not in acc_tool_calls: acc_tool_calls[i] = {"id": "", "name": "", "arguments": ""}
              if tc.id: acc_tool_calls[i]["id"] = tc.id
              fn = tc.function
              if fn and fn.name: acc_tool_calls[i]["name"] = fn.name
              if fn and fn.arguments: acc_tool_calls[i]["arguments"] += fn.arguments

        rs.t_end = time.monotonic()   # [STATS] end of this llama call
        if strip_r is not None:   # [SALAD_BROKEN_STREAM] the echo only exists on the first llama call
          tail = strip_r.flush()
          if tail: r_buf.append(tail)
        if strip_c is not None:
          tail = strip_c.flush()
          if tail: t_buf.append(tail)
        strip_r = strip_c = None
        # Flush remaining buffers before tool dispatch or finish.
        async for ev in _emit(r_buf, "reasoning"): yield ev
        async for ev in _emit(t_buf, "text"):      yield ev

        if finish_reason == "length":
          async for ev in _emit_json("error", {"message": "max_tokens reached; output truncated."}): yield ev

        # llama.cpp may report "stop" even when tool calls were parsed, so key on the calls themselves.
        if acc_tool_calls and finish_reason != "length":
          for tc in acc_tool_calls.values():
            if not tc["id"]: tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
          # Append the assistant's tool_call message to keep history valid.
          messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": tc["id"], "type": "function",
              "function": {"name": tc["name"], "arguments": tc["arguments"]}
            } for tc in acc_tool_calls.values()
          ]})
          tool_msgs: dict = {}  # keyed by call index: results must be appended in CALL order,
          valid_tcs = []        # because Qwen-style templates pair tool results to calls by position, not id.
          for idx, tc in acc_tool_calls.items():
            try: args = json.loads(tc["arguments"]) if tc["arguments"].strip() else {}
            except json.JSONDecodeError: args = None

            # Check for tool call hallucination and tell the model it is hallucinating.
            error = None
            if not isinstance(args, dict):
              error, args = "arguments were not valid JSON.", {}
            elif tc["name"] not in {t["function"]["name"] for t in TOOLS}:
              error = f"unknown tool '{tc['name']}'."
            elif tc["name"] == "web_fetch":
              if not args.get("urls") or not args.get("objective"):
                error = "web_fetch requires 'urls' (list of string) AND 'objective' (string). Got empty arguments."
            elif tc["name"] == "web_search":
              if not args.get("search_queries") or not args.get("objective"):
                error = "web_search requires 'search_queries' (list of string) AND 'objective' (string). Got empty arguments."

            # Notify client. Strip bulky fields for artifact tools; content arrives via the "artifact" event.
            shown = ({k: v for k, v in args.items() if k in ("id", "title", "type", "language")}
              if tc["name"] in ARTIFACT_TOOL_NAMES else args)
            async for ev in _emit_json("tool_call", {"name": tc["name"], "arguments": shown, "id": tc["id"]}): yield ev   # [SALAD_BROKEN_STREAM]

            if error:
              print(f"[tool_error] {tc['name']} bad args: {args}", flush=True)
              tool_msgs[idx] = {"role": "tool", "tool_call_id": tc["id"], "content": f"Error: {error}"}
            elif tc["name"] in ARTIFACT_TOOL_NAMES:
              # Local + cheap + order-dependent: run sequentially, in call order.
              result, snap = artifact_exec(tc["name"], args, artifacts)
              tool_msgs[idx] = {"role": "tool", "tool_call_id": tc["id"], "content": result}
              if snap:
                async for ev in _emit_json("artifact", snap): yield ev
            else:
              valid_tcs.append((idx, tc))  # only gather valid ones.

          # Web tools run in parallel, outside the loop. NOT for artifacts.
          async def run_one(tc):
            try: content = await asyncio.to_thread(run_rust_binary, json.dumps(tc))
            except RuntimeError as e: content = json.dumps({"error": str(e)})
            return {"role": "tool", "tool_call_id": tc["id"], "content": compact_context(content, tc['name'])}

          if valid_tcs:
            results = await asyncio.gather(*[run_one(tc) for _, tc in valid_tcs])
            for (idx, _), res in zip(valid_tcs, results): tool_msgs[idx] = res
          _tool_results_prune()   # [SALAD_BROKEN_STREAM] remember results by tool_call id for cut recovery
          for _i, _tc in acc_tool_calls.items():
            if _i in tool_msgs: TOOL_RESULTS[_tc["id"]] = {"name": _tc["name"], "content": tool_msgs[_i]["content"], "ts": time.time()}
          messages.extend(tool_msgs[i] for i in sorted(tool_msgs))
        else:
          elapsed = time.monotonic() - loop_start
          # NOTE: tool-call rounds re-send the growing history, so prompt tokens are cumulative across
          # rounds. That's accurate: it reflects what was actually processed.
          usage = {'type': 'usage', 'prompt_tokens': tot_prompt_toks,
            'completion_tokens': tot_compl_toks, 'total_tokens': tot_prompt_toks + tot_compl_toks,
            'elapsed_s': round(elapsed, 1),
            'tps': round(tot_compl_toks / elapsed, 1) if elapsed else 0,
            'model': MODEL_NAME}   # [STATS] so the client's CSV model column needs no settings lookup
          try: usage['stats'] = _stats(stat_rounds, loop_start)   # [STATS] never let statistics break a reply
          except Exception as e: print(f'[stats] failed: {e!r}', flush=True)
          yield f"data: {json.dumps(usage)}\n\n"
          yield "data: [DONE]\n\n"
          break
    # [RESUMABLE_STREAM] the generation runs in a task no socket owns; this request only tails its buffer
    gid = x_gen_id if (x_gen_id and re.fullmatch(r'[0-9a-fA-F-]{16,64}', x_gen_id)) else uuid.uuid4().hex
    if gid in GENS: raise HTTPException(409, 'generation id already used')
    for og in GENS.values():   # --parallel 1 means one slot: a leftover generation would block this one
      if not og.done and og.task is not None: og.task.cancel()
    while len(GENS) >= MAX_GENS: GENS.pop(next(iter(GENS)), None)
    global _janitor_task
    if _janitor_task is None or _janitor_task.done(): _janitor_task = asyncio.create_task(_janitor())
    g = _Gen(gid)
    GENS[gid] = g
    g.task = asyncio.create_task(_gen_worker(g, generate()))
    return StreamingResponse(_gen_tail(g, 0), media_type="text/event-stream",
      headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
  return await _guarded(_)

@api.get("/v1/stream")
async def stream_resume(gen: str, since: int = 0): # [RESUMABLE_STREAM]
  g = GENS.get(gen)
  if g is None: raise HTTPException(404, 'no such generation')   # restarted or evicted: the client falls back to a full resend
  return StreamingResponse(_gen_tail(g, max(0, since)), media_type='text/event-stream',
    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# =================================================================
# Tools (web_search and web_fetch using rust binaries baked into image.)
def run_rust_binary(payload: str) -> str:
  """payload is just the tc that you pass in, with id, name, and arguments."""
  result = subprocess.run(["/usr/local/bin/web-searcher"], input=payload, capture_output=True, text=True, timeout=60)
  if result.returncode != 0: raise RuntimeError(f"Binary error: {result.stderr.strip()}")
  return result.stdout.strip()


def compact_context(json_str: str, tool_type: str) -> str:
  """Compact the context before passing back to the LLM.
  tool_type is either web_fetch or web_search; that determines, based on a standardized return
  struct, how we reach the context for compacting it. The library is pure python (GIL-bound),
  runs in under a second, and multiprocessing would cost more than it saves, so it runs sequentially."""
  from claw_compactor.fusion.engine import FusionEngine
  engine = FusionEngine(enable_rewind=False)  # Init takes 0.03 seconds.

  if tool_type == "web_search":
    results = json.loads(json_str)  # Vec<SearchResOl>
    compacted = []
    for i, item in enumerate(results):
      cdata = engine.compress(text=item.get('content', ''))
      cstats = cdata['stats']
      print(f"[compression] (chars) {cstats['original_chars']} -> {cstats['compressed_chars']} "
        f"({cstats['reduction_pct']}% in {cstats['total_timing_ms']} ms)")
      compressed = cdata['compressed']
      title = item.get('title', 'Untitled')
      url = item.get('url', '')
      publish_date = item.get('publish_date', '')
      date_str = f" ({publish_date})" if publish_date else ""

      compacted.append(f"Result {i+1}: {title}{date_str}\n"
        f"URL: {url}\n\n"
        f"{compressed}\n")
    return "\n---\n".join(compacted)

  elif tool_type == "web_fetch":
    results = json.loads(json_str)  # Vec<FetchRetVal>
    compacted = []
    for i, item in enumerate(results):
      cdata = engine.compress(text=item.get('content', ''))
      cstats = cdata['stats']
      print(f"[compression] (chars) {cstats['original_chars']} -> {cstats['compressed_chars']} "
        f"({cstats['reduction_pct']}% in {cstats['total_timing_ms']} ms)")
      compressed = cdata['compressed']
      title = item.get('title', "Untitled")
      links = item.get('links', [])
      publish_date = item.get('publish_date', '')
      date_str = f" ({publish_date})" if publish_date else ""

      full = item.get('full_content', None)
      compressed_full = ""
      if full:
        cdata = engine.compress(text=full)
        cstats = cdata['stats']
        print(f"[compression] [full_content] (chars) {cstats['original_chars']} "
          f"-> {cstats['compressed_chars']} ({cstats['reduction_pct']}% "
          f"in {cstats['total_timing_ms']} ms)")
        compressed_full = cdata['compressed']

      parts = [f"Page {i+1}: {title}{date_str}", f"Content:\n{compressed}"]
      if compressed_full: parts.append(f"Full Content:\n{compressed_full}")
      if links: parts.append("Links:\n" + "\n".join(f"- {link}" for link in links))
      compacted.append("\n".join(parts))
    return "\n---\n".join(compacted)

  return json_str


def _norm_artifacts(raw: dict) -> dict:
  """Validate what the client sent: drop malformed entries, coerce version, strip quotes from titles."""
  out = {}
  for k, v in raw.items():
    if not (isinstance(v, dict) and isinstance(v.get("content"), str)): continue
    try: ver = int(v.get("version", 1))
    except (TypeError, ValueError): ver = 1
    out[k] = {**v, "version": ver, "title": str(v.get("title", "")).replace('"', "'")}
  return out


# ==========================================================
# Tools (artifacts editor). Unit-tested separately; unchanged from the SGLang backend.

def _lines(text: str) -> int: return text.count("\n") + 1 if text else 0
def _line_of(text: str, idx: int) -> int: return text.count("\n", 0, idx) + 1

def render_artifacts(artifacts: dict) -> str:
  if not artifacts: return ""
  out = ["<artifacts>",
    "Current state of the user's artifacts. Authoritative: includes any manual edits by the user. ",
    "Modify with artifact_edit (copy `find` exactly from here). Do not paste artifact contents into your reply."]
  for aid, a in artifacts.items():
    out.append(f'<artifact id="{aid}" title="{a.get("title", "")}" type="{a.get("type", "text")}" '
      f'version="{a["version"]}" lines="{_lines(a["content"])}">')
    out.append(a["content"])
    out.append("</artifact>")
  out.append("</artifacts>")
  return "\n".join(out)

def inject_artifacts(messages: list, artifacts: dict):
  """Put the artifact block as the first text part of the LAST user message 
  (copy, no mutation)."""
  block = render_artifacts(artifacts)
  if not block: return messages
  messages = [dict(m) for m in messages]
  for m in reversed(messages):
    if m.get("role") != "user": continue
    c = m.get("content")
    parts = [{"type": "text", "text": c}] if isinstance(c, str) else list(c or [])
    m["content"] = [{"type": "text", "text": block}] + parts
    return messages
  messages.append({"role": "user", "content": [{"type": "text", "text": block}]})
  return messages

def _miss_hint(text: str, find: str) -> str: 
  first = next((l.strip() for l in find.splitlines() if l.strip()), "")
  tl = text.split("\n")
  hits = [i + 1 for i, l in enumerate(tl) if first and first in l]
  if hits:
    return (f"`find` not found verbatim, but its first line appears at line(s) {hits[:5]}. The mismatch is in "
      "whitespace/indentation or a later line. Copy the text exactly from the artifact.")
  close = difflib.get_close_matches(first, [l.strip() for l in tl], n=1, cutoff=0.6)
  return ("`find` not found." + (f" Closest line: {close[0]!r}." if close else "") +
    " Copy exactly from the artifact or call artifact_view.")

def artifact_exec(name: str, args: dict, artifacts: dict):
  """Returns (tool_result_str, snapshot_event_or_None). Mutates `artifacts`."""
  def err(msg): return json.dumps({"ok": False, "error": msg}), None
  aid = args.get("id")
  if not isinstance(aid, str) or not aid: return err("'id' (string) is required.")

  if name == "artifact_create":
    content, title, typ = args.get("content"), args.get("title"), args.get("type")
    if not isinstance(content, str) or not isinstance(title, str) or not isinstance(typ, str):
      return err("artifact_create requires string 'title', 'type', 'content'.")
    ver = artifacts.get(aid, {}).get("version", 0) + 1
    artifacts[aid] = {"title": title.replace('"', "'"), "type": typ, "language": args.get("language", ""),
      "version": ver, "content": content}
    snap = {"id": aid, **{k: artifacts[aid][k] for k in ("title", "type", "language", "version", "content")}}
    return json.dumps({"ok": True, "id": aid, "version": ver, "lines": _lines(content)}), snap

  a = artifacts.get(aid)
  if a is None: return err(f"No artifact '{aid}'. Existing ids: {list(artifacts) or 'none'}.")
  if name == "artifact_view": 
    return json.dumps({"ok": True, "version": a["version"], "content": a["content"]}), None

  # artifact_edit
  find, replace = args.get("find"), args.get("replace", "")
  if not isinstance(find, str) or not find: return err("'find' must be a non-empty string.")
  if not isinstance(replace, str): return err("'replace' must be a string.")
  replace_all = bool(args.get("replace_all", False))
  text = a["content"]
  n = text.count(find)
  if n == 0: return err(_miss_hint(text, find))
  if n > 1 and not replace_all:  # Multiple counts found. 
    where, pos = [], 0
    for _ in range(min(n, 5)):
      i = text.find(find, pos); where.append(_line_of(text, i)); pos = i + len(find)
    return err(f"`find` matches {n} places (lines {where}). Add a few lines above and below to make it unique, or set " 
      "replace_all=true if you're suppose to replace all of them.")

  idx = text.find(find)
  new = text.replace(find, replace) if replace_all else text[:idx] + replace + text[idx + len(find):]
  a["content"], a["version"] = new, a["version"] + 1
  start = _line_of(new, idx)
  nl = new.split("\n")
  lo, hi = max(0, start - 3), min(len(nl), start - 1 + replace.count("\n") + 1 + 2)
  result = {"ok": True, "id": aid, "version": a["version"], "lines": len(nl),
    "occurrences_replaced": n if replace_all else 1, "first_changed_line": start,
    "preview": "\n".join(nl[lo:hi])}  # raw, no line numbers, so nothing to leak into later `find`s.
  snap = {"id": aid, **{k: a[k] for k in ("title", "type", "language", "version", "content") if k in a},
    "change": {"find": find, "replace": replace, "replace_all": replace_all}}
  return json.dumps(result), snap

if __name__ == '__main__':
  import socket, uvicorn, urllib.request
  if os.environ.get('GATE_ON_LLAMA') == '1':
    _t0 = time.time()
    while True:
      try:
        if urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2).status == 200: break
      except Exception: pass
      if time.time() - _t0 > float(os.environ.get('GATE_TIMEOUT', '850')):
        raise SystemExit('llama-server never became healthy')   # entrypoint.sh sees a child die and exits non-zero
      time.sleep(2)
    print(f'[gate] llama healthy after {time.time() - _t0:.0f}s, binding', flush=True)
  socks = []
  for fam, addr in ((socket.AF_INET, '0.0.0.0'), (socket.AF_INET6, '::')):
    try:
      s = socket.socket(fam, socket.SOCK_STREAM)
      s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      if fam == socket.AF_INET6: s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
      s.bind((addr, 8000))
      socks.append(s)
    except OSError as e: print(f'[bind] {addr} skipped: {e}', flush=True)
  if not socks: raise SystemExit('no listen socket')
  uvicorn.Server(uvicorn.Config(api, log_level='info')).run(sockets=socks)