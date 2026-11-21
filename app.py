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

@api.post("/chat")
    async def chat(request: dict, x_session_id: Optional[str] = Header(default="default")):
      async def _():
        session = await _get_session(x_session_id)
        server_dk = session.pop("dk", None)
        client_ek = session.get("client_ek")
        if not server_dk or not client_ek: raise HTTPException(400, "No handshake found. Call /v1/handshake first.")

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

        params = {**INFERENCE_DEFAULTS, **payload}  # payload overrides defaults.
        # vLLM/SGLang name -> llama.cpp name
        if "repetition_penalty" in params: params["repeat_penalty"] = params.pop("repetition_penalty")
        # Unknown params could raise in the OpenAI client, so split them into extra_body.
        standard = {k: v for k, v in params.items() if k in OPENAI_PARAMS}
        extra    = {k: v for k, v in params.items() if k not in OPENAI_PARAMS}
        # The model reads reasoning_effort from the chat template kwargs; the top-level field is sent too.
        extra_body = {**extra, "chat_template_kwargs": {"reasoning_effort": reasoning_effort}}

        # Remove dk, keep client_ek until stream pre-key is computed.
        encap_res = await _run_mlkem("encapsulate-and-encrypt", "--from-stdin",
          input_data=json.dumps({"ek": client_ek, "plaintext": ""}))
        resp_ct, resp_ss = encap_res["ciphertext"], encap_res["shared_secret"]
        resp_em = encap_res["encrypted_message"]
        session.pop("client_ek", None)
        await _save_session(x_session_id, session)

        async def generate():
          r_buf, t_buf = [], []
          first_sent = False

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

          while True:
            acc_tool_calls: dict = {}
            finish_reason = None
            stream = await client.chat.completions.create(model=MODEL_NAME, messages=messages,
              tools=TOOLS, tool_choice="auto", stream=True, stream_options={"include_usage": True},
              reasoning_effort=reasoning_effort, extra_body=extra_body, **standard)
            async for chunk in stream:
              if not chunk.choices:
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
              if raw:
                async for ev in _emit(t_buf, "text"): yield ev  # flush text if switching back.
                r_buf.append(raw)
                if len(r_buf) >= SEND_PER_BATCH:
                  async for ev in _emit(r_buf, "reasoning"): yield ev

              if delta.content:
                async for ev in _emit(r_buf, "reasoning"): yield ev  # flush reasoning on switch.
                t_buf.append(delta.content)
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
                async for ev in _emit_json("tool_call", {"name": tc["name"], "arguments": shown}): yield ev

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
              messages.extend(tool_msgs[i] for i in sorted(tool_msgs))
            else:
              elapsed = time.monotonic() - loop_start
              # NOTE: tool-call rounds re-send the growing history, so prompt tokens are cumulative across
              # rounds. That's accurate: it reflects what was actually processed.
              usage = {'type': 'usage', 'prompt_tokens': tot_prompt_toks,
                'completion_tokens': tot_compl_toks, 'total_tokens': tot_prompt_toks + tot_compl_toks,
                'elapsed_s': round(elapsed, 1),
                'tps': round(tot_compl_toks / elapsed, 1) if elapsed else 0}
              yield f"data: {json.dumps(usage)}\n\n"
              yield "data: [DONE]\n\n"
              break
        return StreamingResponse(generate(), media_type="text/event-stream")
      return await _guarded(_)

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