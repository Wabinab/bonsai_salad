import asyncio, json
import app

def mk(k): return 'data: ' + json.dumps({'type': 'text', 'content': 'c%d' % k}) + '\n\n'

async def agen(n, gap=0.002, done=True):
  for k in range(n):
    yield mk(k); await asyncio.sleep(gap)
  if done: yield 'data: [DONE]\n\n'

async def drain(g, since, stop_after=None):
  got, rotated, done = [], False, False
  t = app._gen_tail(g, since)
  try:
    async for s in t:
      body = s[6:-2]
      if body == '[DONE]': done = True; break
      p = json.loads(body)
      if p.get('type') == 'rotate': rotated = True; break
      if p.get('type') in ('meta', 'ping'): continue
      got.append(p['seq'])
      if p.get('type') == 'fail': break
      if stop_after and len(got) >= stop_after: break
  finally: await t.aclose()
  return got, rotated, done

async def main():
  app.PING_EVERY = 0.02
  # T1: kill the connection mid-stream, reattach with the cursor: 1..200, no gaps, no duplicates
  g = app._Gen('t1'); g.task = asyncio.create_task(app._gen_worker(g, agen(200)))
  a, _, _ = await drain(g, 0, stop_after=50)
  b, _, done = await drain(g, a[-1])
  assert a + b == list(range(1, 201)) and done, (a[:3], b[-3:], done)
  print('PASS T1 resume: 1..200 in order, no gaps or duplicates')
  # T2: planned rotation: the tail ends cleanly and the cursor continues
  app.ROTATE_AFTER = 0.05
  g = app._Gen('t2'); g.task = asyncio.create_task(app._gen_worker(g, agen(100, gap=0.01)))
  allseq, since, rots = [], 0, 0
  while True:
    got, rot, done = await drain(g, since)
    allseq += got; since = allseq[-1] if allseq else since
    if done: break
    assert rot; rots += 1
  assert allseq == list(range(1, 101)) and rots >= 3, (len(allseq), rots)
  print('PASS T2 rotation: 1..100 across', rots, 'rotations')
  # T3: cancel (janitor or new request): a fail frame is buffered, the tail ends, ACTIVE returns to 0
  app.ROTATE_AFTER = 110.0
  g = app._Gen('t3'); g.task = asyncio.create_task(app._gen_worker(g, agen(1000, gap=0.01)))
  await asyncio.sleep(0.1); g.task.cancel()
  try: await g.task
  except asyncio.CancelledError: pass
  assert g.done and json.loads(g.frames[-1][6:-2])['type'] == 'fail' and app.ACTIVE == 0
  print('PASS T3 cancel: fail frame buffered, ACTIVE =', app.ACTIVE)
  # T4: generator ends without [DONE]: reported as a failure, not as success
  g = app._Gen('t4'); g.task = asyncio.create_task(app._gen_worker(g, agen(5, done=False)))
  await g.task
  assert json.loads(g.frames[-1][6:-2])['type'] == 'fail'
  print('PASS T4 no-DONE: fail frame')
  print('ALL PASS')

asyncio.run(main())