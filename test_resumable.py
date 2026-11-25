"""Offline tests for the v3 resumable generation in app.py (no server needed).

Run from the folder containing app.py:   python3 test_resumable.py

T1: kill the connection mid-stream, reattach with the cursor -- 1..200 in
    order, no gaps, no duplicates.
T2: planned rotation -- the tail ends cleanly with a rotate frame and the
    cursor continues, full 1..100 across multiple rotations.
T3: cancel (janitor or new request) -- a fail frame is buffered, the tail
    ends, ACTIVE returns to 0.
T4: upstream ends without [DONE] -- reported as a fail frame, not success.
T5: replay from since=0 re-delivers frame 1 including the key-exchange
    fields (ct/em), as the client relies on after a mid-drain cut.
T6: three concurrent tailers on one gen with independent cursors, each
    gets the full stream.
"""
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
  # T5 (extra): replay from since=0 re-delivers frame 1 including key-exchange fields
  g = app._Gen('t5')
  first = 'data: ' + json.dumps({'type': 'text', 'content': 'x', 'ct': 'CTBLOB', 'em': 'EMBLOB'}) + '\n\n'
  async def agen5():
    yield first
    yield mk(1)
    yield 'data: [DONE]\n\n'
  g.task = asyncio.create_task(app._gen_worker(g, agen5()))
  allseq, seen_first = [], None
  for since in (0, 0):   # two independent tails, both from 0, as after a mid-drain cut
    got, _, done = await drain(g, since)
    allseq += got
    p0 = json.loads(g.frames[0][6:-2])
    seen_first = (p0.get('ct'), p0.get('em'), p0.get('seq'))
  assert allseq == [1, 2, 1, 2] and seen_first == ('CTBLOB', 'EMBLOB', 1), (allseq, seen_first)
  print('PASS T5 key replay: ct/em in buffered frame 1, re-delivered on since=0')
  # T6 (extra): three concurrent tailers on one gen, different cursors, all complete
  app.ROTATE_AFTER = 110.0
  g = app._Gen('t6'); g.task = asyncio.create_task(app._gen_worker(g, agen(60, gap=0.005)))
  r = await asyncio.gather(drain(g, 0), drain(g, 0), drain(g, 0))
  assert all(seq == list(range(1, 61)) and d for seq, _, d in r)
  print('PASS T6 concurrent tailers: 3 readers, full replay each')
  print('ALL PASS')

asyncio.run(main())
