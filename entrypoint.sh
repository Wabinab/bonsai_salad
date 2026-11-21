#!/usr/bin/env bash
set -u
if [ "${BONSAI_VARIANT:-plain}" = xyz ]; then
  /opt/llama/bin/llama-server $(cat /app/args.xyz) &
else
  /opt/Bonsai-demo/scripts/start_llama_server.sh --alias bonsai-2-27b --parallel 1 &
fi
LLAMA=$!
uvicorn app:api --host "${UVICORN_HOST:-::}" --port 8000 --workers 1 &
API=$!
trap 'kill $LLAMA $API 2>/dev/null' TERM INT
wait -n
code=$?
kill $LLAMA $API 2>/dev/null
exit $(( code == 0 ? 1 : code ))   # either child dying means unhealthy: let Salad restart us