#!/usr/bin/env bash
set -u
cd /app
if [ "${BONSAI_VARIANT:-plain}" = xyz ]; then
  ARGS=$(sed "s/-c [0-9][0-9]*/-c ${XYZ_CTX:-262144}/" /app/args.xyz)
  /opt/llama/bin/llama-server $ARGS &
else
  /opt/Bonsai-demo/scripts/start_llama_server.sh --alias bonsai-2-27b --parallel 1 &
fi
LLAMA=$!
python /app/app.py &
API=$!
trap 'kill $LLAMA $API 2>/dev/null' TERM INT
wait -n
code=$?
kill $LLAMA $API 2>/dev/null
exit $(( code == 0 ? 1 : code ))   # either child dying = unhealthy; Salad restarts the container