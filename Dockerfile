# syntax=docker/dockerfile:1
ARG CUDA_VERSION=13.5.0
ARG UBUNTU=26.04

# ---------- stage 1: compile (no GPU needed) ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU} AS build
ARG REPO
ARG REF
ARG JOBS=8
ARG CUDA_ARCHS="86;89;120"
ARG CMAKE_EXTRA="-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_WEBUI=OFF"   # No need Web UI. Cause errors.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ca-certificates libssl-dev libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*
RUN test -n "$REPO" && test -n "$REF" \
 && git clone "$REPO" /src && git -C /src checkout "$REF"
WORKDIR /src
# Remove any stale UI dist so the OFF flags are actually respected.
RUN rm -rf build tools/ui/dist
# Explicit archs (no GPU to autodetect). NATIVE=OFF so the CPU code runs on any node.
RUN cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" ${CMAKE_EXTRA} \
 && cmake --build build --config Release -j${JOBS} --target llama-server
RUN git clone --depth 1 https://github.com/PrismML-Eng/Bonsai-demo.git /demo && rm -rf /demo/.git

# ---------- stage 2: runtime ----------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU}
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv curl ca-certificates libgomp1 libcurl4t64 libssl3t64 \
 && rm -rf /var/lib/apt/lists/*

# Layers go from least to most volatile. Model first, app code last.
RUN mkdir /models && curl -fL --retry 5 --retry-delay 5 -o /models/model.gguf \
    https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf

ARG VARIANT=plain
RUN if [ "$VARIANT" = xyz ]; then curl -fL --retry 5 -o /models/drafter.gguf \
    https://github.com/xyznq1/xyz-drafter/releases/download/v1.2/xyz-v1.2-drafter.gguf; fi

RUN curl -fL https://github.com/Wabinab/web-searcher-for-modal/releases/download/v0.3.0/web-searcher -o /usr/local/bin/web-searcher \
 && curl -fL https://github.com/Wabinab/mlkem-generator/releases/download/v0.2.0/mlkem-generator -o /usr/local/bin/mlkem-generator \
 && chmod +x /usr/local/bin/web-searcher /usr/local/bin/mlkem-generator
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir \
    fastapi uvicorn openai httpx requests claw-compactor cryptography

ENV LD_LIBRARY_PATH=/opt/llama/bin PATH=/venv/bin:$PATH \
    BONSAI_VARIANT=${VARIANT} \
    BONSAI_FAMILY=bonsai2 BONSAI_MODEL=27B BONSAI_GGUF=/models/model.gguf \
    BONSAI_KV4=1 BONSAI_CTX=262144 BONSAI_NGL=999 BONSAI_HOST=127.0.0.1 \
    PORT=8081 UVICORN_HOST=::

COPY --from=build /src/build/bin/ /opt/llama/bin/
COPY --from=build /demo /opt/Bonsai-demo
RUN mkdir -p /opt/Bonsai-demo/bin && ln -s /opt/llama/bin /opt/Bonsai-demo/bin/cuda \
 && chmod +x /opt/Bonsai-demo/scripts/*.sh
# Fail the build on missing shared libs. libcuda is injected by the driver at runtime, so ignore it.
RUN if ldd /opt/llama/bin/llama-server | grep 'not found' | grep -v libcuda; then echo MISSING LIBS; exit 1; fi

WORKDIR /app
COPY args.xyz /app/args.xyz
COPY entrypoint.sh app.py /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]