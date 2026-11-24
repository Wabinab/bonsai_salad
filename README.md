# bonsai_salad
TEST ONLY: hosting Ternary Bonsai v2 on SaladCloud. This is the compilation script to create the docker container.

```bash
docker buildx build -f docker/Dockerfile.thin \
  --build-arg BASE=ghcr.io/wabinab/bonsai:xyz-0.1 \
  -t ghcr.io/wabinab/bonsai:xyz-0.5 .

docker buildx build -f docker/Dockerfile.thin \
  --build-arg BASE=ghcr.io/wabinab/bonsai:plain-0.1 \
  --build-arg MMPROJ_FILE=Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  -t ghcr.io/wabinab/bonsai:plain-0.5 .

docker push ghcr.io/wabinab/bonsai:xyz-0.5
docker push ghcr.io/wabinab/bonsai:plain-0.5
```

## Build reference for PQ2 matrix

```bash
docker buildx build --progress=plain -f docker/Dockerfile.full2 \
  --build-arg VARIANT=xyz \
  --build-arg REPO=https://github.com/xyznq1/xyz-llama \
  --build-arg REF=f095f4dd559cdb742c9b3ed71b7131855179f935 \
  --build-arg CUDA_VERSION=13.5.0 --build-arg UBUNTU=26.04 \
  --build-arg CMAKE_EXTRA=-DLLAMA_BUILD_BORINGSSL=ON --build-arg JOBS=$JOBS \
  --load -t ghcr.io/wabinab/bonsai:xyz2-0.1 .
docker buildx build --progress=plain -f docker/Dockerfile.full2 \
  --build-arg VARIANT=plain \
  --build-arg REPO=https://github.com/PrismML-Eng/llama.cpp \
  --build-arg REF=d12ca13bb9f2136c8e03bfceccf20b06d115fc52  \
  --build-arg CUDA_VERSION=13.5.0 --build-arg UBUNTU=26.04 \
  --build-arg JOBS=$JOBS \
  --build-arg MMPROJ_FILE=Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --load -t ghcr.io/wabinab/bonsai:plain2-0.1 . 
```

For further build on thin: 

```bash
docker buildx build --progress=plain -f docker/Dockerfile.thin2 \
  --build-arg BASE=ghcr.io/wabinab/bonsai:xyz2-0.1 \
  --load -t ghcr.io/wabinab/bonsai:xyz2-0.2 .
docker buildx build --progress=plain -f docker/Dockerfile.thin2 \
  --build-arg BASE=ghcr.io/wabinab/bonsai:plain2-0.1 \
  --load -t ghcr.io/wabinab/bonsai:plain2-0.2 .
```

## Startup New Instance
When startup a new instance, run these first: 

```bash
sudo apt-get update && sudo apt-get install -y docker.io git tmux
sudo usermod -aG docker $USER && newgrp docker
df -h /var/lib/docker          # want 100+ GB free (if building from scratch, the .full)
tmux new -s build              # an SSH drop will not kill the build

git clone wabinab/bonsai_salad && cd bonsai_salad
JOBS=$(( $(nproc) / 2 ))   # (optional) For building .full only. 
```

Full build on 16 vCPU and 64 GB RAM is going to take about half an hour (at least) per image 
from scratch (when there's nothing to reuse). vRAM usage is, the one I remember, more than 32GB. vCPU usage sometimes goes to full on all cores, other times full only on half of the cores. Storage will in the end use up around 110GB+. 

## Buildx
Usually they come with it, but if they don't, here it is (from manual installation). 

"come with it" means I use a pre-installed docker image when creating my instance. 

```bash
# Install prerequisites and Docker's signing key
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Remove the earlier incorrect source file
sudo rm -f /etc/apt/sources.list.d/docker.sources

# Add the Ubuntu repository; do not put a backslash before the $ in UBUNTU_CODENAME
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "\${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# Refresh package lists and install Buildx
sudo apt update
sudo apt install -y docker-buildx-plugin

# Confirm it works
docker buildx version
```