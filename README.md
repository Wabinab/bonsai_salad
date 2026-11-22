# bonsai_salad
TEST ONLY: hosting Ternary Bonsai v2 on SaladCloud. This is the compilation script to create the docker container.

```bash
docker build -f docker/Dockerfile.thin \
  --build-arg BASE=ghcr.io/wabinab/bonsai:xyz-0.1 \
  -t ghcr.io/wabinab/bonsai:xyz-0.5 .

docker build -f docker/Dockerfile.thin \
  --build-arg BASE=ghcr.io/wabinab/bonsai:plain-0.1 \
  --build-arg MMPROJ_FILE=Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  -t ghcr.io/wabinab/bonsai:plain-0.5 .

docker push ghcr.io/wabinab/bonsai:xyz-0.5
docker push ghcr.io/wabinab/bonsai:plain-0.5
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

