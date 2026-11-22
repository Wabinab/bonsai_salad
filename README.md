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