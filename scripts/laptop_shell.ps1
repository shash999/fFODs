# Open a shell in the DeepStream container with the GPU and this project mounted at /work.
# Build the image first:  docker build -t ffods-ds docker
$root = Split-Path -Parent $PSScriptRoot
docker run --gpus all --rm -it `
  -v "${root}:/work" -w /work `
  ffods-ds bash
