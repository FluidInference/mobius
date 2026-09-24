#!/bin/sh
# Bundle splits + scripts for the GPU box: build/cloud-bundle.tar.gz (unpack anywhere, run cloud/run.sh).
set -eu
cd "$(dirname "$0")"
[ -f splits/manifest.json ] || { echo "run make_splits.py first"; exit 1; }
mkdir -p build
tar czf build/cloud-bundle.tar.gz README.md splits cloud
ls -la build/cloud-bundle.tar.gz
echo "copy:   scp build/cloud-bundle.tar.gz <gpu-host>:~ && ssh <gpu-host> 'mkdir -p exp && tar xzf cloud-bundle.tar.gz -C exp'"
echo "run:    ssh <gpu-host> 'cd exp && BASE=Qwen/Qwen3.5-0.8B sh cloud/run.sh'"
