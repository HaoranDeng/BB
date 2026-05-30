#!/usr/bin/env bash
set -euo pipefail

python tools/train_vit.py --config configs/cifar10_vit.yaml "$@"
