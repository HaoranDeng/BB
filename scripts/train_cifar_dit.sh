#!/usr/bin/env bash
set -euo pipefail

python tools/train_dit.py --config configs/cifar10_dit.yaml "$@"
