#!/usr/bin/env bash
set -e

python run_auto.py -m +experiment=sweep_smoke "$@"
