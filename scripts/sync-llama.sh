#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$ROOT" submodule sync -- third_party/llama.cpp
git -C "$ROOT" submodule update --init --remote --recursive -- third_party/llama.cpp
