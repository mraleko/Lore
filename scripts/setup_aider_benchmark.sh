#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/benchmarks"
AIDER_DIR="$BENCH_DIR/aider"

mkdir -p "$BENCH_DIR"

if [ ! -d "$AIDER_DIR/.git" ]; then
  git clone https://github.com/Aider-AI/aider.git "$AIDER_DIR"
fi

if [ ! -d "$AIDER_DIR/tmp.benchmarks/polyglot-benchmark/.git" ]; then
  mkdir -p "$AIDER_DIR/tmp.benchmarks"
  git clone https://github.com/Aider-AI/polyglot-benchmark "$AIDER_DIR/tmp.benchmarks/polyglot-benchmark"
fi

cat <<EOF
Benchmark repos are ready in:
  $AIDER_DIR

Next commands:
  cd "$AIDER_DIR"
  ./benchmark/docker_build.sh
  ./benchmark/docker.sh

Inside the Docker container:
  pip install -e .[dev]
  export OLLAMA_API_BASE=http://host.docker.internal:11434
  ./benchmark/benchmark.py qwen25-coder-7b-q8-local-baseline --model ollama_chat/qwen2.5-coder:7b-instruct-q8_0 --edit-format whole --threads 1 --num-tests 10 --exercises-dir polyglot-benchmark

After a small run works, remove --num-tests 10 for the full benchmark.
Use --model ollama/qwen2.5-coder:7b-instruct-q8_0 instead if you want to mirror the old leaderboard command exactly.
EOF
