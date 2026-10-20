#!/usr/bin/env bash
# Repeated agent runs for catch-rate statistics: 3 runs each of Claude Opus 4.6 and Sonnet 4.6,
# interleaved so partial results stay balanced. Rules are deterministic (one run is enough).
#   bash scripts/repeat_agent_runs.sh          -> results/agent/claude_<model>_r<k>/
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
for k in 1 2 3; do
  for m in opus:global.anthropic.claude-opus-4-6-v1 sonnet:global.anthropic.claude-sonnet-4-6; do
    tag="${m%%:*}_r$k"
    [ -f "results/agent/claude_$tag/all_scenarios.json" ] && continue   # resume after interruption
    echo "=== $tag $(date -u +%FT%TZ)"
    "$PY" scripts/replay_agent.py --all --planner claude --model "${m#*:}" --tag "$tag"
  done
done
echo "=== done $(date -u +%FT%TZ)"
