#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 SUMMARY_JSON_OR_RUN_DIR [...]" >&2
  exit 1
fi

echo '| profile | total ms | label_generation ms | expected_response_prep ms | graph_construction ms | fast_phase ms | verifier_cpu ms | q/gamma | success_rate |'
echo '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'

for input_path in "$@"; do
  if [ -d "$input_path" ]; then
    summary_path="$input_path/summary.json"
  else
    summary_path="$input_path"
  fi

  if [ ! -f "$summary_path" ]; then
    echo "missing summary: $summary_path" >&2
    exit 1
  fi

  profile_name="$(basename "$(dirname "$(dirname "$summary_path")")")"
  jq -r --arg profile "$profile_name" '
    [
      $profile,
      (.timings_ms.total.mean | tostring),
      (.timings_ms.label_generation.mean | tostring),
      (.timings_ms.expected_response_prep.mean | tostring),
      (.timings_ms.graph_construction.mean | tostring),
      (.timings_ms.fast_phase_total.mean | tostring),
      (.verifier_cpu_time_ms.mean | tostring),
      (.q_over_gamma.mean | tostring),
      (.success_rate | tostring)
    ]
    | "| " + join(" | ") + " |"
  ' "$summary_path"
done
