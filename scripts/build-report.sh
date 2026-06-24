#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <report-name-without-extension>" >&2
  echo "Example: $0 week_5" >&2
  exit 1
fi

name="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
tex="$root/reports/${name}.tex"

if [[ ! -f "$tex" ]]; then
  echo "Report not found: $tex" >&2
  exit 1
fi

cd "$root/reports"
latexmk -pdf -outdir="out/${name}" "${name}.tex"
echo "Built: $root/reports/out/${name}/${name}.pdf"
