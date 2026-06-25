#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <report-name-without-extension>" >&2
  echo "Example: $0 \"Weekly Report 5 (06-22-2026 - 06-28-2026)\"" >&2
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
# latexmkrc sets $out_dir = out/<jobname> relative to reports/
latexmk -pdf "${name}.tex"
echo "Built: $root/reports/out/${name}/${name}.pdf"
