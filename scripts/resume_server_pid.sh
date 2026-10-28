#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root_dir=${PID_GRPO_ROOT:-/mnt/afs/L202500740/GRPO}
python_bin="$root_dir/venvs/pid-grpo-cu124/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing training Python: $python_bin" >&2
  exit 1
fi

exec "$python_bin" "$repo_dir/scripts/resume_server_pid.py" --root-dir "$root_dir" "$@"
