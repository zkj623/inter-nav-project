#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'Usage: bash deploy/scripts/sync.sh user@robot /absolute/project/path [--apply]' >&2
    exit 2
fi
remote_host="$1"
remote_path="$2"
if [[ ! "$remote_host" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ || ! "$remote_path" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
    echo 'Use an SSH host and absolute path without spaces or shell characters.' >&2
    exit 2
fi
if [[ $# == 3 && "$3" != '--apply' ]]; then
    echo 'Only --apply is accepted as the third argument.' >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
options=(-az --itemize-changes --exclude=.git/ --exclude=.venv-deploy/ --exclude=__pycache__/)
if [[ $# == 2 ]]; then
    options+=(--dry-run)
    echo 'Preview only. Stop the robot task before applying an update.'
fi
manifest="$(mktemp)"
source_info="$(mktemp)"
trap 'rm -f "$manifest" "$source_info"' EXIT
# Include new nonignored deployment files as well as tracked code. No --delete.
git -C "$repo_dir" ls-files -co --exclude-standard -z > "$manifest"
rsync "${options[@]}" --from0 --files-from="$manifest" "$repo_dir/" "$remote_host:$remote_path/"

python3 - "$repo_dir" "$source_info" <<'META'
import json
from pathlib import Path
import subprocess
import sys
repo, output = sys.argv[1:]
commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
dirty = bool(subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'], text=True).strip())
Path(output).write_text(json.dumps({'source_commit': commit, 'dirty_worktree': dirty}, indent=2))
META
rsync "${options[@]}" "$source_info" "$remote_host:$remote_path/.deploy-source.json"
