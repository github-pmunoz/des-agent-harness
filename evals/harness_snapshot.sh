#!/bin/bash
# harness_snapshot.sh — freeze the harness code a run executes.
#
# Usage:
#   ./harness_snapshot.sh <DEST>
#
# The venv installs the harness in editable mode, so `chat-des` runs whatever src/ holds at the
# moment each run starts: a sweep that outlives an edit runs its later arms on other code. A run
# launched from a snapshot instead runs on the copy made here, and its manifest names it.
#
# DEST receives
# - src/                the harness package as it was on disk: tracked and untracked files, not
#                       ignored ones — the working tree, uncommitted changes included
# - uncommitted.patch   `git diff HEAD -- src`: what the copy has over the commit, empty when clean
# - harness.json        commit, branch, dirty flag, untracked files, and the sha256 of the copied
#                       tree, so two snapshots made from the same code carry the same hash
#
# Launch from it with
#   PYTHONPATH="$DEST/src" <venv python> -m desh_chat.cli ...
# PYTHONPATH comes before the editable install's finder, so the copy is what gets imported.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <DEST>" >&2
  exit 2
fi
dest="$1"
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$dest"

# the working tree's src/, as git sees it: tracked plus untracked, minus ignored (__pycache__,
# *.egg-info)
(cd "$repo_root" && git ls-files -z -co --exclude-standard -- src) \
  | tar -C "$repo_root" --null -T - -cf - | tar -C "$dest" -xf -
git -C "$repo_root" diff HEAD -- src > "${dest}/uncommitted.patch"

content_sha="$(cd "$dest" && find src -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
untracked="$(git -C "$repo_root" ls-files -o --exclude-standard -- src | jq -R . | jq -s .)"
dirty=false
[[ -n "$(git -C "$repo_root" status --porcelain -- src)" ]] && dirty=true

jq -n \
  --arg commit "$(git -C "$repo_root" rev-parse HEAD)" \
  --arg branch "$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)" \
  --argjson dirty "$dirty" \
  --argjson untracked "$untracked" \
  --arg content_sha256 "$content_sha" \
  --arg path "$(cd "$dest" && pwd)" \
  --arg created "$(date +%Y-%m-%dT%H:%M:%S%z)" \
  '{commit: $commit, branch: $branch, dirty: $dirty, untracked: $untracked,
    content_sha256: $content_sha256, path: $path, created: $created}' > "${dest}/harness.json"
