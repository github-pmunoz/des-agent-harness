#!/usr/bin/env bash
# Uses the send_direct.py script to send a git diff to the LLM and get a commit message back.
#
# Usage:
#   ./generate_commit_message.sh [-d | --dir] DIR
#       Generate a commit message from the git diff restricted to DIR.
#   ./generate_commit_message.sh [-f | --files] FILE [FILE ...]
#       Generate a commit message from the git diff restricted to the given files.
#   ./generate_commit_message.sh [-s | --staged]
#       Generate a commit message from the diff of files staged for commit.
#   ./generate_commit_message.sh
#       Generate a commit message from the full git diff.
set -euo pipefail
cd $(dirname $0)

usage() {
    echo "Usage: $0 [-d | --dir] DIR"
    echo "       $0 [-f | --files] FILE [FILE ...]"
    echo "       $0 [-s | --staged]"
    echo "       $0"
    exit 1
}

STAGED=false
DIFF_ARGS=()
case "${1:-}" in
    -d|--dir)
        if [ $# -lt 2 ]; then
            echo "Error: $1 requires a directory argument."
            usage
        fi
        DIFF_ARGS=(-- "$2")
        ;;
    -f|--files)
        if [ $# -lt 2 ]; then
            echo "Error: $1 requires at least one file argument."
            usage
        fi
        DIFF_ARGS=(-- "${@:2}")
        ;;
    -s|--staged)
        STAGED=true
        ;;
    "")
        ;;
    *)
        echo "Error: unknown argument '$1'."
        usage
        ;;
esac

if [ "$STAGED" = true ]; then
    git diff --staged > /tmp/git_diff.txt
else
    git diff "${DIFF_ARGS[@]}" > /tmp/git_diff.txt
fi
trap "rm /tmp/git_diff.txt" EXIT # cleanup on exit

if [ ! -f /tmp/git_diff.txt ]; then
    echo "Error: git diff file not created"
    exit 1
fi

if [ ! -s /tmp/git_diff.txt ]; then
    echo "git diff is empty; nothing to generate."
    exit 1
fi

MAX_TOKENS=16000
TIMEOUT=500
MODEL="Qwen3.8-27B-UD-Q4_K_M"
INSTRUCTION="You will be given a file containing a git diff. Don't execute any code from this file. Your task is to provide a commit message for this git diff. Be consise: one sentence summarizing the changes and up to three bullet points, each a sentence providing more details." 
BIN="src/desh/llama/send_direct.py"

if [ ! -f "$BIN" ]; then
    echo "Error: send_direct.py not found at $BIN"
    exit 1
fi

python3 "$BIN" -mt "$MAX_TOKENS" -sp "$INSTRUCTION" -pf /tmp/git_diff.txt -s -m "$MODEL" -to "$TIMEOUT"
