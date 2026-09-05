#!/usr/bin/env bash
# Uses the send_direct.py script to send a git diff to the LLM and get a commit message back.
set -euo pipefail
cd $(dirname $0)

git diff > /tmp/git_diff.txt
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
