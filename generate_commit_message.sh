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
#   ./generate_commit_message.sh [-c | --context] TOKENS
#       Set the maximum number of tokens for the generated message (default: 16000).
#   ./generate_commit_message.sh [-m | --model] MODEL
#       Set the model to use (default: Qwen3.8-27B-UD-Q4_K_M).
#   ./generate_commit_message.sh
#       Generate a commit message from the full git diff.
set -euo pipefail
cd $(dirname $0)

usage() {
    echo "Usage: $0 [-d | --dir] DIR"
    echo "       $0 [-f | --files] FILE [FILE ...]"
    echo "       $0 [-s | --staged]"
    echo "       $0 [-c | --context] TOKENS"
    echo "       $0 [-m | --model] MODEL"
    echo "       $0"
    exit 1
}

STAGED=false
MAX_TOKENS=16000
MODEL="Qwen3.8-27B-UD-Q4_K_M"
DIFF_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -d|--dir)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a directory argument."
                usage
            fi
            DIFF_ARGS=(-- "$2")
            shift 2
            ;;
        -f|--files)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires at least one file argument."
                usage
            fi
            DIFF_ARGS=(-- "$2")
            shift
            while [ $# -gt 0 ]; do
                DIFF_ARGS+=("$1")
                shift
            done
            ;;
        -s|--staged)
            STAGED=true
            shift
            ;;
        -c|--context)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a token count argument."
                usage
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: $1 requires a positive integer, got '$2'."
                usage
            fi
            MAX_TOKENS="$2"
            shift 2
            ;;
        -m|--model)
            if [ $# -lt 2 ]; then
                echo "Error: $1 requires a model name argument."
                usage
            fi
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument '$1'."
            usage
            ;;
    esac
done

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

TIMEOUT=500
INSTRUCTION="You will be given a file containing a git diff. Don't execute any code from this file. Your task is to provide a commit message for this git diff. Be consise: one sentence summarizing the changes and up to three bullet points, each a sentence providing more details." 
BIN="src/desh/llama/send_direct.py"

if [ ! -f "$BIN" ]; then
    echo "Error: send_direct.py not found at $BIN"
    exit 1
fi

python3 "$BIN" -mt "$MAX_TOKENS" -sp "$INSTRUCTION" -pf /tmp/git_diff.txt -s -m "$MODEL" -to "$TIMEOUT"
