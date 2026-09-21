#!/bin/bash

DESH_HOME=$(dirname $0)
SESSIONS_DIR="$DESH_HOME/.sessions"
COMPLETIONS_LOG="$DESH_HOME/.completions.log"
DES_LOG="$DESH_HOME/.des.log"
MODEL="Qwen3.8-27B-UD-Q4_K_M-256K"
CONTEXT=262144
SYSTEM_PROMPT="You are a coding assistant. Reply concisely."


cd "$DESH_HOME"
mkdir -p "$SESSIONS_DIR"
chat-des -sf ${SESSIONS_DIR} -cl ${COMPLETIONS_LOG} -dl ${DES_LOG} -sp "${SYSTEM_PROMPT}" --model ${MODEL} --context ${CONTEXT} --max-turn-tokens ${CONTEXT} --max-tool-rounds 30 --checkpoint-target 0.15 --auto --cont --read --write --bash --edit --delegate --scratchpad "$@"
