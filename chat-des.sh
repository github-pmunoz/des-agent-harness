#!/bin/bash

DESH_HOME=$(dirname $0)
SESSIONS_DIR="$DESH_HOME/.sessions"
COMPLETIONS_LOG="$DESH_HOME/.completions.log"
DES_LOG="$DESH_HOME/.des.log"

cd "$DESH_HOME"
mkdir -p "$SESSIONS_DIR"
chat-des -sf ${SESSIONS_DIR} -cl ${COMPLETIONS_LOG} -dl ${DES_LOG} "$@"
