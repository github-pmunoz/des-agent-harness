#!/usr/bin/env python3
"""
Token usage estimation. len//4 heuristic observed ~20% undercount at 
small contexts. Replaced by usage.prompt_tokens where available.
TODO: use the tokenizer directly. 
"""


def estimate_tokens(text: str) -> int:
    return len(text) // 4
