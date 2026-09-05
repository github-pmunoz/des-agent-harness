#!/usr/bin/env python3
"""
Token accounting.

Two regimes:
  - estimate_tokens(text)  pre-completion: nothing has been tokenized yet (gen_budget clamps,
                           system-prompt share of the window). Character heuristic, ~4 chars/token.
  - turn_tokens(...)       post-completion: the server's trailing `usage` frame is the truth for
                           the prompt it just tokenized and the tokens it just generated. Used to
                           price the Turn that goes into history.

TODO: use the tokenizer directly for the pre-completion regime.
"""
from typing import Optional


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def turn_tokens(usage: Optional[dict], user: str, assistant: str, reasoning: str, prior_tokens: int) -> int:
    """
    Price one Turn (user message + assistant reply) for history budgeting.

    usage         the completion's usage dict ({prompt_tokens, completion_tokens, total_tokens, ...})
                  or None when no usage frame arrived (cancelled stream, fake server, old server).
    user          the user message text.
    assistant     the assistant `content` — the part that gets re-sent in later prompts.
    reasoning     the assistant `reasoning_content` — generated (counted in completion_tokens)
                  but never re-sent.
    prior_tokens  what the caller already accounts for in the prompt that produced this usage:
                  estimate_tokens(system_prompt) + tokens of every history Turn in the view.

    Returns the token count to store on the Turn. Must be > 0 for any non-empty turn, otherwise
    Turn.__post_init__ treats 0 as "unpriced" and re-derives the heuristic.
    """
    if usage is None:
        return 0
    user_message_tokens = usage.get("prompt_tokens", estimate_tokens(user)) - prior_tokens
    if user_message_tokens <= 0:
        user_message_tokens = estimate_tokens(user)
    assistant_message_tokens = round(usage.get("completion_tokens", 0) * (len(assistant)/(len(assistant)+len(reasoning)) if reasoning else 1))
    if assistant_message_tokens <= 0:
        assistant_message_tokens = estimate_tokens(assistant)
    return user_message_tokens + assistant_message_tokens
