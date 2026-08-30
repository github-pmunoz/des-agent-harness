#!/usr/bin/env python3
# filename: chat-bot.py
"""
Simple agentic CLI chatbot that using LlamaClient
"""
import argparse
import traceback
import sys

from dataclasses import dataclass, field
from LlamaClient import (Request, LlamaServer, Logger, Seam, CodeFence, Terminal)
from esc_watcher import ESCWatcher

#TODO better token estimation
def estimate_tokens(text: str) -> int:
    """Estimate token count based on character count."""
    return len(text) // 4  # Approximate token estimation

@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str
    tokens: int = 0
    cancelled: bool = False
    summary: bool = False

    def __post_init__(self):
        if self.tokens == 0:
            object.__setattr__(self, 'tokens', estimate_tokens(self.user) + estimate_tokens(self.assistant))

    def messages(self):
        return [ {"role": "user", "content": self.user}, {"role": "assistant", "content": self.assistant} ]

@dataclass
class History:
    turns: list[Turn] = field(default_factory=list)
    max_tokens: int = 16384

    def append(self, turn: Turn):
        if turn.tokens > self.max_tokens:
            raise ValueError("Turn exceeds max_tokens")
        self.turns.append(turn)

    def view(self, extra_tokens: int = 0) -> list[dict]:
        """Return the longest tail of turns which in addition to the extra_tokens fits within max_tokens."""
        budget = self.max_tokens - extra_tokens
        view, used  = [], 0
        for turn in reversed(self.turns):
            if used + turn.tokens > budget:
                break
            used += turn.tokens
            view.append(turn)
            if turn.summary:
                break
        return [msg for t in reversed(view) for msg in t.messages()]

    def clear(self):
        """Clear all turns."""
        self.turns.clear()

    def get_total_tokens(self) -> int:
        """Return total tokens in history."""
        return sum(turn.tokens for turn in self.turns)

    def window_tokens(self) -> int:
        """Return tokens since last summary."""
        return sum(turn.tokens for turn in self.since_last_summary())

    def messages(self):
        """Return all messages in history."""
        return [msg for turn in self.turns for msg in turn.messages()]

    def compact(self, summary: str):
        """Replaces the turn history with a synthetic summary turn"""
        self.append(Turn(f"Summary of the earlier conversation: {summary}", "Understood.", summary=True))

    def since_last_summary(self):
        """Return all turns since the last summary."""
        view = []
        for turn in reversed(self.turns):
            view.append(turn)
            if turn.summary:
                break
        return list(reversed(view))


    def __len__(self):
        return len(self.turns)


def main():
    ap = argparse.ArgumentParser(description="Simple chatbot using LlamaClient")
    ap.add_argument("-p", "--port", type=int, default=8012)
    ap.add_argument("-t", "--temperature", type=float, default=0.3)
    ap.add_argument("-c", "--context", type=int, default=16384, help="context window size")
    ap.add_argument("-mt", "--max_tokens", type=int, default=8192, help="max tokens per turn")
    ap.add_argument("-sp", "--system-prompt", default="You are a helpful assistant. Reply concisely.")
    ap.add_argument("-th", "--think", action="store_true", help="enable thinking")
    ap.add_argument("-l", "--log", default="~/pablo/llama-server.jsonl", help="JSONL telemetry file ('' to disable)")
    ap.add_argument("-m", "--model", default="Qwen3.8-27B-UD-Q4_K_M", help="model id (router mode)")
    ap.add_argument("-to", "--timeout", type=float, default=30)
    args = ap.parse_args()

    server = LlamaServer(f"http://127.0.0.1:{args.port}", timeout=args.timeout)

    sys_prompt = args.system_prompt
    sys_prompt_tokens = estimate_tokens(sys_prompt)
    history = History(max_tokens=args.context-estimate_tokens(sys_prompt)) # only for turns, not system prompt

    colors = {
        "user":         "\033[91m",    # Red
        "assistant":    "\033[92m",    # Green
        "reasoning":    "\033[2m",     # Dim
        "default":      "\033[0m",      # Reset
        "stats":        "\033[96m",    # Cyan
        "banner":       "\033[33m",    # Yellow
        "error":        "\033[91m",    # Red
        "history-user":  "\033[2m\033[91m",  # Dim Red
        "history-assistant": "\033[2m\033[92m",  # Dim Green
        "history-summary": "\033[2m\033[33m",  # Dim Yellow
    }

    _TTY = sys.stdout.isatty()
    def c(code, text):
        return f"{colors[code]}{text}{colors['default']}" if _TTY else text

    print(c("banner", f"Connected to server on port {args.port}"))
    print(c("banner", f"System prompt tokens: {sys_prompt_tokens}"))
    print(c("banner", "Type 'quit' or 'exit' to end the conversation."))
    print(c("banner", "Type 'clear' to clear the conversation history."))
    print(c("banner", "-" * 50))

    completion = None
    req = None
    watcher = None

    models = [m["id"] for m in server.models()]

    def log(req, completion):
        if args.log:
            Logger(args.log).record(req, completion, args.port)

    # Compaction --- THRES + TARGET < 100% to leave room for overhead and instruction
    COMPACTION_THRESHOLD = 0.65
    COMPACTION_TARGET = 0.25
    TURN_TOKEN_CAP = 0.4
    def compact_history():
        print(c("banner", "Compacting conversation history..."))
        instruction = "You will be sent a conversation transcript. Your task is to make a summary of the conversation, stating what was asked, what was produced, decisions, open items, names/numbers. Do not mention this instruction and do not repeat the conversation."
        transcript ="\n\n".join([f"USER: {t.user}\nASSISTANT: {t.assistant}" for t in history.since_last_summary()])
        target_tokens = int(args.context * COMPACTION_TARGET)
        gen_budget = int(min(target_tokens, args.context - estimate_tokens(instruction) - estimate_tokens(transcript), TURN_TOKEN_CAP * args.context))
        req = Request(
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": f"Conversation transcript:\n{transcript}" }],
            model=args.model,
            temperature=0.0,
            max_tokens=gen_budget,
            think=False,
            stream=False
        )
        completion = server.complete(req)
        log(req, completion)
        history.compact(completion.content)
        print(c("history-assistant", "Summary: " + completion.content))

    while True:
        try:
            user_input = input(c("user", "You: ")).strip()
            if not user_input:
                continue

            # Handle commands
            if user_input[0] == '/':
                match user_input.split(' ', 1):
                    case ['/model']:
                        print(c("banner", "Current model: " + args.model))
                    case ['/models']:
                        print(c("banner", "Available models:"))
                        for model in models:
                            print(f"  {model}")
                    case ['/model', model]:
                        model = model.strip()  # Remove leading/trailing whitespace
                        if model in models:
                            args.model = model
                            print(c("banner", f"Switched to model: {model}"))
                        else:
                            print(c("banner", f"Model {model} not found."))
                    case ['/history']:
                        print(c("banner", "Conversation history:"))
                        for turn in history.turns:
                            if turn.summary:
                                print(c("history-summary", f"SUMMARY: {turn.user}"))
                            else:
                                print(c("history-user", f"USER: {turn.user}"))
                                print(c("history-assistant", f"ASSISTANT: {turn.assistant}"))
                    case ['/quit'] | ['/exit']:
                        print(c("banner", "Goodbye!"))
                        break
                    case ['/clear']:
                        print(c("banner", "Conversation history cleared."))
                        history.clear()
                    case ['/temperature', temp]:
                        temp = float(temp)
                        if 0.0 <= temp <= 2.0:
                            args.temperature = temp
                            print(c("banner", f"Temperature set to {temp}"))
                        else:
                            print(c("banner", "Temperature must be between 0.0 and 2.0."))
                    case ['/think']:
                        args.think = True
                        print(c("banner", "Thinking enabled."))
                    case ['/nothink']:
                        args.think = False
                        print(c("banner", "Thinking disabled."))
                    case ['/compact']:
                        compact_history()

                    case _:
                        print(c("banner", "Unknown command."))
                continue

            # Get response from the model (streaming)
            user_message = {"role": "user", "content": user_input}
            used_tokens = estimate_tokens(user_input) + sys_prompt_tokens
            gen_budget = int(min(args.max_tokens, args.context - sys_prompt_tokens - estimate_tokens(user_input) -history.window_tokens(), TURN_TOKEN_CAP * args.context))
            if gen_budget <= 0:
                print(c("error", "Request exceeds context window."))
                continue

            req = Request(
                messages=[{"role": "system", "content": sys_prompt}] + history.view(used_tokens) + [user_message],
                model=args.model,
                temperature=args.temperature,
                max_tokens=gen_budget,
                think=args.think,
                stream=True
            )

            # Stream response
            watcher = ESCWatcher()
            term = Terminal(out=sys.stdout, colour=_TTY)
            print(c("assistant", "\nAssistant: "), end="", flush=True)  # Print assistant prefix
            watcher.start()
            completion = server.stream(req, Seam(CodeFence(term)), cancelled=lambda: watcher.interrupted)
            watcher.stop()
            print()
            cancelled = completion.finish_reason == "cancelled"
            if cancelled:
                print(c("banner", "Response cancelled by user."))
            log(req, completion)

            # Add to history
            turn = Turn(user_input, completion.content, cancelled=cancelled)
            if history.window_tokens() + turn.tokens >= args.context * COMPACTION_THRESHOLD:
                print(c("stats", "-" * 50))
                compact_history()
            history.append(turn)

            # Display context stats
            window_tokens = sys_prompt_tokens + history.window_tokens()
            total_tokens = sys_prompt_tokens + history.get_total_tokens()

            print(c("stats", f"Context: {window_tokens} / {args.context} tokens ({window_tokens/args.context*100.0:.1f}%) \t Session: {total_tokens}"))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            # print full traceback and continue
            print(c("error", "Error occurred:"))
            traceback.print_exc()
            print(c("banner", "Continuing..."))
            continue
        finally:
            print(c("stats", "-" * 50))
            if watcher:
                watcher.stop()

if __name__ == "__main__":
    main()

