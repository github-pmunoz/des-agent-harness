#!/usr/bin/env python3
# filename: chat-bot.py
"""
Simple agentic CLI chatbot that using LlamaClient
"""
import argparse
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
        return [msg for t in reversed(view) for msg in t.messages()]

    def clear(self):
        """Clear all turns."""
        self.turns.clear()

    def get_total_tokens(self):
        """Return total tokens in history."""
        return sum(turn.tokens for turn in self.turns)

    def messages(self):
        """Return all messages in history."""
        return [msg for turn in self.turns for msg in turn.messages()] 

    def __len__(self):
        return len(self.turns)


def main():
    ap = argparse.ArgumentParser(description="Simple chatbot using LlamaClient")
    ap.add_argument("-p", "--port", type=int, default=8012)
    ap.add_argument("-t", "--temperature", type=float, default=0.7)
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
        "history-user":  "\033[2m\033[91m",  # Dim Red 
        "history-assistant": "\033[2m\033[92m",  # Dim Green 
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
                        for msg in history.messages():
                            if msg['role'] == 'user':
                                print(c("history-user", f"  {msg['content']}"))
                            elif msg['role'] == 'assistant':
                                print(c("history-assistant", f"  {msg['content']}"))
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
                    case _:
                        print(c("banner", "Unknown command."))
                continue

            # Get response from the model (streaming)
            user_message = {"role": "user", "content": user_input}
            used_tokens = estimate_tokens(user_input) + sys_prompt_tokens
            req = Request(
                messages=[{"role": "system", "content": sys_prompt}] + history.view(used_tokens) + [user_message],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
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
            print()  # Final newline

            # Add to history
            cancelled = completion.finish_reason == "cancelled"
            turn = Turn(user_input, completion.content, cancelled=cancelled)
            history.append(turn)  # Add to history

            if cancelled:
                print(c("banner", "Response cancelled by user.")) 
                continue

            # Display stats
            total_tokens = sys_prompt_tokens + history.get_total_tokens()
            print(c("stats", f"Context: {total_tokens} / {args.context} tokens ({total_tokens/args.context*100.0:.1f}%)"))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            raise e

        finally:
            print(c("stats", "-" * 50))
            if watcher:
                watcher.stop()

    if args.log and completion:
        # Log the full trace at exit, not turn by turn
        Logger(args.log).record(req, completion, args.port)


           
        





if __name__ == "__main__":
    main()

