#!/usr/bin/env python3
# filename: chat-bot.py
"""
Simple agentic CLI chatbot that using LlamaClient
"""
import argparse
import sys

from dataclasses import dataclass, field
from LlamaClient import (Request, LlamaServer, Logger, LlamaServerError, LlamaUnreachable,
                         Seam, CodeFence, Terminal)

def estimate_tokens(text: str) -> int:
    """Estimate token count based on character count."""
    return len(text) // 4  # Approximate token estimation

@dataclass
class History:
    messages: list = field(default_factory=list)
    max_tokens: int = 16138

    def append(self, message):
        """Add a message and automatically truncate if needed."""
        if estimate_tokens(message['content']) > self.max_tokens:
            raise ValueError("Message exceeds max_tokens")
        self.messages.append(message)
        self._truncate_if_needed()

    def _truncate_if_needed(self):
        """Truncate history if total tokens exceed max_tokens."""
        # Find longest tail that fits within limit
        tail_tokens = 0
        for i, msg in enumerate(reversed(self.messages)):
            msg_tokens = estimate_tokens(msg['content'])
            if tail_tokens + msg_tokens > self.max_tokens:
                # Remove messages from the beginning
                self.messages = self.messages[len(self.messages) - i:]  # Keep last i messages
                break
            tail_tokens += msg_tokens

    def view(self):
        """Return a copy of the messages --- not the mutable list."""
        return self.messages.copy()

    def clear(self):
        """Clear all messages."""
        self.messages.clear()

    def get_total_tokens(self):
        """Return total tokens in history."""
        return sum(estimate_tokens(msg['content']) for msg in self.messages) if self.messages else 0

    def __len__(self):
        return len(self.messages)


def main():
    ap = argparse.ArgumentParser(description="Simple chatbot using LlamaClient")
    ap.add_argument("-p", "--port", type=int, default=8012)
    ap.add_argument("-t", "--temperature", type=float, default=0.7)
    ap.add_argument("-c", "--context", type=int, default=8192, help="context window size")
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

    models = [m["id"] for m in server.models()] 


    while True:
        try:
            user_input = input(c("user", "You: ")).strip()
            if not user_input:
                continue

            if user_input.lower() in ['/quit', '/exit']:
                print(c("banner", "Goodbye!"))
                break
            elif user_input.lower() == '/clear':
                history.clear()
                print(c("banner", "Conversation history cleared."))
                continue
            elif user_input.lower() == '/history':
                print(c("banner", "Conversation history:"))
                for msg in history.view():
                    if msg['role'] == 'user':
                        print(c("history-user", f"  {msg['content']}"))
                    elif msg['role'] == 'assistant':
                        print(c("history-assistant", f"  {msg['content']}"))
                continue
            elif user_input.lower().startswith('/models'):
                # get list of models from server
                print(c("banner", "Available models:"))
                for model in models:
                    print(c("banner", f"  {model}"))
                continue
            elif user_input.lower().startswith('/model'):
                # get list of models from server
                print(c("banner", f"Current model: {args.model}"))  
                continue
            elif user_input.lower().startswith('/model '):
                # change model
                model_id = user_input.split(' ', 1)[1]
                if model_id in models:
                    args.model = model_id
                    print(c("banner", f"Model changed to {model_id}"))
                else:
                    print(c("banner", f"Model {model_id} not found."))
                continue


            # Add user message to history
            history.append({"role": "user", "content": user_input})

            # Get response from the model (streaming)
            req = Request(
                messages=[{"role": "system", "content": sys_prompt}] + history.view(),
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                think=args.think,
                stream=True
            )

            # Stream response
            term = Terminal(out=sys.stdout, colour=_TTY)
            print(c("assistant", "Assistant: "), end="", flush=True)  # Print assistant prefix
            completion = server.stream(req, Seam(CodeFence(term)))
            history.append({"role": "assistant", "content": completion.content})
            print()  # Final newline

            # Display stats
            total_tokens = sys_prompt_tokens + history.get_total_tokens()
            print(c("stats", f"Context: {total_tokens} / {args.context} tokens ({total_tokens/args.context*100.0:.1f}%)"))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
        finally:
            print(c("stats", "-" * 50))

    if args.log and completion:
        # Log the full trace at exit, not turn by turn
        Logger(args.log).record(req, completion, args.port)


           
        





if __name__ == "__main__":
    main()

