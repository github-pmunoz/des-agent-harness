#!/usr/bin/env python3
"""
send_direct.py — CLI over LlamaClient; python port of an earlier bash prototype (send_direct.sh)

Same flags where they exist there (-p -t -mt -sp -up -pf -th -o -fo -to -n -s -l -m).
Rendering: with --stream the chain Seam -> CodeFence -> Terminal draws reasoning dim,
code yellow, and a --- seam; without it the final content is printed once.
"""
import argparse
import json
import sys
from urllib.parse import urlparse

from desh.llama.client import (Request, LlamaServer, Logger, LlamaServerError, LlamaUnreachable,
                         Seam, CodeFence, Terminal)


def prompt_from_file(path: str) -> str:
    """Numbered <file:> block, as the bash version builds with jq."""
    with open(path) as f:
        lines = f.read().rstrip("\n").split("\n")
    body = "\n".join(f"{i+1}\t{l}" for i, l in enumerate(lines))
    return f'<file: "{path.split("/")[-1]}">:\n{body}\n</file>'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Send one request directly to a llama-server.")
    ap.add_argument("-p", "--port", type=int, default=8012)
    ap.add_argument("-t", "--temperature", type=float, default=0.7)
    ap.add_argument("-mt", "--max-tokens", type=int, default=256)
    ap.add_argument("-sp", "--system-prompt", default="")
    ap.add_argument("-up", "--user-prompt", default="Reply with a single word: OK")
    ap.add_argument("-pf", "--prompt-file", help="file whose numbered contents replace the user prompt")
    ap.add_argument("-th", "--think", action="store_true", help="enable thinking")
    ap.add_argument("-o", "--output", help="write content to this file instead of stdout")
    ap.add_argument("-fo", "--full-output", action="store_true", help="print the full chat.completion JSON")
    ap.add_argument("-to", "--timeout", type=float, default=30)
    ap.add_argument("-n", "--no-op", action="store_true", help="print the payload, send nothing")
    ap.add_argument("-s", "--stream", action="store_true")
    ap.add_argument("-l", "--log", default="", help="JSONL telemetry file")
    ap.add_argument("-m", "--model", help="model id (router mode)")
    a = ap.parse_args(argv)

    req = Request.single(
        user=prompt_from_file(a.prompt_file) if a.prompt_file else a.user_prompt,
        system=a.system_prompt, model=a.model, temperature=a.temperature,
        max_tokens=a.max_tokens, think=a.think, stream=a.stream,
    )
    if a.no_op:
        print(json.dumps(req.payload(), indent=2))
        return 0

    server = LlamaServer(f"http://127.0.0.1:{a.port}", timeout=a.timeout)
    try:
        if a.stream:
            sink = Terminal(out=sys.stdout, colour=not a.output)
            completion = server.stream(req, Seam(CodeFence(sink)))
        else:
            completion = server.complete(req)
    except LlamaUnreachable as e:
        print(f"Error: could not reach llama-server: {e}", file=sys.stderr)
        return 1
    except LlamaServerError as e:
        print(f"Error from llama-server: {e}", file=sys.stderr)
        return 1

    if a.log:
        Logger(a.log).record(req, completion, a.port)

    if a.full_output:
        text = json.dumps(completion.to_dict(), indent=2, ensure_ascii=False)
    elif a.stream:
        return 0                                   # already rendered live
    else:
        text = completion.content
    if a.output:
        with open(a.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
