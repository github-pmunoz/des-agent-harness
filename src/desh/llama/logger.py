"""
One JSONL record per completion, same layout send_direct.sh writes ({timestamp, port, payload,
response}) so existing jq queries over the file keep working. Never sees frames.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime

from desh.llama.wire import Completion, Request


class Logger:
    """
    One JSONL record per completion, same layout send_direct.sh writes
    ({timestamp, port, payload, response}) so existing jq queries over the file keep working.
    Errors are not recorded (design choice from the bash version: one record = one completion).
    """
    def __init__(self, path: str):
        self.path = pathlib.Path(path).expanduser()

    def record(self, req: Request, completion: Completion, port: int) -> None:
        rec = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "port": port,
            "payload": req.payload(),
            "response": completion.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
