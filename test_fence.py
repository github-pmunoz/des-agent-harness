#!/usr/bin/env python3
"""CodeFence stage: roundtrip + channel checks on hand-split deltas. Run after each change."""
import LlamaClient as L

class Cap(L.Stage):
    def __init__(self): super().__init__(None); self.got = []
    def feed(self, ch, t): self.got.append((ch, t))

def run(chunks):
    cap = Cap(); st = L.CodeFence(cap)
    for c in chunks: st.feed("content", c)
    st.flush()
    merged = []                                   # merge adjacent same-channel spans for readability
    for ch, t in cap.got:
        if merged and merged[-1][0] == ch: merged[-1] = (ch, merged[-1][1] + t)
        else: merged.append((ch, t))
    return merged, "".join(t for _, t in cap.got)

CASES = {   # name: (deltas, expected channel sequence after merge)
    "whole":            (["hello ```py\nx=1\n```\ndone"],            ["content", "code", "content"]),
    "split fence":      (["hello ``", "`py\nx=1\n``", "`\ndone"],    ["content", "code", "content"]),
    "char by char":     (list("a ```b``` c"),                        ["content", "code", "content"]),
    "lone backticks":   (["use `x` and ``y`` ok"],                   ["content"]),
    "ends in fence":    (["text ```"],                               ["content", "code"]),
    "backticks at end": (["text ``"],                                ["content"]),
    "two blocks":       (["```a```b```c```"],                        ["code", "content", "code"]),
}
fails = 0
for name, (chunks, want) in CASES.items():
    spans, joined = run(chunks)
    rt = joined == "".join(chunks)
    chans = [ch for ch, _ in spans]
    ok = rt and chans == want
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name:18} roundtrip={'ok' if rt else 'LOST TEXT'}  {spans}")
raise SystemExit(fails)
