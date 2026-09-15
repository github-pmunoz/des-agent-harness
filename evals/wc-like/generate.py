#!/usr/bin/env python3
"""Generate train/ and test/ fixture sets for the wc-like CLI eval case.

Each set contains:
  - N text files with random content (lengths from tenths to hundreds of lines)
  - goldens.json with the correct words / lines / bytes for every file

The goldens are computed with the same rules a `wc`-like tool uses:
  - lines: number of newline characters in the file
  - words: whitespace-separated tokens (str.split())
  - bytes: size of the file on disk
"""
import json
import os
import random
import string

random.seed(20240517)  # deterministic so goldens are reproducible

WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
    "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
    "aliquip", "ex", "ea", "commodo", "consequat", "duis", "aute", "irure",
    "in", "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
    "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
    "non", "proident", "sunt", "culpa", "qui", "officia", "deserunt",
    "mollit", "anim", "id", "est", "laborum",
]


def random_line() -> str:
    n = random.randint(3, 14)
    return " ".join(random.choice(WORDS) for _ in range(n))


def make_file(path: str) -> None:
    # Lengths from tenths (1-9 lines) up to hundreds (100-250 lines).
    if random.random() < 0.35:
        n_lines = random.randint(1, 9)
    else:
        n_lines = random.randint(10, 250)
    lines = [random_line() for _ in range(n_lines)]
    # Most files end with a trailing newline; a few do not.
    trailing = random.random() < 0.8
    content = "\n".join(lines) + ("\n" if trailing else "")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def compute_goldens(directory: str) -> dict:
    goldens = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".txt"):
            continue
        full = os.path.join(directory, name)
        with open(full, "rb") as f:
            data = f.read()
        text = data.decode("utf-8")
        goldens[name] = {
            "lines": data.count(b"\n"),
            "words": len(text.split()),
            "bytes": len(data),
        }
    return goldens


def build_set(directory: str, count: int) -> None:
    os.makedirs(directory, exist_ok=True)
    for i in range(1, count + 1):
        make_file(os.path.join(directory, f"file_{i:02d}.txt"))
    goldens = compute_goldens(directory)
    with open(os.path.join(directory, "goldens.json"), "w", encoding="utf-8") as f:
        json.dump(goldens, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{directory}: {len(goldens)} files, goldens written")


if __name__ == "__main__":
    build_set("train", 30)
    build_set("test", 10)
