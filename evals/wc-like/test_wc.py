"""Pytest suite for wc.py, driven only by the training fixtures.

Uses train/file_*.txt and train/goldens.json. Does not reference test/.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "train")
GOLDENS_PATH = os.path.join(TRAIN_DIR, "goldens.json")
WC = os.path.join(ROOT, "wc.py")

with open(GOLDENS_PATH, "r", encoding="utf-8") as f:
    GOLDENS = json.load(f)


def run_wc(*args):
    """Run wc.py with the given arguments, returning (stdout, stderr, returncode)."""
    proc = subprocess.run(
        [sys.executable, WC, *args],
        capture_output=True,
        text=True,
    )
    return proc.stdout, proc.stderr, proc.returncode


def parse_line(line):
    """Parse one output line into (lines, words, bytes, filename)."""
    # Format: "<lines> <words> <bytes>  <filename>" (two spaces before name).
    nums, _, filename = line.rstrip("\n").partition("  ")
    l, w, b = nums.split(" ")
    return int(l), int(w), int(b), filename


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_counts_match_goldens(name):
    expected = GOLDENS[name]
    stdout, stderr, rc = run_wc(os.path.join(TRAIN_DIR, name))
    assert rc == 0, f"non-zero exit for {name}: {stderr}"
    assert stdout.endswith("\n")
    lines_out = stdout.strip().splitlines()
    assert len(lines_out) == 1
    l, w, b, fname = parse_line(lines_out[0])
    assert l == expected["lines"], f"{name}: lines {l} != {expected['lines']}"
    assert w == expected["words"], f"{name}: words {w} != {expected['words']}"
    assert b == expected["bytes"], f"{name}: bytes {b} != {expected['bytes']}"
    assert fname == name


def test_multiple_files():
    names = ["file_01.txt", "file_05.txt", "file_15.txt"]
    stdout, stderr, rc = run_wc(*[os.path.join(TRAIN_DIR, n) for n in names])
    assert rc == 0, stderr
    lines_out = stdout.strip().splitlines()
    assert len(lines_out) == len(names)
    for line, name in zip(lines_out, names):
        l, w, b, fname = parse_line(line)
        expected = GOLDENS[name]
        assert (l, w, b) == (expected["lines"], expected["words"], expected["bytes"])
        assert fname == name


def test_missing_file_errors():
    stdout, stderr, rc = run_wc(os.path.join(TRAIN_DIR, "does_not_exist.txt"))
    assert rc != 0
    assert stderr.strip() != ""
    assert stdout == ""


def test_missing_file_among_existing():
    good = os.path.join(TRAIN_DIR, "file_01.txt")
    bad = os.path.join(TRAIN_DIR, "nope.txt")
    stdout, stderr, rc = run_wc(good, bad)
    assert rc != 0
    assert stderr.strip() != ""


def test_no_args_usage():
    stdout, stderr, rc = run_wc()
    assert rc != 0
    assert stderr.strip() != ""
    assert stdout == ""


def test_output_format_two_spaces():
    stdout, stderr, rc = run_wc(os.path.join(TRAIN_DIR, "file_01.txt"))
    assert rc == 0
    line = stdout.rstrip("\n")
    # Exactly two spaces separate the byte count from the filename.
    assert "  " in line
    nums, _, fname = line.partition("  ")
    assert len(nums.split(" ")) == 3
    assert fname == "file_01.txt"
