"""
Streaming python scanner for the renderer: classifies the text of a python fence body into
coloured spans while the text is still arriving as arbitrary fragments.

Pure function over (ScanState, text) -> (spans, ScanState). A span is (channel, text) in the
renderer's vocabulary: "py_kw", "py_builtin", "py_call", "py_str", "py_num", "py_comment", and
"code_py" for everything that renders in the terminal's default colour. The stage that owns the
state (PyHighlight) forwards spans downstream; Terminal maps channels to colours. Nothing here
knows what a terminal is.

The streaming rule: a token is emitted only once it is DECISIVE — the next character that could
arrive cannot change its classification. Three tokens can run off the end of a fragment and are
held back as `pending` for the next call: an identifier/number run (the run may continue, and an
identifier's class depends on the character after it), a run of one or two quotes (a triple quote
may be forming), and a backslash inside a string (its escaped character has not arrived).
Comments need no hold: from "#" to newline everything is a comment whatever follows.

Not a tokenizer: no f-string internals, no decorators, no attribute chains. Strings, numbers,
comments and identifier classification are the whole scope — the model writes those, the eye
needs those.
"""
from __future__ import annotations

import keyword
from dataclasses import dataclass, replace
from string import ascii_letters, digits

Span = tuple[str, str]

DEFAULT = "code_py"
IDENT_CHARS = frozenset(ascii_letters + digits + "_")
IDENT_START = frozenset(ascii_letters + "_")
NUMBER_CHARS = IDENT_CHARS | {"."}
QUOTES = frozenset("'\"")
STRING_PREFIXES = frozenset({"r", "b", "f", "u", "rb", "br", "fr", "rf"})

KEYWORDS = frozenset(keyword.kwlist)
BUILTINS = frozenset({
    # types
    "int", "float", "str", "bytes", "bool", "list", "dict", "set", "frozenset", "tuple",
    "object", "type", "complex", "bytearray", "memoryview", "range", "slice",
    # functions the model reaches for
    "print", "len", "isinstance", "issubclass", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "sum", "min", "max", "abs", "round", "any", "all", "open", "iter", "next",
    "getattr", "setattr", "hasattr", "repr", "hash", "id", "input", "super", "vars", "dir",
    "callable", "format", "chr", "ord", "divmod", "pow",
    # exceptions
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "StopIteration", "NotImplementedError", "AttributeError", "OSError", "FileNotFoundError",
    "AssertionError",
    # names that are not keywords but read as constants
    "self", "cls", "__name__",
})


@dataclass(frozen=True)
class ScanState:
    string: str = ""      # the open string's closing delimiter ("'", '"', "'''", '"""'); "" outside
    comment: bool = False  # inside a "#" comment, waiting for its newline
    pending: str = ""     # held-back tail, not yet decisive


def classify(word: str, after: str) -> str:
    """
    Channel for an identifier run, given the first character after it ("" at end of stream).

    Returns one of "py_kw", "py_builtin", "py_call" or DEFAULT. Identity beats use: a keyword
    or builtin keeps its colour even when called, so "py_call" marks only names the code
    itself introduced — the ones the reader has to go and find.
    """
    if word in KEYWORDS:
        return "py_kw"
    if word in BUILTINS:
        return "py_builtin"
    if after == "(":
        return "py_call"
    return DEFAULT


def scan(state: ScanState, text: str, final: bool = False) -> tuple[list[Span], ScanState]:
    """
    Classify `state.pending + text` into spans, holding back whatever is not yet decisive.
    With final=True nothing is held: the end of the stream is the character after every token.
    Adjacent spans on the same channel are coalesced.
    """
    work = state.pending + text
    string, comment = state.string, state.comment
    spans: list[Span] = []
    n = len(work)
    i = 0

    def put(channel: str, s: str) -> None:
        if not s:
            return
        if spans and spans[-1][0] == channel:
            spans[-1] = (channel, spans[-1][1] + s)
        else:
            spans.append((channel, s))

    while i < n:
        if string:
            j, closed, hold = _scan_string(work, i, string, final)
            put("py_str", work[i:j])
            if closed:
                string = ""
            i = j
            if hold:
                break
            continue

        if comment:
            nl = work.find("\n", i)
            end = n if nl < 0 else nl
            put("py_comment", work[i:end])
            i = end
            if nl >= 0:
                comment = False
            continue

        c = work[i]

        if c == "#":
            comment = True
            continue

        if c in IDENT_START:
            j = _run(work, i, IDENT_CHARS)
            if j == n and not final:
                break  # the run may continue in the next fragment
            after = work[j] if j < n else ""
            word = work[i:j]
            if after in QUOTES and word.lower() in STRING_PREFIXES:
                delim = _opening_delimiter(work, j, final)
                if delim is None:
                    break  # quote run too short to tell single from triple
                put("py_str", word + delim)
                string = delim
                i = j + len(delim)
                continue
            put(classify(word, after), word)
            i = j
            continue

        if c in digits:
            j = _run(work, i, NUMBER_CHARS)
            if j == n and not final:
                break
            put("py_num", work[i:j])
            i = j
            continue

        if c in QUOTES:
            delim = _opening_delimiter(work, i, final)
            if delim is None:
                break
            put("py_str", delim)
            string = delim
            i += len(delim)
            continue

        put(DEFAULT, c)
        i += 1

    pending = "" if final else work[i:]
    if final:
        string, comment = "", False
    return spans, replace(state, string=string, comment=comment, pending=pending)


def finish(state: ScanState) -> tuple[list[Span], ScanState]:
    """End of the python body: classify whatever is pending and reset for the next fence."""
    return scan(state, "", final=True)


def _run(work: str, i: int, chars: frozenset[str]) -> int:
    j = i
    while j < len(work) and work[j] in chars:
        j += 1
    return j


def _opening_delimiter(work: str, i: int, final: bool) -> str | None:
    """The string delimiter starting at work[i]: three quotes if they are all here, one otherwise.
    None when the fragment ends inside a run of fewer than three quotes and more may follow."""
    q = work[i]
    j = _run(work, i, frozenset(q))
    if j - i >= 3:
        return q * 3
    if j == len(work) and not final:
        return None
    return q


def _scan_string(work: str, i: int, delim: str, final: bool) -> tuple[int, bool, bool]:
    """
    Advance through string text from work[i]. Returns (end, closed, hold): `end` is the index
    after the consumed text, `closed` whether the delimiter was found, `hold` whether the scan
    stopped on a tail that must wait for the next fragment (a trailing backslash, or a partial
    closing delimiter of a triple-quoted string).
    """
    n = len(work)
    j = i
    while j < n:
        c = work[j]
        if c == "\\":
            if j + 1 >= n:
                return (n if final else j), False, not final
            j += 2
            continue
        if work.startswith(delim, j):
            return j + len(delim), True, False
        if len(delim) == 3 and c == delim[0] and n - j < 3 and work[j:] == c * (n - j) and not final:
            return j, False, True  # "" at the end: the closing """ may be arriving
        if len(delim) == 1 and c == "\n":
            return j, True, False  # an unterminated single-quoted string ends at the line
        j += 1
    return n, False, False
