"""
Streaming python scanner: the pure function pyscan.scan over (state, text). Two properties are
pinned for every construct: the spans of a whole text fed at once, and the SAME spans when the
text arrives split at every possible character boundary — the hold-back rules exist so that
fragmentation never changes the output.
"""
import pytest

from desh.llama.pyscan import DEFAULT, ScanState, classify, finish, scan


def scan_all(text: str, final: bool = True) -> list[tuple[str, str]]:
    spans, state = scan(ScanState(), text)
    if final:
        tail, _ = finish(state)
        spans = coalesce(spans + tail)
    return spans


def scan_split(text: str, at: int) -> list[tuple[str, str]]:
    spans, state = scan(ScanState(), text[:at])
    more, state = scan(state, text[at:])
    tail, _ = finish(state)
    return coalesce(spans + more + tail)


def coalesce(spans):
    out = []
    for channel, text in spans:
        if out and out[-1][0] == channel:
            out[-1] = (channel, out[-1][1] + text)
        else:
            out.append((channel, text))
    return out


def every_split(text: str):
    """Every way of cutting `text` into two fragments must yield the unfragmented spans."""
    whole = scan_all(text)
    for at in range(len(text) + 1):
        assert scan_split(text, at) == whole, f"split at {at}: {text[:at]!r} | {text[at:]!r}"
    return whole


# ---------------------
# Strings
# ---------------------

class TestStrings:
    def test_string_span_and_default_around_it(self):
        spans = scan_all('("a", \'b\')')
        assert ("py_str", '"a"') in spans and ("py_str", "'b'") in spans

    def test_escaped_quote_does_not_close(self):
        assert scan_all(r'"a\"b"') == [("py_str", r'"a\"b"')]

    def test_triple_quoted_spans_newlines(self):
        assert scan_all('"""a\nb"""') == [("py_str", '"""a\nb"""')]

    def test_quotes_inside_triple_do_not_close(self):
        assert scan_all('"""a "b" c"""') == [("py_str", '"""a "b" c"""')]

    def test_empty_string_then_text(self):
        assert scan_all('""+1') == [("py_str", '""'), (DEFAULT, "+"), ("py_num", "1")]

    def test_prefixed_string_is_one_span(self):
        assert scan_all('f"{x}"') == [("py_str", 'f"{x}"')]
        assert scan_all("rb'x'") == [("py_str", "rb'x'")]

    def test_hash_inside_string_is_not_a_comment(self):
        assert scan_all('"#x"') == [("py_str", '"#x"')]

    def test_unterminated_single_quoted_string_ends_at_newline(self):
        assert scan_all('"abc\n1') == [("py_str", '"abc'), (DEFAULT, "\n"), ("py_num", "1")]

    def test_unterminated_string_at_end_flushes_as_string(self):
        assert scan_all('"ab') == [("py_str", '"ab')]


# ---------------------
# Identifier classification: identity beats use
# ---------------------

class TestClassify:
    @pytest.mark.parametrize("word", ["if", "def", "return", "None", "True", "lambda", "async"])
    def test_keywords(self, word):
        assert classify(word, " ") == "py_kw"

    def test_keyword_followed_by_paren_is_still_a_keyword(self):
        assert classify("if", "(") == "py_kw"

    @pytest.mark.parametrize("word", ["print", "len", "str", "self", "ValueError"])
    def test_builtins_whether_called_or_not(self, word):
        assert classify(word, "(") == "py_builtin"
        assert classify(word, " ") == "py_builtin"

    def test_own_name_called_is_a_call(self):
        assert classify("compute", "(") == "py_call"

    def test_own_name_not_called_is_default(self):
        for after in [" ", ".", "", "\n", ")"]:
            assert classify("compute", after) == DEFAULT, repr(after)

    @pytest.mark.parametrize("word", ["match", "case", "_"])
    def test_soft_keywords_are_plain_names(self, word):
        assert classify(word, " ") == DEFAULT

    def test_type_is_a_builtin_not_a_keyword(self):
        assert classify("type", "(") == "py_builtin"

    def test_call_detection_through_the_scanner(self):
        assert scan_all("def f(x):\n    return g(x)") == [
            ("py_kw", "def"), (DEFAULT, " "), ("py_call", "f"), (DEFAULT, "(x):\n    "),
            ("py_kw", "return"), (DEFAULT, " "), ("py_call", "g"), (DEFAULT, "(x)"),
        ]


# ---------------------
# Numbers and comments
# ---------------------

class TestNumbersAndComments:
    @pytest.mark.parametrize("n", ["0", "42", "3.14", "1e5", "0x1F", "1_000"])
    def test_number_forms(self, n):
        assert scan_all(n + " ") == [("py_num", n), (DEFAULT, " ")]

    def test_comment_to_end_of_line(self):
        assert scan_all("# c\n1") == [("py_comment", "# c"), (DEFAULT, "\n"), ("py_num", "1")]

    def test_quote_inside_comment_is_not_a_string(self):
        assert scan_all('# "x\n') == [("py_comment", '# "x'), (DEFAULT, "\n")]

    def test_comment_without_newline_is_emitted_immediately(self):
        spans, state = scan(ScanState(), "# abc")
        assert spans == [("py_comment", "# abc")]
        assert state.comment and state.pending == ""


# ---------------------
# Hold-back rules: what stays pending at a fragment boundary
# ---------------------

class TestHoldBack:
    def test_identifier_run_is_held(self):
        spans, state = scan(ScanState(), "x = fo")
        assert "".join(t for _, t in spans) == "x = "
        assert state.pending == "fo"

    def test_number_run_is_held(self):
        spans, state = scan(ScanState(), "a + 12")
        assert state.pending == "12"

    def test_short_quote_run_is_held(self):
        for q in ['"', '""', "'", "''"]:
            spans, state = scan(ScanState(), "x = " + q)
            assert state.pending == q, q

    def test_three_quotes_are_decisive(self):
        spans, state = scan(ScanState(), '"""')
        assert spans == [("py_str", '"""')] and state.string == '"""' and state.pending == ""

    def test_backslash_inside_string_is_held(self):
        spans, state = scan(ScanState(), '"ab\\')
        assert spans == [("py_str", '"ab')] and state.pending == "\\" and state.string == '"'

    def test_partial_closing_triple_quote_is_held(self):
        spans, state = scan(ScanState(), '"""ab""')
        assert spans == [("py_str", '"""ab')] and state.pending == '""'

    def test_operators_and_whitespace_are_never_held(self):
        spans, state = scan(ScanState(), "a = (")
        assert state.pending == "" and "".join(t for _, t in spans) == "a = ("

    def test_finish_resets_state(self):
        _, state = scan(ScanState(), '"""open')
        _, state = finish(state)
        assert state == ScanState()


# ---------------------
# Fragment invariance over a realistic snippet
# ---------------------

SNIPPET = '''def f(x, y=1.5):
    """Doc "string" here."""
    s = f"{x}\\n"  # trailing
    return len(s) + y
'''


class TestFragmentInvariance:
    def test_snippet_is_invariant_under_every_two_way_split(self):
        every_split(SNIPPET)

    def test_snippet_reassembles_verbatim(self):
        assert "".join(t for _, t in scan_all(SNIPPET)) == SNIPPET
