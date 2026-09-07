"""
Renderer stage tests: CodeFence (fence machine, language capture) and PyHighlight (python
token colouring inside a python fence). Every test feeds fragmented deltas through a stage into
a Recorder sink and asserts the ORDERED (channel, text) events — order is the property, since a
held-back tail emitted late is exactly the bug class this chain exists to avoid.
"""
import pytest

import io

from desh.llama.stages import CodeFence, PyHighlight, Stage, Terminal


class Recorder(Stage):
    """Renderer sink that records (channel, text) events and counts flushes."""
    def __init__(self):
        super().__init__(None)
        self.events: list[tuple[str, str]] = []
        self.flushed = 0

    def feed(self, channel: str, text: str) -> None:
        self.events.append((channel, text))

    def flush(self) -> None:
        self.flushed += 1


def joined(events: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Coalesce adjacent same-channel events so tests assert on spans, not delta boundaries."""
    out: list[tuple[str, str]] = []
    for channel, text in events:
        if out and out[-1][0] == channel:
            out[-1] = (channel, out[-1][1] + text)
        else:
            out.append((channel, text))
    return out


def run_fence(*deltas: str, channel: str = "content") -> Recorder:
    sink = Recorder()
    stage = CodeFence(sink)
    for d in deltas:
        stage.feed(channel, d)
    stage.flush()
    return sink


# ---------------------
# CodeFence — characterization of the fence machine as shipped
# ---------------------

class TestCodeFenceBaseline:
    def test_plain_content_passes_through(self):
        sink = run_fence("hello ", "world")
        assert sink.events == [("content", "hello "), ("content", "world")]
        assert sink.flushed == 1

    def test_complete_fence_in_one_delta_rechannels_the_body(self):
        sink = run_fence("say ```\nx\n``` done")
        assert joined(sink.events) == [("content", "say "), ("code", "```\nx\n```"), ("content", " done")]

    def test_fence_split_across_deltas_is_held_back(self):
        # "``" alone must not be emitted as content: it may be the start of a fence.
        sink = run_fence("a ``", "`\nx\n`", "``")
        assert joined(sink.events) == [("content", "a "), ("code", "```\nx\n```")]

    def test_single_backtick_inline_code_stays_content(self):
        sink = run_fence("use `x` here")
        assert joined(sink.events) == [("content", "use `x` here")]

    def test_unterminated_fence_flushes_as_code(self):
        sink = run_fence("```\nx", "``")
        assert joined(sink.events) == [("code", "```\nx``")]

    def test_non_content_channel_flushes_held_tail_first(self):
        # A reasoning event must not overtake content that was held back as a possible fence start.
        sink = Recorder()
        stage = CodeFence(sink)
        stage.feed("content", "a `")
        stage.feed("reasoning", "r")
        stage.flush()
        assert sink.events == [("content", "a "), ("content", "`"), ("reasoning", "r")]

    def test_two_fences_toggle_state(self):
        sink = run_fence("```\na\n```\nmid\n```\nb\n```")
        assert joined(sink.events) == [
            ("code", "```\na\n```"), ("content", "\nmid\n"), ("code", "```\nb\n```"),
        ]


# ---------------------
# CodeFence — language capture: a python fence body goes out as "code_py"
# ---------------------

class TestCodeFenceLanguage:
    def test_python_body_is_code_py_and_fence_lines_stay_code(self):
        sink = run_fence("```python\nx = 1\n```")
        assert joined(sink.events) == [("code", "```python\n"), ("code_py", "x = 1\n"), ("code", "```")]

    @pytest.mark.parametrize("info", ["py", "python3", "Python", "python title=x"])
    def test_python_aliases_and_extra_info_words(self, info):
        sink = run_fence(f"```{info}\nx\n```")
        assert ("code_py", "x\n") in joined(sink.events)

    def test_other_language_body_stays_code(self):
        sink = run_fence("```bash\nls\n```")
        assert joined(sink.events) == [("code", "```bash\nls\n```")]

    def test_bare_fence_body_stays_code(self):
        sink = run_fence("```\nls\n```")
        assert joined(sink.events) == [("code", "```\nls\n```")]

    def test_info_string_split_across_deltas(self):
        sink = run_fence("``", "`py", "thon", "\nx\n", "```")
        assert joined(sink.events) == [("code", "```python\n"), ("code_py", "x\n"), ("code", "```")]

    def test_language_resets_between_fences(self):
        sink = run_fence("```python\na\n```\n```\nb\n```")
        assert joined(sink.events) == [
            ("code", "```python\n"), ("code_py", "a\n"), ("code", "```"),
            ("content", "\n"), ("code", "```\nb\n```"),
        ]

    def test_held_info_fragment_flushes_as_code(self):
        # Stream ends inside the info line: it is a fence line, never a body.
        sink = run_fence("```pyth")
        assert sink.events == [("code", "```"), ("code", "pyth")]

    def test_non_content_event_flushes_held_info_first(self):
        sink = Recorder()
        stage = CodeFence(sink)
        stage.feed("content", "```py")
        stage.feed("tool_name", "Read")
        assert sink.events == [("code", "```"), ("code", "py"), ("tool_name", "Read")]


# ---------------------
# PyHighlight — the scanner as a stage, composed under CodeFence
# ---------------------

def run_chain(*deltas: str) -> list[tuple[str, str]]:
    sink = Recorder()
    stage = CodeFence(PyHighlight(sink))
    for d in deltas:
        stage.feed("content", d)
    stage.flush()
    return joined(sink.events)


class TestPyHighlight:
    def test_python_body_is_tokenised_and_fence_lines_stay_code(self):
        assert run_chain("```python\nreturn 1\n```") == [
            ("code", "```python\n"), ("py_kw", "return"), ("code_py", " "), ("py_num", "1"),
            ("code_py", "\n"), ("code", "```"),
        ]

    def test_fragmentation_does_not_change_spans(self):
        text = "```python\nx = f(\"a\")  # c\n```\n"
        whole = run_chain(text)
        for at in range(len(text) + 1):
            assert run_chain(text[:at], text[at:]) == whole, at

    def test_non_python_fence_is_untouched(self):
        assert run_chain("```bash\nreturn 1\n```") == [("code", "```bash\nreturn 1\n```")]

    def test_prose_is_untouched(self):
        assert run_chain("return 1 is ", "python") == [("content", "return 1 is python")]

    def test_held_identifier_is_emitted_before_a_pass_through_event(self):
        sink = Recorder()
        stage = PyHighlight(sink)
        stage.feed("code_py", "x = foo")
        stage.feed("tool_name", "Read")
        assert sink.events == [("code_py", "x = "), ("code_py", "foo"), ("tool_name", "Read")]

    def test_closing_fence_resets_scanner_state(self):
        # An unterminated string in the first fence must not swallow the second fence's body.
        spans = run_chain('```python\ns = "open\n```\n```python\nreturn 1\n```')
        assert ("py_kw", "return") in spans

    def test_call_and_builtin_colouring(self):
        spans = run_chain("```py\nprint(compute(len(x)))\n```")
        assert ("py_builtin", "print") in spans
        assert ("py_call", "compute") in spans
        assert ("py_builtin", "len") in spans


class TestTerminalPython:
    def test_colour_off_writes_plain_text(self):
        out = io.StringIO()
        term = Terminal(out=out, colour=False)
        chain = CodeFence(PyHighlight(term))
        chain.feed("content", "```python\nreturn len(x)\n```")
        chain.flush()
        assert out.getvalue() == "```python\nreturn len(x)\n```\n"

    def test_every_python_channel_has_a_colour_entry(self):
        for channel in ["code_py", "py_kw", "py_builtin", "py_call", "py_str", "py_num", "py_comment"]:
            assert channel in Terminal.COLOURS, channel
