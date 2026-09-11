"""
Gutter: the stream wrapper that draws a prefix at every line start, used to indent a subagent's
output under its banner. The cases that matter are the ones a plain "prefix each line of this
string" would get wrong: chunks that split a line across writes, "\\r" redraws, and colour that
must survive the prefix's reset. The last test drives a real child run and checks its lines.
"""
import io

import pytest

from desh.engine import Engine
from desh.render import Gutter
from desh_chat import gate
from desh_chat.events import TurnStart
from desh_chat.gate import Answer

from conftest import FakeServer


@pytest.fixture
def always_yes(monkeypatch):
    monkeypatch.setattr(gate, "ask", lambda tc: Answer("yes"))


def gutter():
    out = io.StringIO()
    return out, Gutter(out, "| ")


class TestGutter:
    def test_each_line_gets_the_prefix(self):
        out, g = gutter()
        g.write("one\ntwo\n")
        assert out.getvalue() == "| one\n| two\n"

    def test_a_line_split_across_writes_is_prefixed_once(self):
        out, g = gutter()
        for chunk in ("hel", "lo wor", "ld\nne", "xt"):
            g.write(chunk)
        assert out.getvalue() == "| hello world\n| next"

    def test_a_trailing_newline_does_not_open_a_prefixed_empty_line(self):
        out, g = gutter()
        g.write("done\n")
        assert out.getvalue() == "| done\n"       # the next line's prefix waits for its content

    def test_empty_lines_stay_empty(self):
        out, g = gutter()
        g.write("a\n\nb\n")
        assert out.getvalue() == "| a\n\n| b\n"

    def test_carriage_return_redraw_keeps_the_prefix_before_the_clear(self):
        out, g = gutter()
        g.write("⚙ Bash … 10 chars")
        g.write("\r\033[K⚙ Bash … 20 chars\n")
        assert out.getvalue() == "| ⚙ Bash … 10 chars\r| \033[K⚙ Bash … 20 chars\n"

    def test_content_colour_is_re_emitted_after_the_prefix(self):
        out, g = gutter()
        g.write("\033[2mline one\nline two\033[0m\nplain\n")
        assert out.getvalue() == "| \033[2mline one\n| \033[2mline two\033[0m\n| plain\n"

    def test_stacked_colours_are_kept_until_the_reset(self):
        out, g = gutter()
        g.write("\033[2m\033[33mdim yellow\nstill\033[0m\n")
        assert out.getvalue() == "| \033[2m\033[33mdim yellow\n| \033[2m\033[33mstill\033[0m\n"

    def test_a_bare_escape_after_the_line_break_does_not_open_a_line(self):
        # Terminal wraps a chunk as colour + text + "\n" + reset: the reset must not draw a lone prefix
        out, g = gutter()
        g.write("\033[2mdone\n\033[0m")
        g.write("next\n")
        assert out.getvalue() == "| \033[2mdone\n\033[0m| next\n"

    def test_colour_set_before_the_carriage_return_is_carried_into_the_redraw(self):
        # the progress line arrives as dim + "\r\033[K" + text + "\n" + reset, all in one chunk
        out, g = gutter()
        g.write("\033[2m\r\033[K⚙ Bash … 157 chars\n\033[0m")
        assert out.getvalue() == "\033[2m\r| \033[2m\033[K⚙ Bash … 157 chars\n\033[0m"

    def test_a_prefix_with_a_leading_reset_is_drawn_clean_then_the_content_colour_returns(self):
        out = io.StringIO()
        g = Gutter(out, "\033[0m\033[33m|\033[0m ")
        g.write("\033[2mone\ntwo\033[0m\n")
        assert out.getvalue() == "\033[0m\033[33m|\033[0m \033[2mone\n\033[0m\033[33m|\033[0m \033[2mtwo\033[0m\n"

    def test_everything_else_is_delegated_to_the_wrapped_stream(self):
        out, g = gutter()
        assert g.isatty() is out.isatty()
        g.flush()
        assert g.write("x") == 1

    def test_a_gutter_in_a_gutter_nests_the_prefixes(self):
        out = io.StringIO()
        inner = Gutter(Gutter(out, "| "), "> ")
        inner.write("deep\n")
        assert out.getvalue() == "| > deep\n"


class TestChildRunIsGuttered:
    DELEGATION = {"tool_calls": [{"name": "delegate", "arguments": '{"task": "count the files"}'}]}

    def test_every_child_line_sits_behind_the_gutter_and_the_banners_do_not(self, make_state, always_yes, no_esc_watcher, capsys):
        from test_delegate import parent_with_delegate
        server = FakeServer(script=[self.DELEGATION, {"content": "There are 12 files."}, {"content": "Twelve."}])
        state = parent_with_delegate(make_state, server)
        Engine[type(state)]().run(state, seed=[TurnStart("how many files?")])
        lines = capsys.readouterr().out.splitlines()
        start = next(i for i, l in enumerate(lines) if "subagent starts" in l)
        end = next(i for i, l in enumerate(lines) if "back to the main agent" in l)
        between = [l for l in lines[start + 1:end] if l]
        assert between, "the child printed nothing"
        assert all(l.startswith("│ ") for l in between)
        assert any("12 files" in l for l in between)
        assert not lines[start].startswith("│") and not lines[end].startswith("│")
