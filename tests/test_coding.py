"""
The coding toolset: Read / Write / Edit / Bash as methods of one Workspace, confined to its root.
All against tmp_path — no network, no gate (the gate is generic and tested in test_confirm.py).
"""
import os

import pytest

from desh.llama.wire import ToolCall
from desh.tools import Tool, ToolRegistry
from desh_chat.coding import Workspace, coding_registry, edit_preview
from desh_chat.gate import describe_call, shorten


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("line 1\nline 2\nline 3\n")
    return Workspace(str(tmp_path))


# ---------------------
# Confinement
# ---------------------

class TestWorkspacePath:
    def test_relative_and_absolute_paths_inside_the_root_resolve(self, ws):
        assert ws.path("src/a.py") == os.path.join(ws.root, "src", "a.py")
        assert ws.path(os.path.join(ws.root, "src", "a.py")) == os.path.join(ws.root, "src", "a.py")
        assert ws.path(".") == ws.root

    @pytest.mark.parametrize("bad", ["../x", "src/../../x", "/etc/passwd", "~/x"])
    def test_paths_leaving_the_root_are_refused(self, ws, bad):
        with pytest.raises(ValueError, match="outside"):
            ws.path(bad)

    def test_a_symlink_anywhere_under_the_root_is_refused(self, ws, tmp_path):
        os.symlink(str(tmp_path / "src"), str(tmp_path / "link"))          # link inside -> inside
        with pytest.raises(ValueError, match="symlink"):
            ws.path("link/a.py")
        os.symlink("/etc", str(tmp_path / "out"))                          # link inside -> outside
        with pytest.raises(ValueError, match="symlink"):
            ws.path("out/passwd")

    def test_a_nonexistent_path_is_fine_for_writing(self, ws):
        assert ws.path("new/dir/file.txt").endswith("new/dir/file.txt")


# ---------------------
# Read / Write
# ---------------------

class TestReadWrite:
    def test_read_whole_file(self, ws):
        assert ws.read("src/a.py") == "line 1\nline 2\nline 3\n"

    def test_read_offset_and_limit_are_line_based_1_indexed(self, ws):
        assert ws.read("src/a.py", offset=2) == "line 2\nline 3\n"
        assert ws.read("src/a.py", offset=2, limit=1) == "line 2\n"
        assert ws.read("src/a.py", offset=10) == ""

    def test_read_over_the_cap_returns_whole_lines_and_says_where_to_continue(self, tmp_path):
        """The model's view of a big file: lines that fit, then a trailer with the file's length,
        the range shown, the cap (so it can size its next limit) and the offset to go on from.
        The whole reply stays within the cap, so the registry's blind cut never fires on top."""
        (tmp_path / "f.py").write_text("".join(f"line {i:03d} " + "x" * 40 + "\n" for i in range(1, 41)))    # 40 lines of 50 chars
        ws = Workspace(str(tmp_path), result_chars=300)
        out = ws.read("f.py")
        body, trailer = out.rsplit("\n", 1)
        assert body.splitlines() == [f"line {i:03d} " + "x" * 40 for i in range(1, 5)]   # whole lines only
        assert trailer == "[showing lines 1-4 of 40 (cap 300 chars); continue with offset=5]"
        assert len(out) <= 300
        assert ws.read("f.py", offset=5).startswith("line 005 ")                       # the pointer works
        assert ws.read("f.py", offset=39) == "line 039 " + "x" * 40 + "\nline 040 " + "x" * 40 + "\n"   # fits: no trailer
        assert coding_registry(str(tmp_path), result_chars=300).invoke("Read", '{"file_path": "f.py"}') == out    # unchanged by the bound

    def test_read_of_one_line_over_the_cap_cuts_it_by_characters(self, tmp_path):
        (tmp_path / "one.txt").write_text("y" * 2000 + "\n" + "z" * 10 + "\n")
        out = Workspace(str(tmp_path), result_chars=300).read("one.txt")
        assert out.startswith("y" * 50) and "z" not in out and len(out) <= 300
        assert "[showing lines 1-1 of 2" in out and "line truncated" in out and "offset=2" in out

    def test_edit_reads_the_file_whole_whatever_the_cap(self, tmp_path):
        """Regression: an Edit that went through the capped read() would miss a target past the
        cut, and write the cut content plus the trailer back over the file for a target inside it."""
        text = "".join(f"line {i:03d}\n" for i in range(1, 101))
        (tmp_path / "big.py").write_text(text)
        ws = Workspace(str(tmp_path), result_chars=200)
        assert ws.edit("big.py", "line 099", "LINE 099") == "One occurrence of `old_string` replaced."
        assert ws.edit("big.py", "line 001", "LINE 001") == "One occurrence of `old_string` replaced."
        assert (tmp_path / "big.py").read_text() == text.replace("line 099", "LINE 099").replace("line 001", "LINE 001")

    def test_read_missing_file_raises_for_invoke_to_report(self, ws):
        with pytest.raises(FileNotFoundError):
            ws.read("nope.py")

    def test_write_creates_parents_and_reports(self, ws):
        assert ws.write("new/dir/f.txt", "hi\n") == "wrote 3 characters to new/dir/f.txt"
        assert ws.read("new/dir/f.txt") == "hi\n"

    def test_write_overwrites(self, ws):
        ws.write("src/a.py", "x")
        assert ws.read("src/a.py") == "x"


# ---------------------
# Edit
# ---------------------

class TestEdit:
    def test_unique_match_is_replaced_and_the_rest_of_the_file_is_untouched(self, ws):
        out = ws.edit("src/a.py", "line 2", "LINE TWO")
        assert ws.read("src/a.py") == "line 1\nLINE TWO\nline 3\n"
        assert "replaced" in out
        assert "line 2" not in out          # the model already has old_string; do not send it back

    def test_multi_line_match_with_indentation(self, ws):
        ws.write("src/b.py", "def f():\n    return 1\n\n\ndef g():\n    return 1\n")
        ws.edit("src/b.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
        assert ws.read("src/b.py") == "def f():\n    return 2\n\n\ndef g():\n    return 1\n"

    def test_no_match_says_so_and_changes_nothing(self, ws):
        out = ws.edit("src/a.py", "line 9", "x")
        assert "not found" in out
        assert ws.read("src/a.py") == "line 1\nline 2\nline 3\n"

    def test_ambiguous_match_refuses_and_reports_the_count(self, ws):
        out = ws.edit("src/a.py", "line", "row")
        assert "3" in out and "replace_all" in out
        assert ws.read("src/a.py") == "line 1\nline 2\nline 3\n"

    def test_replace_all_replaces_every_occurrence(self, ws):
        out = ws.edit("src/a.py", "line", "row", replace_all=True)
        assert ws.read("src/a.py") == "row 1\nrow 2\nrow 3\n"
        assert "3" in out

    def test_missing_file_is_reported_not_created(self, ws, tmp_path):
        out = ws.edit("nope.py", "a", "b")
        assert "does not exist" in out
        assert not (tmp_path / "nope.py").exists()

    def test_empty_old_string_is_refused(self, ws):
        # "".count("") is len+1, and replace("", x) would interleave x between every character
        out = ws.edit("src/a.py", "", "x")
        assert ws.read("src/a.py") == "line 1\nline 2\nline 3\n"
        assert "empty" in out.lower()

    def test_confinement_applies(self, ws):
        with pytest.raises(ValueError, match="outside"):
            ws.edit("../x", "a", "b")


# ---------------------
# Bash
# ---------------------

class TestBash:
    def test_runs_in_the_root_and_returns_stdout(self, ws):
        assert ws.bash("list src", "ls src") == "a.py"
        assert ws.bash("where am I", "pwd") == ws.root

    def test_stderr_and_exit_code_are_reported(self, ws):
        out = ws.bash("exercise every channel", "echo out; echo err >&2; exit 3")
        assert out == "out\nstderr:\nerr\nexit code 3"

    def test_no_output(self, ws):
        assert ws.bash("nothing", "true") == "(no output)"

    def test_timeout(self, ws):
        assert ws.bash("hang", "sleep 5", timeout=1) == "command timed out after 1s"

    def test_output_over_the_cap_is_spilled_to_a_file_the_model_can_read(self, tmp_path):
        """Bash cannot cut for the model, so it keeps the whole output where Read can reach it and
        says so on its last line; the registry's cut keeps the tail, so the pointer survives."""
        from desh_chat.coding import SPILL_DIR
        ws = Workspace(str(tmp_path), result_chars=800)
        out = ws.bash("a lot", "python3 -c 'print(\"x\" * 1500)'")
        assert out.startswith("x" * 1500) and "the whole of it is saved at" in out.splitlines()[-1]
        rel = out.split("saved at ")[1].split(" ")[0]
        assert rel.startswith(SPILL_DIR) and (tmp_path / rel).read_text() == "x" * 1500
        bounded = coding_registry(str(tmp_path), result_chars=800).invoke("Bash", '{"reason": "r", "command": "python3 -c \'print(\\"x\\" * 1500)\'"}')
        assert len(bounded) <= 800 + 60 and "characters truncated" in bounded
        assert bounded.rstrip().endswith("Read it with offset and limit]") and "saved at " + SPILL_DIR in bounded

    def test_output_within_the_cap_is_not_spilled(self, tmp_path):
        from desh_chat.coding import SPILL_DIR
        ws = Workspace(str(tmp_path), result_chars=800)
        assert ws.bash("small", "echo hi") == "hi"
        assert not (tmp_path / SPILL_DIR).exists()

    def test_reason_is_mandatory_and_comes_before_the_command_in_the_schema(self, tmp_path):
        bash = coding_registry(str(tmp_path)).get("Bash")
        assert bash.identity == ("command",)     # the reason is wording, not part of the call
        assert list(bash.parameters["properties"]) == ["reason", "command", "timeout"]
        assert bash.parameters["required"] == ["reason", "command"]
        assert "why" in bash.parameters["properties"]["reason"]["description"]
        # the model must not be able to run a command without saying what it is for
        assert "rejected the arguments" in coding_registry(str(tmp_path)).invoke("Bash", '{"command": "ls"}')

    def test_a_mention_of_a_call_shows_the_path_or_the_command(self, tmp_path):
        """What the expiring line of the scratchpad block says a call was about (Tool.target)."""
        registry = coding_registry(str(tmp_path))
        assert registry.target("Read", '{"file_path": "src/a.py", "offset": 10}') == "src/a.py"
        assert registry.target("Edit", '{"file_path": "src/a.py", "old_string": "x", "new_string": "y"}') == "src/a.py"
        assert registry.target("Write", '{"file_path": "src/a.py", "content": "..."}') == "src/a.py"
        assert registry.target("Bash", '{"reason": "look", "command": "ls -la"}') == "ls -la"


# ---------------------
# Registry wiring and result bound
# ---------------------

class TestCodingRegistry:
    def test_names_and_policy(self, tmp_path):
        r = coding_registry(str(tmp_path))
        assert [t.name for t in r.tools] == ["Read", "Write", "Edit", "Bash"]
        assert [t.confirm for t in r.tools] == [False, True, True, True]

    def test_schemas_derive_from_the_methods_without_self_or_root(self, tmp_path):
        r = coding_registry(str(tmp_path))
        props = r.get("Read").parameters["properties"]
        assert list(props) == ["file_path", "offset", "limit"]
        assert r.get("Read").parameters["required"] == ["file_path"]
        assert "project root" in props["file_path"]["description"]
        assert r.get("Edit").parameters["properties"]["replace_all"] == {"type": "boolean", "description": "Replace every occurrence. Default False: old_string must occur exactly once."}

    def test_invoke_reports_confinement_and_io_errors_as_text(self, tmp_path):
        r = coding_registry(str(tmp_path))
        assert "outside the project root" in r.invoke("Read", '{"file_path": "../x"}')
        assert "FileNotFoundError" in r.invoke("Read", '{"file_path": "nope"}')

    def test_results_are_bounded_head_and_tail(self, tmp_path):
        r = ToolRegistry(max_result_chars=100).add(lambda: "a" * 80 + "b" * 70, name="big", confirm=False,
                                                    description="big", parameters={"type": "object", "properties": {}})
        out = r.invoke("big", "{}")
        assert out.startswith("a" * 75 + "\n") and out.endswith("\n" + "b" * 25)
        assert "[... 50 characters truncated ...]" in out
        assert ToolRegistry(max_result_chars=100).bound("short") == "short"

    def test_the_cap_defaults_to_the_16k_share_and_the_harness_sets_it(self, tmp_path):
        from desh.tools import DEFAULT_RESULT_CHARS
        assert ToolRegistry().max_result_chars == DEFAULT_RESULT_CHARS == 8192
        r = coding_registry(str(tmp_path), result_chars=300)
        assert r.max_result_chars == 300 and r.get("Bash") is not None

    def test_big_read_is_bounded_by_default(self, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 20000)
        out = coding_registry(str(tmp_path)).invoke("Read", '{"file_path": "big.txt"}')
        assert len(out) <= 8192 and "line truncated" in out


# ---------------------
# Edit preview: what the operator sees for an Edit at the gate (colour is off under pytest)
# ---------------------

class TestEditPreview:
    def preview(self, old, new, **extra):
        return edit_preview({"file_path": "x.py", "old_string": old, "new_string": new, **extra}).splitlines()

    def test_one_line_change_is_a_minus_and_a_plus_under_the_path(self):
        assert self.preview("    return a - b", "    return a + b") == ["x.py", "-    return a - b", "+    return a + b"]

    def test_unshared_fragments_group_all_removals_before_all_additions(self):
        assert self.preview("def f():\n    return 1", "def g(x):\n    return x * 2") == [
            "x.py", "-def f():", "-    return 1", "+def g(x):", "+    return x * 2"]

    def test_shared_lines_appear_as_context(self):
        assert self.preview("foo(\n    1,\n)", "bar(\n    2,\n)") == ["x.py", "-foo(", "-    1,", "+bar(", "+    2,", " )"]

    def test_headers_are_dropped(self):
        out = "\n".join(self.preview("a", "b"))
        assert "---" not in out and "+++" not in out and "@@" not in out

    def test_replace_all_is_shown(self):
        assert self.preview("a", "b", replace_all=True)[-1] == "(replace_all: true)"
        assert "(replace_all" not in "\n".join(self.preview("a", "b"))

    def test_missing_path_is_named(self):
        assert edit_preview({"old_string": "a", "new_string": "b"}).splitlines()[0] == "(no file)"

    def test_malformed_call_raises_and_the_gate_falls_back(self, tmp_path):
        with pytest.raises(Exception):
            edit_preview({"file_path": "x.py"})
        tool = coding_registry(str(tmp_path)).get("Edit")
        tc = ToolCall(index=0, id="c", type="function", name="Edit", arguments='{"file_path": "x.py"}')
        assert describe_call(tc, tool) == '→ Edit\n  file_path: "x.py"'

    def test_colour_wraps_only_when_stdout_is_a_tty(self, monkeypatch):
        from desh import render
        monkeypatch.setattr(render.c_out, "enabled", True)
        out = edit_preview({"file_path": "x.py", "old_string": "a", "new_string": "b"})
        assert "\033[31m-a\033[0m" in out and "\033[32m+b\033[0m" in out


# ---------------------
# What the operator sees at the gate
# ---------------------

class TestDescribeCall:
    def test_multi_line_values_become_blocks(self):
        tc = ToolCall(index=0, id="c", type="function", name="Write", arguments='{"file_path": "a.py", "content": "x = 1\\ny = 2\\n"}')
        assert describe_call(tc).splitlines() == ["→ Write", "  content:", "    x = 1", "    y = 2", '  file_path: "a.py"']

    def test_scalar_values_are_shown_in_sorted_key_order(self):
        tc = ToolCall(index=0, id="c", type="function", name="Bash", arguments='{"timeout": 5, "reason": "list files", "command": "ls"}')
        assert describe_call(tc) == '→ Bash\n  command: "ls"\n  reason: "list files"\n  timeout: 5'

    def test_the_reason_value_is_rendered_in_the_reason_colour(self, monkeypatch):
        from desh import render
        monkeypatch.setattr(render.c_out, "enabled", True)
        tc = ToolCall(index=0, id="c", type="function", name="Bash", arguments='{"command": "ls", "reason": "list files"}')
        out = describe_call(tc)
        assert '\033[38;5;208m"list files"' in out
        assert '\033[0m"ls"' in out

    def test_only_the_reason_key_gets_the_reason_colour(self, monkeypatch):
        from desh import render
        monkeypatch.setattr(render.c_out, "enabled", True)
        tc = ToolCall(index=0, id="c", type="function", name="Bash", arguments='{"reason": "list files", "command": "ls"}')
        out = describe_call(tc)
        assert "\033[38;5;208m" in out
        assert '\033[38;5;208m"ls"' not in out

    def test_non_object_arguments_fall_back_to_the_wire_string(self):
        assert describe_call(ToolCall(index=0, id="c", type="function", name="T", arguments="oops")) == "→ T(oops)"
        assert describe_call(ToolCall(index=0, id="c", type="function", name="T", arguments="{}")) == "→ T({})"

    def test_a_tool_preview_replaces_the_generic_listing(self):
        tool = Tool.define(lambda a: a, name="T", description="t", parameters={"type": "object", "properties": {}},
                           preview=lambda args: f"PREVIEW of {args['a']}")
        tc = ToolCall(index=0, id="c", type="function", name="T", arguments='{"a": "x"}')
        assert describe_call(tc, tool) == "→ T\nPREVIEW of x"

    def test_a_raising_preview_falls_back_to_the_generic_listing(self):
        def boom(args):
            raise KeyError("old_string")
        tool = Tool.define(lambda a: a, name="T", description="t", parameters={"type": "object", "properties": {}}, preview=boom)
        tc = ToolCall(index=0, id="c", type="function", name="T", arguments='{"a": "x"}')
        assert describe_call(tc, tool) == '→ T\n  a: "x"'

    def test_edit_is_registered_with_a_preview(self, tmp_path):
        assert coding_registry(str(tmp_path)).get("Edit").preview is not None
        assert coding_registry(str(tmp_path)).get("Write").preview is None

    def test_shorten_indents_lines_and_bounded(self):
        assert shorten("a\nb") == "    a\n    b"
        assert shorten("x" * 300) == "    " + "x" * 200 + " ... 300 chars"


# ---------------------
# What a round echoes of a Write or Edit once it ran (Tool.fold)
# ---------------------

class TestWrittenFold:
    """A call's arguments are echoed in every later request and never expire: a Write carries the
    whole file, and a round that wrote two files could not be checkpointed at 4k. Once the call
    ran, the round records the path, a head and the size; the file is one Read away."""

    def test_a_long_write_folds_to_its_head_and_size(self):
        from desh_chat.coding import WRITTEN_HEAD_CHARS, fold_written
        folded = fold_written({"file_path": "wc.py", "content": "x" * 3000})
        assert folded["file_path"] == "wc.py" and folded["content"].startswith("x" * WRITTEN_HEAD_CHARS + "... [3000 characters written")
        assert len(folded["content"]) < 200

    def test_a_short_write_and_a_malformed_one_are_kept(self):
        from desh_chat.coding import fold_written
        assert fold_written({"file_path": "a", "content": "short"}) == {"file_path": "a", "content": "short"}
        assert fold_written({"file_path": "a"}) == {"file_path": "a"}

    def test_a_long_edit_folds_both_strings(self):
        from desh_chat.coding import fold_edited
        args = {"file_path": "a", "old_string": "o" * 500, "new_string": "n" * 500, "replace_all": False}
        folded = fold_edited(args)
        assert folded["old_string"].endswith("... [500 characters]") and folded["new_string"].endswith("... [500 characters]") and folded["replace_all"] is False
        assert fold_edited({"file_path": "a", "old_string": "x", "new_string": "y"}) == {"file_path": "a", "old_string": "x", "new_string": "y"}

    def test_the_round_records_the_fold_after_the_call_ran(self, tmp_path, make_state):
        """The full arguments run; the round echoes the fold from then on (as a delegate brief does)."""
        import json
        from desh.llama.wire import ToolCall
        from desh_chat.events import ExecuteToolCalls
        from desh_chat.state import PendingTurn, Round, Settings
        from conftest import MODELS
        registry = coding_registry(str(tmp_path))
        assert registry.get("Write").fold is not None and registry.get("Edit").fold is not None
        tc = ToolCall(index=0, id="w_0", type="function", name="Write", arguments=json.dumps({"file_path": "big.py", "content": "y" * 2000}))
        state = make_state(pending=PendingTurn("q").add_round(Round("", (tc,), tokens=10)), tools=registry,
                           settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, auto=True))
        new_state, _ = ExecuteToolCalls(0).execute(state)
        assert (tmp_path / "big.py").read_text() == "y" * 2000                       # the call ran whole
        recorded = json.loads(new_state.pending.rounds[-1].tool_calls[0].arguments)
        assert "2000 characters written" in recorded["content"] and len(recorded["content"]) < 200
