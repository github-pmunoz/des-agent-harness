"""
The coding toolset: Read / Write / Edit / Bash as methods of one Workspace, confined to its root.
All against tmp_path — no network, no gate (the gate is generic and tested in test_confirm.py).
"""
import os

import pytest

from desh.llama.client import ToolCall
from desh.tools import ToolRegistry
from desh_chat.coding import Workspace, coding_registry
from desh_chat.events import describe_call, shorten


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

    def test_reason_is_mandatory_and_comes_before_the_command_in_the_schema(self, tmp_path):
        bash = coding_registry(str(tmp_path)).get("Bash")
        assert list(bash.parameters["properties"]) == ["reason", "command", "timeout"]
        assert bash.parameters["required"] == ["reason", "command"]
        assert "why" in bash.parameters["properties"]["reason"]["description"]
        # the model must not be able to run a command without saying what it is for
        assert "rejected the arguments" in coding_registry(str(tmp_path)).invoke("Bash", '{"command": "ls"}')


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

    def test_big_read_is_bounded_by_default(self, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 20000)
        out = coding_registry(str(tmp_path)).invoke("Read", '{"file_path": "big.txt"}')
        assert len(out) < 8100 and "truncated" in out


# ---------------------
# What the operator sees at the gate
# ---------------------

class TestDescribeCall:
    def test_multi_line_values_become_blocks(self):
        tc = ToolCall(index=0, id="c", type="function", name="Write", arguments='{"file_path": "a.py", "content": "x = 1\\ny = 2\\n"}')
        assert describe_call(tc).splitlines() == ["→ Write", '  file_path: "a.py"', "  content:", "    x = 1", "    y = 2"]

    def test_scalar_values_stay_inline_in_wire_order(self):
        tc = ToolCall(index=0, id="c", type="function", name="Bash", arguments='{"reason": "list files", "command": "ls", "timeout": 5}')
        assert describe_call(tc) == '→ Bash\n  reason: "list files"\n  command: "ls"\n  timeout: 5'

    def test_non_object_arguments_fall_back_to_the_wire_string(self):
        assert describe_call(ToolCall(index=0, id="c", type="function", name="T", arguments="oops")) == "→ T(oops)"
        assert describe_call(ToolCall(index=0, id="c", type="function", name="T", arguments="{}")) == "→ T({})"

    def test_shorten_is_one_line_and_bounded(self):
        assert shorten("a\nb") == "a⏎b"
        assert shorten("x" * 300) == "x" * 199 + "…"
