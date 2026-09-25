"""
Edit (Workspace.edit): one replacement or several in one file in one call, applied in order,
all or none. What a whole-file rewrite does, carrying only the lines that change — at 16K a
rewrite needs the whole file read back in and written out again, and a checkpoint folds it away
in between.
"""
import argparse
import json

import pytest

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer
from desh_chat.cli import build_tools
from desh_chat.coding import Workspace, edit_parameters, edit_preview, fold_edited
from desh_chat.state import InferenceEngine, Settings


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\n| --context | 16384 |\n| --model | old |\n\nText one.\nText two.\n", encoding="utf-8")
    return Workspace(str(tmp_path)), tmp_path / "doc.md"


def e(old, new, **kw):
    return {"old_string": old, "new_string": new, **kw}


class TestApply:
    def test_the_edits_apply_in_order_each_to_the_text_the_earlier_ones_left(self, ws):
        w, doc = ws
        answer = w.edit("doc.md", [e("16384", "65536"), e("| old |", "| new |"), e("Text one.", "Text one, edited.")])
        assert answer.startswith("3 edits applied to doc.md (3 replacements;")
        assert doc.read_text() == "# Title\n\n| --context | 65536 |\n| --model | new |\n\nText one, edited.\nText two.\n"
        # a later edit can build on an earlier one
        w.edit("doc.md", [e("65536", "X"), e("| X |", "| 131072 |")])
        assert "| --context | 131072 |" in doc.read_text()

    def test_one_failure_applies_nothing_and_every_failure_is_reported(self, ws):
        w, doc = ws
        before = doc.read_text()
        answer = w.edit("doc.md", [e("16384", "65536"), e("missing", "x"), e("Text", "T"), e("a", "a")])
        assert doc.read_text() == before
        assert answer.startswith("No edit applied (3 of 4 failed):")
        assert "edit 2: old_string not found" in answer and "edit 3: old_string occurs 2 times" in answer
        assert "edit 4: old_string and new_string are identical" in answer

    def test_replace_all_replaces_every_occurrence(self, ws):
        w, doc = ws
        assert "(2 replacements;" in w.edit("doc.md", [e("Text", "Line", replace_all=True)])
        assert doc.read_text().count("Line") == 2

    def test_an_edit_whose_target_an_earlier_one_changed_says_so(self, ws):
        w, _ = ws
        answer = w.edit("doc.md", [e("16384", "65536"), e("16384", "1")])
        assert "edit 2: old_string not found (an earlier edit may have changed it)" in answer

    def test_bad_input_is_answered_not_raised(self, ws, tmp_path):
        w, _ = ws
        assert w.edit("nope.md", [e("a", "b")]) == "File nope.md does not exist."
        assert "non-empty list" in w.edit("doc.md", [])
        assert "edit 1: not an object" in w.edit("doc.md", ["x"])
        assert "edit 1: old_string cannot be empty" in w.edit("doc.md", [e("", "b")])
        with pytest.raises(Exception):
            w.edit("../outside.md", [e("a", "b")])      # confined to the root


class TestSchemaAndEcho:
    def test_the_schema_spells_out_one_edit(self, tmp_path):
        schema = edit_parameters(Workspace(str(tmp_path)))
        edits = schema["properties"]["edits"]
        assert schema["required"] == ["file_path", "edits"] and edits["type"] == "array" and edits["minItems"] == 1
        assert edits["items"]["required"] == ["old_string", "new_string"] and "replace_all" in edits["items"]["properties"]
        assert edits["description"].startswith("The replacements, applied in order")

    def test_a_long_call_is_echoed_folded_and_a_short_one_whole(self):
        short = {"file_path": "a.md", "edits": [e("x", "y")]}
        assert fold_edited(short) == short
        long = {"file_path": "a.md", "edits": [e("x" * 200, "y" * 300), e("z" * 10, "w" * 20)]}
        assert fold_edited(long) == {"file_path": "a.md", "folded": "2 edits replacing 210 characters with 320; the file holds the new text"}

    def test_the_preview_numbers_each_edit(self):
        text = edit_preview({"file_path": "a.md", "edits": [e("old one", "new one"), e("old two", "new two")]})
        assert "a.md — 2 edits" in text and "edit 1" in text and "edit 2" in text and "new two" in text


class TestRegistration:
    def build(self, tmp_path, **flags):
        args = argparse.Namespace(workspace=str(tmp_path), read=True, write=True, edit=True, bash=True, current_time=False,
                                  delegate=True, delegate_records=False, debug=False, scratchpad=False, memory="", tool_cap=10.0)
        for k, v in flags.items():
            setattr(args, k, v)
        inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=FakeServer(script=[]), port=PORT)
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192)
        return build_tools(args, inference, settings)

    def test_offered_to_the_main_agent_with_the_flag_and_always_to_the_subagents(self, tmp_path):
        plain = self.build(tmp_path, edit=False)
        assert "Edit" not in plain and "Edit" in plain.get("delegate").fn.__self__.tools
        tools = self.build(tmp_path)
        tool = tools.get("Edit")
        assert tool.confirm and tool.acting and tool.target == "file_path"     # it changes the workspace
        # both registries carry the one schema, with the shape of an edit spelled out
        assert tool.parameters == tools.get("delegate").fn.__self__.tools.get("Edit").parameters
        assert "items" in tool.parameters["properties"]["edits"]

    def test_a_call_through_the_registry(self, tmp_path):
        (tmp_path / "a.md").write_text("one\ntwo\n")
        tools = self.build(tmp_path)
        out = tools.invoke("Edit", json.dumps({"file_path": "a.md", "edits": [e("one", "1"), e("two", "2")]}))
        assert out.startswith("2 edits applied") and (tmp_path / "a.md").read_text() == "1\n2\n"
