"""The geo plugin and what it needs from the harness: tool results that carry images, the
metrology instruments, and the tools a layout agent is offered per arm."""
import json

import pytest
from PIL import Image

from test_turn_loop import run_chat

from desgeo import Layout, Metrology, Rect
from desh.tools import ToolOutput, ToolRegistry
from desh_chat.geo import NOT_SUBMITTED, GeoSession, GeoSettings, GeoTask
from desh_chat.state import EXPIRED_RESULT, Round, ToolResult
from desh.llama.wire import ToolCall

U_TARGET = "outer = rect(237, 143, 318, 411)\nslot = rect(311, 260, 170, 294)\nM1 = outer - slot\n"
U_EXACT = "M1 = rect(237, 143, 318, 411) - rect(311, 260, 170, 294)"


def u_task(**kw) -> GeoTask:
    return GeoTask(id="u", width=800, height=800, layers={"M1": (0, 0, 0)}, target=U_TARGET,
                   outputs=("M1",), prompt="Reproduce M1.", **kw)


def via_task() -> GeoTask:
    return GeoTask(id="via", width=600, height=400, layers={"M1": (220, 40, 40), "VIA": (40, 70, 220)},
                   inputs="M1 = rect(40, 60, 520, 50)\n", outputs=("VIA",),
                   target="v = rect(58, 73, 24, 24)\nVIA = v | move(v, 120, 0)\n")


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(path)
    return str(path)


# ---- harness: images in tool results ----------------------------------------------------------

class TestImageResults:
    def test_a_result_with_images_is_a_content_list_and_a_plain_one_is_text(self, png):
        r = ToolResult("c0", "geo_render", "target rendered", (png,))
        m = r.message()
        assert m["content"][0] == {"type": "text", "text": "target rendered"}
        assert m["content"][1]["type"] == "image_url"
        assert m["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert ToolResult("c1", "x", "plain").message()["content"] == "plain"

    def test_a_stubbed_result_drops_its_images(self, png):
        tc = ToolCall(0, "c0", "function", "geo_render", "{}")
        rnd = Round("", (tc,), (ToolResult("c0", "geo_render", "t", (png,)),))
        assert rnd.messages(stubbed=True)[1]["content"] == EXPIRED_RESULT
        assert f"[image {png}]" in rnd.transcript()

    def test_images_round_trip_and_are_priced(self, png):
        r = ToolResult("c0", "n", "t", (png,))
        assert ToolResult.from_dict(json.loads(json.dumps(r.to_dict()))) == r
        assert "images" not in ToolResult("c0", "n", "t").to_dict()
        assert r.tokens() > ToolResult("c0", "n", "t").tokens() + 500

    def test_registry_call_keeps_images_and_bounds_only_the_text(self, png):
        def show() -> str:
            """Show."""
            return ToolOutput("x" * 5000, (png,))
        reg = ToolRegistry(max_result_chars=100).add(show)
        out = reg.call("show", "{}")
        assert out.images == (png,) and len(out.text) < 200
        assert reg.invoke("show", "{}") == out.text


# ---- metrology --------------------------------------------------------------------------------

class TestMetrology:
    def layout(self):
        L = Layout(400, 400)
        L.add("M1", Rect(50, 50, 100, 40), Rect(150, 90, 40, 40),      # corner touch: two shapes
              Rect(250, 50, 37, 200))
        return L

    def test_shapes_are_edge_connected_and_numbered_bottom_up(self):
        m = Metrology(self.layout())
        assert m.shape_count("M1") == 3
        assert [h["shape"] for h in m.inspect(60, 60)] == [1]
        assert [h["shape"] for h in m.inspect(260, 60)] == [2]
        assert [h["shape"] for h in m.inspect(160, 100)] == [3]

    def test_ruler_snaps_to_edges_and_reads_straight_across(self):
        r = Metrology(self.layout()).measure(245, 150, 292, 158, radius=10)
        assert (r.a.x, r.b.x, r.dx, r.dy) == (250, 287, 37, 0)
        assert "left edge x=250" in r.a.describe() and "right edge x=287" in r.b.describe()

    def test_ruler_prefers_a_vertex_and_reports_nothing_in_range_as_free(self):
        m = Metrology(self.layout())
        a, _ = m.snap(253, 47, radius=10)
        assert (a.kind, a.x, a.y) == ("vertex", 250, 50)
        free, others = m.snap(350, 350, radius=10)
        assert free.kind == "free" and others == ()

    def test_ambiguity_lists_other_distinct_candidates(self):
        L = Layout(200, 200)
        L.add("M1", Rect(10, 10, 50, 50), Rect(64, 10, 50, 50))      # a 4-unit gap
        best, others = Metrology(L).snap(62, 30, radius=5)
        assert best.kind == "edge" and others and {best.x, others[0].x} == {60, 64}

    def test_auto_measure_width_inside_and_space_in_a_gap(self):
        m = Metrology(self.layout())
        inside = m.auto_measure(260, 100, "M1")
        assert inside["inside"] and inside["x"]["length"] == 37 and inside["y"]["length"] == 200
        gap = m.auto_measure(220, 100, "M1")
        assert not gap["inside"] and gap["x"]["from"] == 190 and gap["x"]["to"] == 250
        assert gap["x"]["between"] == [3, 2]


# ---- the plugin -------------------------------------------------------------------------------

class TestGeoSession:
    def test_exact_submission_is_logged_with_cost(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        out = s.submit(U_EXACT)
        assert out.startswith("submission 1: ALL EXACT.") and "cost: 3 operations, 11 variables" in out
        rec = json.loads((tmp_path / "submissions.jsonl").read_text().splitlines()[0])
        assert rec["exact"] and (rec["ops"], rec["variables"]) == (3, 11) and rec["layers"]["M1"]["iou"] == 1.0

    def test_feedback_levels(self, tmp_path):
        near = "M1 = rect(240, 140, 320, 410) - rect(310, 260, 170, 300)"
        mismatch = GeoSession(u_task(), str(tmp_path / "a")).submit(near)
        assert "IoU 0.936" in mismatch and "missing 2103 in 5 rects" in mismatch and "extra 3285" in mismatch
        iou = GeoSession(u_task(), str(tmp_path / "b"), GeoSettings(feedback="iou")).submit(near)
        assert "IoU 0.936" in iou and "missing" not in iou

    def test_rejected_program_names_the_line(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert s.submit("M1 = rect(1, 1, 1)") == "submission 1 rejected: line 1: rect() takes 4 arguments, got 3"
        assert s.submit("x = rect(1, 1, 1, 1)").startswith("submission 2 rejected: output 'M1' is never assigned")

    def test_inputs_are_readable_and_rendered_with_the_target(self, tmp_path):
        s = GeoSession(via_task(), str(tmp_path))
        assert s.task.input_layers() == ("M1",)
        assert "ALL EXACT" in s.submit("v = rect(58, 73, 24, 24)\nVIA = (v | move(v, 120, 0)) & M1")
        out = s.render_view("target")
        assert isinstance(out, ToolOutput) and out.images[0].endswith("render-001-target.png")
        assert Image.open(out.images[0]).size == (800, 533)

    def test_render_needs_a_submission_for_current_and_diff(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert s.render_view("diff").startswith("Nothing submitted yet")
        s.submit(U_EXACT)
        out = s.render_view("diff", window=[200, 100, 600, 600], labels=True)
        assert isinstance(out, ToolOutput) and "x 200..600" in out.text
        assert s.render_view("target", window=[0, 0, 900, 10]).startswith("window")

    def test_measure_persists_rulers_and_the_bias_arm_shifts_lengths(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert "dx = 74, dy = 0" in s.measure(240, 300, 305, 305)
        assert len(s.rulers) == 1
        assert "1 ruler(s)" in s.render_view("target").text
        biased = GeoSession(u_task(), str(tmp_path / "b"), GeoSettings(ruler_bias=3))
        assert "dx = 77, dy = 0" in biased.measure(240, 300, 305, 305)
        assert "width 77" in biased.auto_measure(270, 300)

    def test_auto_measure_and_inspect_read_the_target(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert "space 170 (x 311..481), between shape 1 and shape 1" in s.auto_measure(400, 400)
        assert "M1 shape 1: bbox [237, 143, 318, 411], area 80718, 8 vertices" in s.inspect(250, 150)
        assert s.inspect(700, 700) == "nothing at (700, 700) on target"

    def test_arms_register_only_their_tools(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert [t.name for t in s.register(ToolRegistry(), ()).tools] == ["geo_submit"]
        names = [t.name for t in s.register(ToolRegistry(), ("render", "measure")).tools]
        assert names == ["geo_submit", "geo_render", "geo_measure"]
        with pytest.raises(ValueError, match="unknown geo tool"):
            s.register(ToolRegistry(), ("zoom",))

    def test_origin_changes_the_picture_the_prompt_and_the_words_not_the_geometry(self, tmp_path):
        down = GeoSession(u_task(), str(tmp_path / "d"), GeoSettings(origin="top-left"))
        up = GeoSession(u_task(), str(tmp_path / "u"))
        assert "y grows downward" in u_task().prompt_text("top-left") and "top-left corner (x, y)" in u_task().prompt_text("top-left")
        assert "y grows upward" in u_task().prompt_text()
        assert down.submit(U_EXACT).startswith("submission 1: ALL EXACT")
        a = Image.open(up.render_view("target").images[0])
        b = Image.open(down.render_view("target").images[0])
        assert a.transpose(Image.FLIP_TOP_BOTTOM).tobytes() == b.tobytes()
        assert "bottom edge y=143" in up.measure(300, 146, 300, 150)
        assert "top edge y=143" in down.measure(300, 146, 300, 150)

    def test_render_text_states_the_pixel_mapping(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert "layout x = px, y = 800 - py" in s.render_view("target").text
        zoom = s.render_view("target", window=[200, 100, 600, 500]).text
        assert "x = 200 + px / 2, y = 500 - py / 2" in zoom and "1 layout unit = 2 px" in zoom
        down = GeoSession(u_task(), str(tmp_path / "d"), GeoSettings(origin="top-left", ticks=100))
        assert "x = px - 44, y = py - 14" in down.render_view("target").text

    def test_prompt_states_frame_outputs_and_inputs_but_no_image(self):
        text = via_task().prompt_text()
        assert "600 x 400" in text and "output layer VIA" in text and "Input layer M1 is given" in text
        assert "image" not in text.lower() and "render" not in text.lower()

    def test_task_file_loads_programs_from_files(self, tmp_path):
        (tmp_path / "t.geo").write_text(U_TARGET)
        (tmp_path / "t.json").write_text(json.dumps({"width": 800, "height": 800, "layers": {"M1": [0, 0, 0]},
                                                     "outputs": ["M1"], "target": "@t.geo"}))
        t = GeoTask.load(str(tmp_path / "t.json"))
        assert t.id == "t" and t.target == U_TARGET


class TestGeoLoop:
    def test_render_then_submit_through_the_turn_loop(self, make_state, no_esc_watcher, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        tools = s.register(ToolRegistry(), ("render",))
        script = [
            {"tool_calls": [{"name": "geo_render", "arguments": '{"view": "target"}'}]},
            {"tool_calls": [{"name": "geo_submit", "arguments": json.dumps({"program": U_EXACT})}]},
            {"content": "Done."},
        ]
        final, server = run_chat(make_state, script, ["reproduce M1"], tools=tools)
        turn = final.history.turns[0]
        assert turn.assistant == "Done." and "ALL EXACT" in turn.rounds[1].results[0].content
        # the image went out with the render's result in every later request of the turn
        for _, req in server.calls[1:]:
            tool_msg = [m for m in req.messages if m["role"] == "tool"][0]
            assert tool_msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        # once the turn is history, the result is a stub and the pixels are gone
        assert all(m.get("content") == EXPIRED_RESULT for m in turn.messages() if m["role"] == "tool")


class TestTaskCheck:
    def run_task(self, make_state, script, tmp_path, **overrides):
        from conftest import FakeServer
        from test_turn_loop import with_server
        from desh.engine import Engine
        from desh_chat.events import TurnStart
        s = GeoSession(u_task(), str(tmp_path))
        server = FakeServer(script=script)
        state = with_server(make_state, server, tools=s.register(ToolRegistry(), ()), operator=False,
                            task_check=s.unfinished, **overrides)
        return Engine[type(state)]().run(state, seed=[TurnStart("reproduce M1")]), server

    def test_an_answer_with_nothing_submitted_is_nudged_until_the_budget_runs_out(self, make_state, no_esc_watcher, tmp_path):
        script = [{"content": "It is a U."},
                  {"tool_calls": [{"name": "geo_submit", "arguments": json.dumps({"program": U_EXACT})}]},
                  {"content": "Done."}]
        final, server = self.run_task(make_state, script, tmp_path, max_nudges=1)
        assert [t.user for t in final.history.turns] == ["reproduce M1", NOT_SUBMITTED]
        assert final.nudges == 1 and "ALL EXACT" in final.history.turns[1].rounds[0].results[0].content

    def test_no_nudge_without_budget_or_once_submitted(self, make_state, no_esc_watcher, tmp_path):
        final, _ = self.run_task(make_state, [{"content": "It is a U."}], tmp_path / "a", max_nudges=0)
        assert len(final.history.turns) == 1
        script = [{"tool_calls": [{"name": "geo_submit", "arguments": json.dumps({"program": "M1 = rect(0, 0, 10, 10)"})}]},
                  {"content": "Close enough."}]
        final, _ = self.run_task(make_state, script, tmp_path / "b", max_nudges=2)
        assert len(final.history.turns) == 1 and final.nudges == 0

    def test_unfinished_counts_only_accepted_submissions(self, tmp_path):
        s = GeoSession(u_task(), str(tmp_path))
        assert s.unfinished() == NOT_SUBMITTED
        s.submit("M1 = rect(1, 1, 1)")
        assert s.unfinished() == NOT_SUBMITTED
        s.submit("M1 = rect(0, 0, 10, 10)")
        assert s.unfinished() is None
