# Author: Aarav Shah
# Portfolio: aaravshah1311.is-great.net
# github: github.com/aaravshah1311

"""Provider feed repair — parallel tool calls (the real 400s).

Both wire formats require that an assistant tool call be answered IMMEDIATELY
and COMPLETELY by the next message(s). The failure modes pinned here are the
whole-request 400s a live provider actually returned:

    OpenAI/AgentRouter: an assistant message with two `tool_calls` was followed
        by only ONE `role: tool` reply, because the old repair checked `out[-1]`
        for the declaration and the second tool message saw the first as its
        "preceding assistant". Every parallel-tool turn died.
    Anthropic-compatible: "`tool_use` ids were found without `tool_result`
        blocks immediately after", the same dangling half from the other side.

Two invariants beyond the happy path:

* An unanswered call is STRIPPED from its assistant message (the message is
  dropped if that empties it). Dropping the result while keeping the call is the
  same bug in the other direction.
* Repair is READ-ONLY with respect to its input. The loops rebind the returned
  list, but the dicts are shared with the feed they passed in, so editing one
  in place would leak a fix between iterations and between surfaces.
"""

import copy

from agent2.llm import providers as P


def _openai_assistant(call_ids, content=""):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}
                           for cid in call_ids]}


def _anth_assistant(use_ids):
    return {"role": "assistant",
            "content": [{"type": "text", "text": "working"},
                        *[{"type": "tool_use", "id": u, "name": "t", "input": {}}
                          for u in use_ids]]}


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def test_openai_parallel_tool_replies_are_all_kept():
    feed = [{"role": "user", "content": "go"},
            _openai_assistant(["a", "b"]),
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "tool", "tool_call_id": "b", "content": "rb"}]
    out = P.repair_feed(feed, "openai")
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool"]


def test_openai_orphan_tool_message_is_dropped():
    feed = [{"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "ghost", "content": "ra"}]
    out = P.repair_feed(feed, "openai")
    assert [m["role"] for m in out] == ["user"]


def test_openai_unanswered_call_is_stripped_before_a_plain_user_turn():
    feed = [_openai_assistant(["a"], content="thinking"),
            {"role": "user", "content": "next"}]
    out = P.repair_feed(feed, "openai")
    assert out[0]["role"] == "assistant"
    assert "tool_calls" not in out[0]
    assert out[0]["content"] == "thinking"
    assert out[1] == {"role": "user", "content": "next"}


def test_openai_an_emptied_assistant_message_is_removed():
    feed = [_openai_assistant(["a"], content=""),
            {"role": "user", "content": "next"}]
    out = P.repair_feed(feed, "openai")
    assert [m["role"] for m in out] == ["user"]


def test_openai_partial_batch_keeps_only_the_answered_call():
    feed = [_openai_assistant(["a", "b"]),
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "user", "content": "next"}]
    out = P.repair_feed(feed, "openai")
    declared = [tc["id"] for tc in out[0]["tool_calls"]]
    assert declared == ["a"]
    assert out[1]["role"] == "tool" and out[1]["tool_call_id"] == "a"


def test_openai_trailing_unanswered_calls_are_stripped():
    feed = [{"role": "user", "content": "go"}, _openai_assistant(["a"])]
    out = P.repair_feed(feed, "openai")
    assert [m["role"] for m in out] == ["user"]


def test_openai_repair_does_not_mutate_its_input():
    feed = [_openai_assistant(["a", "b"]),
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "user", "content": "next"}]
    original = copy.deepcopy(feed)
    P.repair_feed(feed, "openai")
    assert feed == original


# ── Anthropic ──────────────────────────────────────────────────────────────────

def test_anthropic_combined_results_are_kept():
    feed = [{"role": "user", "content": "go"},
            _anth_assistant(["a", "b"]),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "ra"},
                {"type": "tool_result", "tool_use_id": "b", "content": "rb"}]}]
    out = P.repair_feed(feed, "anthropic")
    assert [m["role"] for m in out] == ["user", "assistant", "user"]


def test_anthropic_partial_results_strip_the_unanswered_tool_use():
    feed = [_anth_assistant(["a", "b"]),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "ra"}]}]
    out = P.repair_feed(feed, "anthropic")
    uses = [b["id"] for b in out[0]["content"] if b.get("type") == "tool_use"]
    assert uses == ["a"], "the unanswered tool_use must be stripped"
    assert out[-1]["role"] == "user"


def test_anthropic_unanswered_tool_use_before_a_plain_user_turn_is_stripped():
    feed = [_anth_assistant(["a"]),
            {"role": "user", "content": "next"}]
    out = P.repair_feed(feed, "anthropic")
    assert all(b.get("type") != "tool_use" for b in out[0]["content"])
    assert out[-1] == {"role": "user", "content": "next"}


def test_anthropic_orphan_tool_result_is_dropped():
    feed = [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ghost", "content": "x"}]},
            {"role": "user", "content": "next"}]
    out = P.repair_feed(feed, "anthropic")
    assert all(
        not (isinstance(m.get("content"), list) and m["content"]
             and any(isinstance(b, dict) and b.get("type") == "tool_result"
                     for b in m["content"]))
        for m in out)


def test_anthropic_repair_does_not_mutate_its_input():
    feed = [_anth_assistant(["a", "b"]),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "ra"}]}]
    original = copy.deepcopy(feed)
    P.repair_feed(feed, "anthropic")
    assert feed == original


def test_repair_is_idempotent():
    feed = [_openai_assistant(["a", "b"]),
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "user", "content": "next"}]
    once = P.repair_feed(feed, "openai")
    twice = P.repair_feed(once, "openai")
    assert once == twice


# ── Textual tool calls (DeepSeek DSML / Claude-Code markup) ────────────────────

_DSML = (
    "Empty workspace - I'll build this from scratch.\n\n"
    "<\uff5c\uff5cDSML\uff5c\uff5c calls> <\uff5c\uff5cDSML\uff5c\uff5c invoke name=\"update_todo\"> "
    "<\uff5c\uff5cDSML\uff5c\uff5c parameter name=\"todos\" string=\"true\">"
    "[{\"id\": \"1\", \"title\": \"Design UI\", \"status\": \"in_progress\"}]"
    "</\uff5c\uff5cDSML\uff5c\uff5c parameter> </\uff5c\uff5cDSML\uff5c\uff5c invoke> "
    "</\uff5c\uff5cDSML\uff5c\uff5c calls>"
)


def test_a_dsml_text_call_is_parsed_into_a_real_call():
    calls = P.parse_textual_tool_calls(_DSML)
    assert len(calls) == 1
    assert calls[0]["name"] == "update_todo"
    assert calls[0]["args"]["todos"][0]["title"] == "Design UI"


def test_a_claude_code_text_call_is_parsed():
    text = ('<invoke name="write_file">'
            '<parameter name="path">a.html</parameter>'
            '<parameter name="content">hi</parameter></invoke>')
    calls = P.parse_textual_tool_calls(text)
    assert calls[0]["name"] == "write_file"
    assert calls[0]["args"] == {"path": "a.html", "content": "hi"}


def test_plain_prose_is_not_mistaken_for_a_call():
    assert P.parse_textual_tool_calls("Let me use update_todo later.") == []


def test_adopt_promotes_a_text_call_and_clears_the_native_payload():
    result = {"text": _DSML, "tool_calls": [], "tokens": 4,
              "raw_assistant": {"role": "assistant", "content": _DSML}}
    adopted = P.adopt_textual_tool_calls(result)
    assert [c["name"] for c in adopted["tool_calls"]] == ["update_todo"]
    # The raw markup must not be replayed as the assistant turn.
    assert adopted["raw_assistant"] is None
    assert adopted["raw_content"] == []
    # The original is untouched.
    assert result["tool_calls"] == [] and result["raw_assistant"] is not None


def test_adopt_is_a_noop_for_a_structured_call():
    result = {"text": "", "tool_calls": [{"id": "x", "name": "t", "args": {}}],
              "raw_assistant": {"role": "assistant", "tool_calls": []}}
    assert P.adopt_textual_tool_calls(result) is result


def test_adopt_is_a_noop_when_nothing_parses():
    result = {"text": "There is an invoke in the docs, see <invoke name= x>.",
              "tool_calls": [], "raw_assistant": {"role": "assistant"}}
    assert P.adopt_textual_tool_calls(result) is result


# ── Reasoning-channel fallback (blank replies) ─────────────────────────────────

def test_call_openai_falls_back_to_reasoning_content(monkeypatch):
    """An empty `content` with text in `reasoning_content` must not read blank."""
    canned = {"choices": [{"message": {
        "role": "assistant", "content": "",
        "reasoning_content": "The capital of France is Paris."}}],
        "usage": {"total_tokens": 5}}
    monkeypatch.setattr(P, "_http_post", lambda *a, **k: canned)
    prov = {"base_url": "https://x/v1", "api_key": "k",
            "model_id": "deepseek-v4-flash", "format": "openai"}
    r = P.call_openai(prov, [{"role": "user", "content": "capital of France?"}],
                      "sys", use_tools=False)
    assert r["text"] == "The capital of France is Paris."


def test_call_openai_keeps_content_when_present(monkeypatch):
    canned = {"choices": [{"message": {
        "role": "assistant", "content": "Paris",
        "reasoning_content": "long chain of thought"}}],
        "usage": {"total_tokens": 5}}
    monkeypatch.setattr(P, "_http_post", lambda *a, **k: canned)
    prov = {"base_url": "https://x/v1", "api_key": "k",
            "model_id": "m", "format": "openai"}
    r = P.call_openai(prov, [{"role": "user", "content": "?"}], "sys",
                      use_tools=False)
    assert r["text"] == "Paris"


# ── User-Agent gating (AgentRouter blocks unrecognised clients) ────────────────

def _capture_headers(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        P, "_http_post",
        lambda url, headers, payload, timeout=None: (
            seen.update(headers) or
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}))
    return seen


def test_a_provider_without_a_user_agent_gets_the_allowlisted_default(monkeypatch):
    seen = _capture_headers(monkeypatch)
    prov = {"base_url": "https://x/v1", "api_key": "k", "model_id": "m"}
    P.call_openai(prov, [{"role": "user", "content": "hi"}], "sys", use_tools=False)
    assert seen["User-Agent"] == P.DEFAULT_USER_AGENT
    # The old value was rejected outright by AgentRouter with 401.
    assert P.DEFAULT_USER_AGENT != "Agent2/2.0"


def test_a_per_provider_user_agent_overrides_the_default(monkeypatch):
    seen = _capture_headers(monkeypatch)
    prov = {"base_url": "https://x/v1", "api_key": "k", "model_id": "m",
            "user_agent": "my-gateway/9.9"}
    P.call_openai(prov, [{"role": "user", "content": "hi"}], "sys", use_tools=False)
    assert seen["User-Agent"] == "my-gateway/9.9"
