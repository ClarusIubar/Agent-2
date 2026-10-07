# Author: Aarav Shah
# Portfolio: aaravshah1311.is-great.net
# github: github.com/aaravshah1311

"""Provider tool-schema conformance (the `update_project_doc` 400).

The defect this file pins is a whole-app outage caused by ONE omitted key.

An OpenAI-compatible gateway that reads a missing `required` as JSON `null`
answers the entire request with:

    Invalid schema for function 'update_project_doc': null is not of type "array"

so a tool whose parameters are all optional (`update_project_doc`,
`file_capabilities`) — or any connected MCP bridge tool whose `inputSchema`
omits `required` — takes down EVERY custom-provider call, on every turn. The
error names whichever tool comes first in list order, which is why it points at
an unrelated tool and reads like the model's fault when it is ours.

Three invariants keep it fixed:

1. **`required` is always an array**, at the top level and in every nested object
   schema, so `missing` can never be re-read as `null`.
2. **The conformer copies, never mutates.** `agent_tool_schema()` and the MCP
   bridges hand back cached dicts; writing into one would leak a fix made for a
   single provider request into every later surface, the Gemini paths included.
3. **A rejected schema is `schema`-classified.** It is our request, identical on
   every model, so it must never be retried, never fall back, and never cool a
   model — checking it ahead of the transient markers so a stray `5xx`-looking
   token in a trace id cannot turn a deterministic 400 into key rotations.
"""

import copy

import pytest

from agent2.llm import providers as P
from agent2.llm import router as R
from agent2.llm.resilience import classify_error


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_bridges(monkeypatch):
    """Local tools only, so a connected bridge cannot make these tests flaky."""
    monkeypatch.setattr(P, "_burp_tool_schemas", lambda: [])
    monkeypatch.setattr(P, "_mcp_tool_schemas", lambda: [])
    yield
    R.reset_breaker()


def _params(node):
    """The parameters/input_schema object out of either wire shape."""
    if "function" in node:
        return node["function"]["parameters"]
    return node["input_schema"]


# ── 1. `required` is always an array ───────────────────────────────────────────

def test_every_canonical_tool_declares_a_required_array():
    """The source list itself is valid: no tool may omit `required`."""
    missing = [t["name"] for t in P.agent_tool_schema()
               if not isinstance(t["parameters"].get("required"), list)]
    assert missing == [], f"tools without a required array: {missing}"


def test_the_two_reported_tools_carry_an_empty_required_array():
    """The exact schemas from the live error: all-optional, `required` present."""
    by_name = {t["function"]["name"]: t["function"]["parameters"]
               for t in P._openai_tools()}
    assert by_name["update_project_doc"]["required"] == []
    assert by_name["file_capabilities"]["required"] == []


def test_every_openai_tool_conforms():
    for t in P._openai_tools():
        params = t["function"]["parameters"]
        name = t["function"]["name"]
        assert params["type"] == "object", name
        assert isinstance(params["properties"], dict), name
        assert isinstance(params["required"], list), name


def test_every_anthropic_tool_conforms():
    for t in P._anthropic_tools():
        params = t["input_schema"]
        assert params["type"] == "object", t["name"]
        assert isinstance(params["properties"], dict), t["name"]
        assert isinstance(params["required"], list), t["name"]


# ── 2. Third-party schemas are conformed at the render choke point ─────────────

def test_an_mcp_schema_missing_required_is_conformed(monkeypatch):
    """A bridge tool with no `required` must not poison the whole request."""
    fake = {"name": "mcp_thing", "description": "d",
            "parameters": {"type": "object",
                           "properties": {"a": {"type": "object",
                                                "properties": {"b": {"type": "string"}}}}}}
    monkeypatch.setattr(P, "_mcp_tool_schemas", lambda: [fake])

    out = {t["function"]["name"]: t["function"]["parameters"]
           for t in P._openai_tools()}
    params = out["mcp_thing"]
    assert params["required"] == []
    assert params["properties"]["a"]["required"] == [], "nested object not conformed"


def test_conform_recurses_items_and_combinators():
    schema = {"type": "object",
              "properties": {
                  "rows": {"type": "array",
                           "items": {"type": "object",
                                     "properties": {"x": {"type": "integer"}}}},
                  "either": {"anyOf": [{"type": "object",
                                        "properties": {"y": {"type": "string"}}}]},
              }}
    out = P._conform_tool_params(schema, top=True)
    assert out["required"] == []
    assert out["properties"]["rows"]["items"]["required"] == []
    assert out["properties"]["either"]["anyOf"][0]["required"] == []


def test_conform_leaves_a_ref_node_whole():
    """Injecting siblings onto `$ref` is invalid in every dialect — do not."""
    schema = {"type": "object", "properties": {"r": {"$ref": "#/$defs/x"}}}
    out = P._conform_tool_params(schema, top=True)
    assert out["properties"]["r"] == {"$ref": "#/$defs/x"}


def test_conform_never_mutates_its_input():
    schema = {"type": "object", "properties": {"a": {"type": "object",
                                                     "properties": {"b": {}}}}}
    original = copy.deepcopy(schema)
    P._conform_tool_params(schema, top=True)
    assert schema == original, "the conformer wrote into a shared cached schema"


def test_conform_falls_back_for_a_non_dict_top_level():
    assert P._conform_tool_params(None, top=True) == {
        "type": "object", "properties": {}, "required": []}


def test_conform_preserves_a_valid_required_list_and_order():
    schema = {"type": "object",
              "properties": {"a": {}, "b": {}},
              "required": ["b", "a"]}
    out = P._conform_tool_params(schema, top=True)
    assert out["required"] == ["b", "a"]


# ── 3. A rejected schema is deterministic, never a retry/fallback ──────────────

_THE_ERROR = ("Invalid schema for function 'update_project_doc': "
              "null is not of type \"array\" (request_id: "
              "0da8648b-b769-4ff0-bddf-cd75dc34d083)")


def test_the_live_error_classifies_as_schema():
    assert classify_error(_THE_ERROR) == "schema"


def test_a_5xx_looking_token_in_the_trace_id_cannot_make_it_transient():
    """Schema is checked before the transient markers, because it is our request."""
    noisy = _THE_ERROR + " [trace_id=500abc502]"
    assert classify_error(noisy) == "schema"


def test_schema_is_not_a_fallback_kind():
    assert "schema" not in R.FALLBACK_KINDS


def test_no_model_falls_back_for_a_schema_error():
    assert R.next_model("2.5-flash", kind="schema") == ""


def test_a_schema_error_never_trips_the_breaker():
    R.reset_breaker()
    for _ in range(50):
        assert R.note_failure("2.5-flash", "schema") is False
    assert R.cooling("2.5-flash") is False
    assert R.breaker_state()["cooling"] == {}


# ── Classification gaps found in the field database ────────────────────────────

def test_socket_and_dns_failures_are_transient():
    for msg in (
        "[WinError 10054] An existing connection was forcibly closed by the remote host",
        "[WinError 10053] An established connection was aborted",
        "Cannot reach https://x/v1/chat/completions: [Errno 11001] getaddrinfo failed",
        "Temporary failure in name resolution",
    ):
        assert classify_error(msg) == "transient", msg


def test_content_blocked_is_a_safety_failure():
    assert classify_error(
        'HTTP 400 from https://x: {"error":{"code":"content-blocked"}}') == "safety"


def test_a_content_block_wrapped_in_a_500_is_safety_not_transient():
    """Gateways wrap content blocks in 5xx; the numeric code must not win."""
    from agent2.llm.providers import ProviderHTTPError
    exc = ProviderHTTPError(
        500, "https://x",
        '{"error":{"code":"sensitive_words_detected",'
        '"message":"sensitive words detected"}}')
    assert classify_error(exc) == "safety"
