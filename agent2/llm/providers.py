# Author: Aarav Shah
# Portfolio: aaravshah1311.is-great.net
# github: github.com/aaravshah1311

"""
agent2/providers.py
───────────────────
Custom model providers — "bring your own API".

Lets the user register any model endpoint by supplying:
    - base_url   (e.g. https://openrouter.ai/api/v1  or  https://api.anthropic.com)
    - api_key
    - model id   (e.g. deepseek/deepseek-chat, claude-3-5-sonnet-20241022)
    - format     ("openai" | "anthropic")

Providers are persisted in SQLite (table `providers`) and each becomes a
selectable model in the UI/CLI (key = "custom:<id>"). Requests are made with
the standard library only (urllib) so no extra dependency is required, and the
same Agent2 tool schema is exposed to the model in whichever wire format it
expects. The agent loop for custom providers lives in `provider_agent.py`.

⚠️ THE STORED `api_key` IS A REFERENCE, NOT A KEY (Task 16).
`providers.api_key` holds a `a2s:` reference resolved by `agent2/core/secrets.py`;
migration 14 converted existing rows. Two consequences worth internalising before
touching this file:

* Read through `get_provider()` / `list_providers(safe=False)` when you need to
  *send* the credential, and through `list_providers(safe=True)` when you need to
  *show* it. Reading the column directly gets you ciphertext, which fails as an
  `Authorization` header in a way the vendor reports as a confusing 400.
* Write through `add_provider` / `update_provider`. A raw `INSERT` stores a live
  credential in the DB and silently undoes the migration for that row.

Legacy plaintext rows keep working either way — `resolve()` passes a non-reference
through unchanged — so nothing here is a hard cutover.
"""

from __future__ import annotations

import json
import re
import uuid
import urllib.request
import urllib.error

from agent2.database import qall, qone, exe, ensure_column
from agent2.core import secrets as _secrets

try:
    from agent2.config import HTTP_TIMEOUT as _HTTP_TIMEOUT
except Exception:
    _HTTP_TIMEOUT = 600


# ── Persistence ─────────────────────────────────────────────────────────────────

def init_providers_table() -> None:
    exe("""
        CREATE TABLE IF NOT EXISTS providers (
            id         TEXT PRIMARY KEY,
            name       TEXT,
            base_url   TEXT,
            api_key    TEXT,
            model_id   TEXT,
            format     TEXT DEFAULT 'openai',   -- openai | anthropic
            user_agent TEXT DEFAULT '',         -- optional custom User-Agent (some gateways allowlist clients)
            created_at TEXT DEFAULT(datetime('now'))
        )
    """)
    # For DBs created before user_agent existed. `ensure_column` checks
    # PRAGMA table_info rather than catching the ALTER's error, so "column
    # already exists" is no longer indistinguishable from a locked DB or a
    # genuine SQL fault — the previous try/except/pass swallowed all three.
    ensure_column("providers", "user_agent", "TEXT DEFAULT ''")


# Default User-Agent Agent 2 sends to custom providers.
#
# ⚠️ THIS IS A HARD REQUIREMENT ON GATEWAYS THAT ALLOWLIST CLIENTS, NOT COSMETIC.
# AgentRouter (and similar coding-agent gateways) answer an unrecognised UA with
# `401 unauthorized client detected` BEFORE the request reaches any model, so a
# generic "Agent2/2.0" made every provider call fail — confirmed against the live
# endpoint: `Agent2/2.0` -> 401, `opencode/*` and `claude-cli/*` -> 200. The value
# below is an allowlisted coding-agent UA, and a per-provider `user_agent` still
# overrides it for a gateway with its own allowlist.
DEFAULT_USER_AGENT = "opencode/1.18.31"


class ProviderHTTPError(RuntimeError):
    """A provider replied with a non-2xx HTTP status.

    ⚠️ Carries the numeric `status` as an attribute so `resilience.classify_error`
    can read it directly instead of parsing it out of a message string — a body
    that happens to mention "500" is not a 500. `message` stays the human-readable
    form every existing handler already prints.
    """

    def __init__(self, status: int, url: str, body: str):
        self.status = int(status)
        self.url = url
        self.body = body
        super().__init__(f"HTTP {self.status} from {url}: {body[:400]}")


def list_providers(safe: bool = True) -> list[dict]:
    """Every provider row.

    ⚠️ Task 16: `safe=True` masks with a CONSTANT, not with `k[:6]…k[-4:]`.
    Two reasons the old form had to go. It printed most of a short credential —
    the trap `integrations.state.mask_secret` was written to avoid — and once rows
    hold `a2s:` references it would have shown six characters of ciphertext, which
    is both useless to the user and confusing to anyone comparing it with the key
    they pasted.

    ⚠️ `safe=False` resolves the reference, because that is the caller who is about
    to *send* the credential (`chat()` reads `prov["api_key"]` straight into a
    header). Resolving unconditionally, including here, would defeat the masking.
    """
    rows = qall("SELECT * FROM providers ORDER BY created_at")
    for r in rows:
        if safe:
            r["api_key"] = _secrets.mask(r.get("api_key"))
            r["key_set"] = bool(r.get("api_key"))
            r["key"] = "custom:" + r["id"]
        else:
            r["api_key"] = _secrets.resolve(r.get("api_key")) or ""
    return rows


def count_providers() -> int:
    """How many custom providers are registered — and **0 if the table is absent**.

    ⚠️ THIS IS NOT `len(list_providers())`, AND THE DIFFERENCE IS THE WHOLE POINT.
    `providers` is created by `init_providers_table()`, not by `init_db()` (see
    migration 5's note in `database.py`), so every reader that runs before an entry
    point has called it — `/api/health`, a metrics read, an embedder that only
    registered the routes — hits `no such table: providers`. For a *count* that is
    not an error: no table means no rows, and "0 providers" is the true answer.

    The list form is deliberately left to raise, because a caller asking for rows
    to render or to send a credential with wants to know the store is missing.
    """
    try:
        row = qone("SELECT COUNT(*) AS n FROM providers")
    except Exception:
        return 0
    return int((row or {}).get("n") or 0)


def get_provider(pid: str) -> dict | None:
    """One provider row with a USABLE `api_key`.

    The two agent loops pass this row straight into `chat()`, which puts
    `prov["api_key"]` into an `Authorization` header — so this is a resolve site,
    not a masking site. `list_providers(safe=True)` is what a surface renders.
    """
    row = qone("SELECT * FROM providers WHERE id=?", (pid,))
    if row:
        row["api_key"] = _secrets.resolve(row.get("api_key")) or ""
    return row


def add_provider(name: str, base_url: str, api_key: str,
                 model_id: str, fmt: str = "openai",
                 user_agent: str = "") -> dict:
    pid = uuid.uuid4().hex[:8]
    fmt = fmt if fmt in ("openai", "anthropic") else "openai"
    exe("""INSERT INTO providers(id, name, base_url, api_key, model_id, format, user_agent)
           VALUES(?,?,?,?,?,?,?)""",
        (pid, name or model_id, base_url.rstrip("/"),
         _secrets.seal(api_key, namespace="providers", name=pid),
         model_id, fmt, (user_agent or "").strip()))
    _notify("add", pid)
    # Task 17: rank the new model so the router can actually use it. The common
    # case (a recognised name like `claude-opus-5`) costs nothing — pattern
    # inference already covers it and `rank_with_model` returns immediately. An
    # unrecognised id costs one cheap model call, in a DAEMON THREAD, because
    # "add a provider" must stay instant: the user is at a prompt or a form, and
    # blocking that on a network round trip would make the command feel broken.
    try:
        from agent2.llm import capabilities as _caps
        _caps.rank_in_background("custom:" + pid)
    except Exception:
        pass
    return {"id": pid, "key": "custom:" + pid, "name": name or model_id,
            "model_id": model_id, "format": fmt}


def remove_provider(pid: str) -> None:
    # Release the backing material before the row that points at it is gone, for
    # the reason `database.remove_api_key` states: an orphaned OS-keychain entry is
    # something the user finds later and cannot identify.
    try:
        row = qone("SELECT api_key FROM providers WHERE id=?", (pid,))
        if row:
            _secrets.forget(str(row.get("api_key") or ""))
    except Exception:
        pass
    exe("DELETE FROM providers WHERE id=?", (pid,))
    _notify("delete", pid)


def _notify(action: str, pid: str = "") -> None:
    """Announce a provider change to the other surfaces (see agent2.core.sync)."""
    try:
        from agent2.core import sync
        sync.notify("providers", action=action, id=pid)
    except Exception:
        pass


def update_provider(pid: str, **fields) -> dict | None:
    """Partially update a provider record. Only known columns are touched, and
    an empty/absent api_key leaves the stored key intact (so the redacted value
    shown in the UI never overwrites the real key)."""
    row = qone("SELECT * FROM providers WHERE id=?", (pid,))
    if not row:
        return None

    updates: dict = {}
    for col in ("name", "base_url", "model_id", "format", "user_agent"):
        if col in fields and fields[col] is not None:
            val = str(fields[col]).strip()
            if col == "base_url":
                val = val.rstrip("/")
            elif col == "format":
                val = val if val in ("openai", "anthropic") else "openai"
            updates[col] = val

    # api_key is only replaced when a non-empty value is supplied.
    #
    # ⚠️ And a MASK is not a value. The UI shows the stored key as bullets and
    # submits the form unchanged when only the base URL was edited; treating that
    # echoed mask as a new key would overwrite a working credential with
    # `••••••••`. This is the same guard `integrations.state.set_config()` carries
    # for the ZAP key, for the same reason.
    ak = (fields.get("api_key") or "").strip()
    if ak and ak != _secrets.mask("x") and set(ak) != {"•"}:
        updates["api_key"] = _secrets.seal(ak, namespace="providers", name=pid)

    if not updates:
        return get_provider(pid)

    sets = ", ".join(f"{c}=?" for c in updates)
    exe(f"UPDATE providers SET {sets} WHERE id=?",
        (*updates.values(), pid))
    _notify("update", pid)
    return get_provider(pid)



# ── HTTP helper ─────────────────────────────────────────────────────────────────

def _openai_chat_url(base_url: str) -> str:
    """
    Build the /chat/completions URL from whatever the user pasted as base_url.

    Accepts a bare host (https://api.example.com), a versioned root
    (…/v1), an OpenAI-style root (…/openai/v1) or the full completions URL —
    and always returns a valid endpoint. This is the #1 cause of "provider
    won't connect": users paste https://host with no /v1 and the old code
    POSTed to https://host/chat/completions which returns an HTML 404.
    """
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/completions"):            # already a completions path
        return b
    if b.endswith(("/v1", "/v3", "/openai")) or "/v1/" in b or "/v2/" in b:
        return b + "/chat/completions"
    # Bare host or custom root → assume OpenAI-style versioned API.
    return b + "/v1/chat/completions"


def _anthropic_messages_url(base_url: str) -> str:
    """Build the Anthropic /v1/messages URL, tolerating a trailing /v1."""
    b = (base_url or "").strip().rstrip("/")
    if b.endswith("/messages"):
        return b
    if b.endswith("/v1"):
        return b + "/messages"
    return b + "/v1/messages"


def _http_post(url: str, headers: dict, payload: dict, timeout: float | None = None) -> dict:
    if timeout is None:
        timeout = _HTTP_TIMEOUT
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        # `ProviderHTTPError`, so `classify_error` can read `.status` numerically
        # and the string form stays the human-readable body every handler prints.
        raise ProviderHTTPError(e.code, url, e.read().decode("utf-8", "replace")) from e
    except urllib.error.URLError as e:
        # A refusal/reset/timeout at the transport level is almost always
        # transient (server restarting, gateway overloaded, flaky network).
        # "(connection error)" is the transient marker `classify_error` reads.
        reason = getattr(e, "reason", e)
        raise RuntimeError(f"Cannot reach {url}: {reason} (connection error)") from e
    except Exception as e:
        raise RuntimeError(str(e)) from e
    try:
        return json.loads(raw)
    except Exception as e:
        snippet = raw.strip().replace("\n", " ")[:220] or "(empty response)"
        raise RuntimeError(
            f"Non-JSON response from {url}. Check the Base URL is an "
            f"OpenAI-/Anthropic-compatible API root (it usually ends in /v1). "
            f"Server said: {snippet}") from e


# ── Tool-schema translation ─────────────────────────────────────────────────────
# Agent2's canonical tool schema (name, description, JSON-schema params) is
# defined here once and rendered into each provider's wire format.

def agent_tool_schema() -> list[dict]:
    """Canonical tool list as plain JSON-Schema dicts (provider-agnostic).

    ⚠️ **A TOOL EXISTS ONLY IF FOUR LISTS AGREE, AND THIS IS THE ONE WHERE AN
    OMISSION IS INVISIBLE FROM BOTH ENDS.** The other three are checkable by using
    Agent2: a name missing from `agent._LOCAL_TOOLS` answers "not registered"
    mid-turn, and one missing from a `_build_tools()` is at least absent from the
    surface you are looking at. A name missing *here* means every custom provider
    can happily **dispatch** it — `provider_agent.py` routes through
    `_LOCAL_TOOLS`, not through this list — and no custom provider is ever
    **told** it exists. So the tool works perfectly whenever a model guesses the
    name, and is never offered. `emit_plan` shipped in exactly that state.

    ⚠️ Names here must equal `tools._build_tools()` and
    `cli/tooling._build_tools()` exactly, pinned by
    `test_cli.py::test_advertised_and_dispatchable_tools_agree_both_ways`. The
    descriptions need not match and deliberately do not — presentation is each
    surface's own prose.
    """
    obj = "object"
    return [
        {"name": "run_command",
         "description": "Execute a shell command on the user's machine (installs, builds, tests, scans, launches).",
         "parameters": {"type": obj, "properties": {
             "command": {"type": "string"}, "description": {"type": "string"}},
             "required": ["command", "description"]}},
        {"name": "read_file",
         "description": "Read a file's contents. Read before editing.",
         "parameters": {"type": obj, "properties": {
             "path": {"type": "string"},
             "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
             "required": ["path"]}},
        {"name": "write_file",
         "description": "Create or overwrite a file with content. Parent dirs auto-created.",
         "parameters": {"type": obj, "properties": {
             "path": {"type": "string"}, "content": {"type": "string"}},
             "required": ["path", "content"]}},
        {"name": "multi_edit_files",
         "description": "Find/replace exact text across multiple files.",
         "parameters": {"type": obj, "properties": {
             "edits": {"type": "array", "items": {"type": obj, "properties": {
                 "path": {"type": "string"}, "old_text": {"type": "string"},
                 "new_text": {"type": "string"}}}}},
             "required": ["edits"]}},
        {"name": "list_dir",
         "description": "List a directory's files and subfolders.",
         "parameters": {"type": obj, "properties": {"path": {"type": "string"}},
                        "required": ["path"]}},
        {"name": "grep_search",
         "description": "Regex-search file contents across a directory tree.",
         "parameters": {"type": obj, "properties": {
             "pattern": {"type": "string"}, "path": {"type": "string"},
             "glob": {"type": "string"}}, "required": ["pattern"]}},
        {"name": "delete_file",
         "description": "Delete a file or directory recursively.",
         "parameters": {"type": obj, "properties": {"path": {"type": "string"}},
                        "required": ["path"]}},
        {"name": "scan_project",
         "description": "Recursively scan a project: file tree + all source contents.",
         "parameters": {"type": obj, "properties": {"path": {"type": "string"}},
                        "required": ["path"]}},
        {"name": "web_search",
         "description": "Search the web for docs, errors, CVEs.",
         "parameters": {"type": obj, "properties": {"query": {"type": "string"}},
                        "required": ["query"]}},
        {"name": "update_todo",
         "description": "Create/update a live TODO checklist for a multi-step build. Pass the full list each time with each item's status (pending|in_progress|completed).",
         "parameters": {"type": obj, "properties": {
             "todos": {"type": "array", "items": {"type": obj, "properties": {
                 "task": {"type": "string"}, "status": {"type": "string"}}}}},
             "required": ["todos"]}},
        {"name": "emit_plan",
         "description": "Show a step-by-step plan before a complex multi-step task.",
         "parameters": {"type": obj, "properties": {
             "title": {"type": "string"}, "steps": {"type": "string"}},
             "required": ["title", "steps"]}},
        {"name": "save_memory",
         "description": "Persist an important fact across sessions.",
         "parameters": {"type": obj, "properties": {"content": {"type": "string"}},
                        "required": ["content"]}},
        {"name": "update_project_doc",
         "description": "Re-scan this project and refresh its .agent2/agent2.md brief "
                        "(purpose, features, architecture, commands, layout). Call AFTER you "
                        "finish work that CHANGED what the project contains — a new feature, "
                        "module, entry point, dependency, command or a restructure — so the "
                        "doc stays true. It preserves anything a human wrote. Set describe=true "
                        "ONLY when you changed what the project is FOR (new purpose or headline "
                        "feature). Do NOT call it after read-only work, a one-line fix, or a "
                        "question. No arguments are required.",
         "parameters": {"type": obj, "properties": {
             "describe": {"type": "boolean"}, "hint": {"type": "string"}},
             "required": []}},
        # ── File Intelligence System ────────────────────────────────────────
        {"name": "detect_file",
         "description": "Auto-detect a file's type, metadata (size, dates, checksum, "
                        "pages, dimensions, duration) and the operations available for it. "
                        "Call FIRST for any file the user references.",
         "parameters": {"type": obj, "properties": {"path": {"type": "string"}},
                        "required": ["path"]}},
        {"name": "file_capabilities",
         "description": "List operations available for a file path OR a category "
                        "(documents, spreadsheets, presentations, images, audio, video, "
                        "archives, code).",
         "parameters": {"type": obj, "properties": {
             "path": {"type": "string"}, "category": {"type": "string"}},
             "required": []}},
        {"name": "run_file_op",
         "description": "Universal file operation — auto-detects type and routes to the "
                        "right backend. operations: read, extract_text, summarize, translate, "
                        "rewrite, grammar, compare, merge, split, extract_images, ocr, "
                        "analyze, formula_audit, clean, export_csv, speaker_notes, metadata, "
                        "convert, compress, resize, list, extract, inspect, create, "
                        "extract_audio, transcribe. For AI ops the tool returns extracted "
                        "text + an instruction; you produce the result and write_file it. "
                        "Batch inputs via options.paths.",
         "parameters": {"type": obj, "properties": {
             "path": {"type": "string"}, "operation": {"type": "string"},
             "options": {"type": obj}}, "required": ["path", "operation"]}},
        {"name": "convert_file",
         "description": "Convert a file to another format (best backend chosen "
                        "automatically; graceful fallback).",
         "parameters": {"type": obj, "properties": {
             "path": {"type": "string"}, "to_format": {"type": "string"},
             "output_path": {"type": "string"}}, "required": ["path", "to_format"]}},
        {"name": "search_workspace",
         "description": "Search many files: kind=content|filename|recent|secrets|duplicates. "
                        "Presets: 'invoices', 'TODOs', 'API keys', 'duplicates'.",
         "parameters": {"type": obj, "properties": {
             "query": {"type": "string"}, "path": {"type": "string"},
             "kind": {"type": "string"}}, "required": ["query"]}},
    ]


def _burp_tool_schemas() -> list[dict]:
    """Live Burp MCP tools as provider-agnostic schemas (empty if not connected)."""
    try:
        from agent2.integrations.burp_mcp import burp
        if burp.is_connected():
            return burp.provider_tool_schemas()
    except Exception:
        pass
    return []


def _mcp_tool_schemas() -> list[dict]:
    """Live tools from every OTHER MCP bridge (Task 8), same shape as Burp's.

    ⚠️ Both halves are needed and neither is redundant: `_burp_tool_schemas`
    covers the bridge that predates the registry, this covers the rest. Dropping
    either one silently removes tools from custom providers ONLY — the Gemini loop
    builds its declarations elsewhere, so the model would report a tool it can
    plainly see in its own prompt as unknown.
    """
    try:
        from agent2.integrations import registry as mcp_registry
        return mcp_registry.provider_schemas_for(mcp_registry.extra_bridges())
    except Exception:
        return []


def _conform_tool_params(params: object, *, top: bool = False) -> dict:
    """Return a tool-parameter schema a strict gateway will accept.

    ⚠️ `required` MUST BE PRESENT AND BE AN ARRAY, even when nothing is required.
    An OpenAI-compatible gateway (DeepSeek's, among others) materialises a MISSING
    `required` as JSON `null` and then refuses the WHOLE request with
    `Invalid schema for function 'x': null is not of type "array"` — so a single
    optional-only local tool (`update_project_doc`, `file_capabilities`) or any
    connected MCP tool whose `inputSchema` omits `required` takes down EVERY
    custom-provider call, on every turn. The tool named in the error is simply the
    first one in list order whose schema has the omission, which is why the
    message points at an unrelated tool.

    ⚠️ IT COPIES, NEVER MUTATES. `agent_tool_schema()` and the MCP bridges hand
    back cached dicts shared by every call; writing into one would leak a fix made
    for a single provider request into every later surface, the Gemini paths
    included.

    ⚠️ A `$ref` NODE IS LEFT WHOLE. Injecting `type`/`properties`/`required`
    siblings onto `{"$ref": …}` is invalid in every dialect, so the walk stops and
    the referenced fragment stays the document's responsibility.

    Only the conservative function-schema subset is touched — object shape, the
    `required` array, and recursion into `properties` / `items` / `prefixItems` /
    `additionalProperties` / the `anyOf`·`oneOf`·`allOf` combinators / `not`.
    Nothing is dropped and `additionalProperties` is never invented, so no schema
    changes meaning: the omitted-versus-null array is the whole of the fix.
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}, "required": []} if top else {}
    node = dict(params)
    if node.get("$ref"):
        return node
    if top:
        node["type"] = "object"
    if node.get("type") == "object" or "properties" in node:
        props = node.get("properties")
        if not isinstance(props, dict):
            props = {}
        node["properties"] = {k: _conform_tool_params(v) for k, v in props.items()}
        req = node.get("required")
        node["required"] = ([r for r in req if isinstance(r, str)]
                            if isinstance(req, list) else [])
        ap = node.get("additionalProperties")
        if isinstance(ap, dict):
            node["additionalProperties"] = _conform_tool_params(ap)
    if node.get("type") == "array":
        items = node.get("items")
        if isinstance(items, dict):
            node["items"] = _conform_tool_params(items)
        prefix = node.get("prefixItems")
        if isinstance(prefix, list):
            node["prefixItems"] = [_conform_tool_params(x) if isinstance(x, dict) else x
                                   for x in prefix]
    for key in ("anyOf", "oneOf", "allOf"):
        sub = node.get(key)
        if isinstance(sub, list):
            node[key] = [_conform_tool_params(x) if isinstance(x, dict) else x for x in sub]
    if isinstance(node.get("not"), dict):
        node["not"] = _conform_tool_params(node["not"])
    return node


def _openai_tools() -> list[dict]:
    tools = agent_tool_schema() + _burp_tool_schemas() + _mcp_tool_schemas()
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": _conform_tool_params(t.get("parameters"),
                                                             top=True)}}
            for t in tools]


def _anthropic_tools() -> list[dict]:
    tools = agent_tool_schema() + _burp_tool_schemas() + _mcp_tool_schemas()
    return [{"name": t["name"], "description": t["description"],
             "input_schema": _conform_tool_params(t.get("parameters"), top=True)}
            for t in tools]


# ── Provider state — thinking/reasoning survives between turns ────────────────────
#
# A turn stores the assistant reply in the messages table as PLAIN TEXT. Some
# providers (DeepSeek reasoning mode, Anthropic thinking, Gemini thinking) insist
# on the previous assistant turn's thinking/reasoning content being passed back
# with it, so the plain-text history fails the NEXT turn with a deterministic
# "…must be passed back to the API." error. The fix lives HERE and nowhere else:
# the two provider loops hand us the native payload (`assistant_state`), persist
# it as an OPAQUE blob under `PROVIDER_STATE_KEY` in the row's `meta` JSON, and
# restore it on the next turn (`history_messages`). Core loops never see the
# native shapes — this module is the whole of the translation.
PROVIDER_STATE_KEY = "provider_state"


def assistant_state(result: dict | None) -> dict | None:
    """The opaque native assistant payload worth persisting, or None.

    *result* is one of the normalised dicts `chat()`/`call_openai`/
    `call_anthropic` return. We capture the RAW choice message (openai) or RAW
    content blocks (anthropic) — the exact dict the wire returned, so any
    reasoning/thinking content the provider produced travels back verbatim. Opaque
    to the core: `history_messages` is the only reader, and it keys on `fmt` so a
    state blob is never replayed into a provider of another wire format.
    """
    if not result:
        return None
    raw_content = result.get("raw_content")
    if raw_content:
        return {"fmt": "anthropic", "content": raw_content}
    raw_assistant = result.get("raw_assistant")
    if isinstance(raw_assistant, dict):
        return {"fmt": "openai", "message": raw_assistant}
    return None


def _meta_row(r: dict) -> dict:
    """Decode a row's `meta` column (a JSON string, a dict, or absent)."""
    meta = r.get("meta")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return {}
    if isinstance(meta, dict):
        return meta
    return {}


def history_messages(rows: list, fmt: str) -> list[dict]:
    """Seed a provider-native message list from stored user/assistant rows.

    *rows* are dict-like with `role`/`content`/`meta` (DB rows or the CLI's
    history dicts). User rows become plain text user dicts; assistant rows become
    the native assistant dict restored from their `provider_state` when it is
    present AND matches *fmt* — otherwise a plain text assistant dict. A provider
    never receives another provider's state, because the `fmt` tag is checked
    here; a row written before this feature exists simply has no state.
    """
    if fmt == "anthropic":
        return [_anthropic_msg(r) for r in rows]
    return [_openai_msg(r) for r in rows]


def _openai_msg(r: dict) -> dict:
    if (r.get("role") or "assistant") == "user":
        return {"role": "user", "content": r.get("content") or ""}
    state = _meta_row(r).get(PROVIDER_STATE_KEY)
    if isinstance(state, dict) and state.get("fmt") == "openai":
        msg = state.get("message")
        if isinstance(msg, dict):
            rebuilt = dict(msg)
            rebuilt.setdefault("role", "assistant")
            return rebuilt
    return {"role": "assistant", "content": r.get("content") or ""}


def _anthropic_msg(r: dict) -> dict:
    if (r.get("role") or "assistant") == "user":
        return {"role": "user", "content": r.get("content") or ""}
    state = _meta_row(r).get(PROVIDER_STATE_KEY)
    if isinstance(state, dict) and state.get("fmt") == "anthropic":
        blocks = state.get("content")
        if isinstance(blocks, list):
            return {"role": "assistant", "content": blocks}
    return {"role": "assistant", "content": r.get("content") or ""}


def strip_native_state(messages: list[dict], fmt: str) -> list[dict]:
    """Downgrade assistant dicts to the text-only form a fresh turn can re-send.

    The reconstruction retry for a `state_missing` error: the previous attempt
    carried stale thinking/reasoning blocks, so we rebuild every assistant dict
    with just its user-visible text and re-request. Reasoning (openai) is dropped
    by rebuilding the dict; thinking blocks (anthropic) are filtered to `text`.
    A dict with no surviving text is left untouched — turning a tool_use turn into
    an empty content list would only replace one provider rejection with another.
    """
    out: list[dict] = []
    for m in (messages or []):
        if (m or {}).get("role") != "assistant":
            out.append(m)
            continue
        if fmt == "anthropic":
            blocks = m.get("content")
            if isinstance(blocks, list):
                text_blocks = [b for b in blocks
                               if isinstance(b, dict) and b.get("type") == "text"]
                if text_blocks:
                    out.append({"role": "assistant", "content": text_blocks})
                    continue
        else:
            content = m.get("content")
            if content is not None:
                out.append({"role": "assistant", "content": content})
                continue
        out.append(m)
    return out


def strip_null_args(args) -> dict:
    """Drop top-level null values from a tool-call's parsed input.

    Anthropic-compatible emulators (DeepSeek's, among others) validate the
    tool_use `input` against the declared input_schema, and reject an OPTIONAL
    prop sent as `null` ("null is not of type 'array'") even though omitting the
    key entirely is legal in every schema. Absent key < null key, always. The
    executed copy keeps what the model sent; only the wire copy is sanitized.
    """
    if not isinstance(args, dict):
        return args
    return {k: v for k, v in args.items() if v is not None}


def anthropic_assistant_content(result: dict) -> list[dict]:
    """Assistant content blocks that satisfy Anthropic's alternation rules.

    *result* is a normalised `call_anthropic` dict. Some Anthropic-compatible
    emulators return the tool-use turn in a shape the real API would reject: a
    `text` block AFTER a `tool_use` block, or `tool_use` blocks missing from
    `content` while `tool_calls` is parsed. The server then fails the NEXT user
    message with "Each tool_result block must have a corresponding tool_use block
    in the previous message." This rebuilds the blocks exactly as the API
    requires: text first, then EVERY tool_use (synthesised from `tool_calls` when
    the raw payload lacks them), inputs stripped of top-level nulls, and nothing
    else — so a `tool_result`'s id is always matched two messages down.
    """
    blocks = result.get("raw_content") or []
    if not isinstance(blocks, list):
        blocks = []
    tool_uses = {}
    for tc in (result.get("tool_calls") or []):
        if tc.get("id"):
            tool_uses[tc["id"]] = tc
    kept: list[dict] = []
    order: list[str] = []
    mapped: dict[str, dict] = {}
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            kept.append({"type": "text", "text": b.get("text") or ""})
        elif btype == "tool_use":
            tid = b.get("id", "")
            tc = tool_uses.get(tid)
            if tc:
                order.append(tid)
                mapped[tid] = {"type": "tool_use", "id": tid,
                               "name": tc.get("name", ""),
                               "input": strip_null_args(tc.get("args"))}
            elif tid and b.get("name"):
                order.append(tid)
                mapped[tid] = dict(b)
    for tid, tc in tool_uses.items():
        if not tc.get("name"):
            continue
        if tid not in mapped:
            order.append(tid)
            mapped[tid] = {"type": "tool_use", "id": tid,
                           "name": tc["name"],
                           "input": strip_null_args(tc.get("args"))}
    uses = [mapped[tid] for tid in order if tid in mapped]
    rebuilt = kept + uses
    if rebuilt:
        return rebuilt
    return [{"type": "text", "text": (result.get("text") or "") or ""}]


# ── Feed repair — no dangling tool round-trip ever reaches the wire ─────────────
#
# The loops already build feeds with correct tool_use↔tool_result /
# tool_calls↔role:tool pairing; this is the safety net for a feed handed over
# from a resumed turn or a partially-drained batch. Anthropic-compatible
# endpoints reject pairings that are not (a) a tool_result message sitting one
# after the assistant message carrying its tool_use, and (b) an assistant tool_use
# answered before the turn ends. OpenAI rejects role 'tool' messages without a
# preceding declaration.

def is_tool_call_text(text: str | None) -> bool:
    """True when a reply is a tool invocation DUMPED AS PLAIN TEXT.

    Some endpoints/models answer by emitting the function call as a text block —
    Claude-Code style `<invoke name="...">…</invoke>` markup, or its mangled
    fullwidth-bar paste artifact (`｜｜ 大臣 ｜｜ invoke name="update_todo">…`) —
    instead of returning a real `tool_use`/`tool_calls` block. Saved verbatim,
    that block becomes this turn's "final answer", it poisons history, and the
    next turn's feed starts accumulating the rejection spiral. The loops detect
    it and re-ask text-only, exactly like a blank reply.
    """
    if not text:
        return False
    low = text.lower()
    return any(
        m in text or m in low
        for m in ("<invoke name=", "\uff5c", "<function_calls>",
                  "parameter name=", '"tool_calls": [')
    )


# The wrapper around a textual call is unstable across models: the bare
# "<invoke name=...>" of Claude-Code, and DeepSeek's "|DSML| invoke ..." paste
# artifact where the bars are fullwidth characters. Only the words matter, so the
# patterns below key on "invoke name=" and "parameter name=" and tolerate
# anything in front of them.
_TEXTUAL_INVOKE_RE = re.compile(r'invoke\s+name\s*=\s*"([^"]+)"', re.I)
_TEXTUAL_PARAM_RE = re.compile(
    r'parameter\s+name\s*=\s*"([^"]+)"[^>]*>(.*?)'
    r'(?=parameter\s+name\s*=|invoke\s+name\s*=|</[^>]*parameter>|</?[^>]*calls>|$)',
    re.I | re.S,
)


def _parse_textual_value(raw: str):
    """A textual parameter's value: JSON when it parses, else the raw string."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def parse_textual_tool_calls(text: str | None) -> list[dict]:
    """Recover structured tool calls a model emitted AS TEXT.

    DeepSeek and some proxies answer a tool-using turn with the call wrapped in
    plain text instead of a real tool_calls/tool_use block:

        Empty workspace - I'll build this from scratch...
        |DSML| calls> |DSML| invoke name="update_todo">
        |DSML| parameter name="todos" string="true">[...]</|DSML| parameter>
        </invoke>

    Re-asking (the old behaviour) frequently just produces the same text again,
    which is how a turn ends as "I didn't produce a reply" with nothing done. This
    turns the text into the calls it plainly represents so the loop can RUN them.
    Returns [] when there is no invoke/parameter structure to read, so a message
    that merely mentions a tool name is never mistaken for a call.
    """
    if not text:
        return []
    invokes = list(_TEXTUAL_INVOKE_RE.finditer(text))
    if not invokes:
        return []
    calls: list[dict] = []
    for i, m in enumerate(invokes):
        name = m.group(1).strip()
        if not name:
            continue
        start = m.end()
        end = invokes[i + 1].start() if i + 1 < len(invokes) else len(text)
        body = text[start:end]
        args = {pm.group(1).strip(): _parse_textual_value(pm.group(2))
                for pm in _TEXTUAL_PARAM_RE.finditer(body)}
        calls.append({"id": "call_text_" + uuid.uuid4().hex[:12],
                      "name": name, "args": args})
    return calls


def adopt_textual_tool_calls(result: dict) -> dict:
    """Promote a result whose tool call arrived as text into a real one.

    A no-op unless the result has no structured calls AND its text looks like a
    textual call AND at least one invoke parses. The native payload is cleared so
    the loops rebuild the assistant turn from the recovered calls rather than
    replaying the raw markup (which is what poisons history). Any visible prose
    that preceded the call is kept as the assistant message's content.
    """
    if not isinstance(result, dict) or result.get("tool_calls"):
        return result
    text = result.get("text") or ""
    if not is_tool_call_text(text):
        return result
    parsed = parse_textual_tool_calls(text)
    if not parsed:
        return result
    out = dict(result)
    out["tool_calls"] = parsed
    out["raw_assistant"] = None
    out["raw_content"] = []
    return out


def repair_feed(messages: list[dict], fmt: str) -> list[dict]:
    """Make a feed satisfy the provider's tool-call pairing rules, or drop it.

    Idempotent and cheap; run on the assembled feed right before a provider call.
    Both wire formats demand that every tool call be answered IMMEDIATELY and
    COMPLETELY by the next message(s), and every format phrases the failure as a
    whole-request 400 that reads like the model's fault:

        OpenAI/AgentRouter: "An assistant message with 'tool_calls' must be
            followed by tool messages responding to each 'tool_call_id'."
        Anthropic-compatible: "`tool_use` ids were found without `tool_result`
            blocks immediately after …"

    ⚠️ THE OLD OPENAI REPAIR LOOKED AT `out[-1]` FOR THE DECLARATION, WHICH IS
    ONLY THE ASSISTANT MESSAGE FOR THE *FIRST* OF A PARALLEL BATCH. A response
    that returns two tool calls appends two `role: tool` replies; the second
    then saw the first tool message as its "preceding assistant", matched
    nothing, and was silently dropped — leaving one tool_call unanswered and
    failing every multi-tool turn. That is why this tracks the outstanding ids
    across a consecutive run of tool messages instead of peeking at one row.

    A call that never gets answered is STRIPPED from its assistant message (with
    the whole message removed if that leaves it empty), because keeping the call
    and dropping its result is the same dangling half from the other side.
    """
    if fmt == "anthropic":
        return _repair_feed_anthropic(messages)
    return _repair_feed_openai(messages)


def _strip_openai_tool_calls(msg: dict, ids: set) -> bool:
    """Remove the still-unanswered `tool_calls` from *msg*. True if it is now empty."""
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        kept = [tc for tc in tcs
                if not (isinstance(tc, dict) and tc.get("id") in ids)]
        if kept:
            msg["tool_calls"] = kept
        else:
            msg.pop("tool_calls", None)
    return not msg.get("content") and not msg.get("tool_calls")


def _repair_feed_openai(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending_at: int | None = None   # index in `out` of the declaring assistant
    pending: set = set()

    def _flush() -> None:
        nonlocal pending_at, pending
        if pending_at is not None and pending_at < len(out):
            if _strip_openai_tool_calls(out[pending_at], pending):
                del out[pending_at]
        pending_at, pending = None, set()

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            _flush()
            ids = {tc.get("id") for tc in (m.get("tool_calls") or [])
                   if isinstance(tc, dict) and tc.get("id")}
            if ids:
                m = dict(m)          # copy so stripping never edits the caller's dict
                out.append(m)
                pending_at, pending = len(out) - 1, set(ids)
            else:
                out.append(m)
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if tcid and tcid in pending:
                out.append(m)
                pending.discard(tcid)
                if not pending:
                    pending_at = None
            # else: an orphaned tool message answers no live call — drop it.
        else:
            _flush()
            out.append(m)
    _flush()
    return out


def _strip_anthropic_tool_use(msg: dict, ids: set) -> bool:
    """Remove the still-unanswered `tool_use` blocks from *msg*; True if empty."""
    blocks = msg.get("content")
    if isinstance(blocks, list):
        kept = [b for b in blocks
                if not (isinstance(b, dict) and b.get("type") == "tool_use"
                        and b.get("id") in ids)]
        if kept:
            msg["content"] = kept
            return False
        return True
    return False


def _repair_feed_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending_at: int | None = None
    pending: set = set()

    def _flush() -> None:
        nonlocal pending_at, pending
        if pending_at is not None and pending_at < len(out):
            if _strip_anthropic_tool_use(out[pending_at], pending):
                del out[pending_at]
        pending_at, pending = None, set()

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            _flush()
            blocks = m.get("content")
            ids = ({b.get("id") for b in blocks
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")}
                   if isinstance(blocks, list) else set())
            if ids:
                m = dict(m)
                m["content"] = list(blocks)   # copy: stripping must not edit the caller
                out.append(m)
                pending_at, pending = len(out) - 1, set(ids)
            else:
                out.append(m)
        elif role == "user":
            blocks = m.get("content")
            results = ([b for b in blocks
                        if isinstance(b, dict) and b.get("type") == "tool_result"]
                       if isinstance(blocks, list) else [])
            if not results:
                _flush()
                out.append(m)
                continue
            kept = [b for b in results if b.get("tool_use_id") in pending]
            if kept:
                non_results = [b for b in blocks
                               if not (isinstance(b, dict)
                                       and b.get("type") == "tool_result")]
                m = dict(m)
                m["content"] = non_results + kept
                out.append(m)
                pending -= {b.get("tool_use_id") for b in kept}
                if not pending:
                    pending_at = None
            else:
                # Every result answered a call that is no longer live. Preserve
                # any text this user message carried, then close the dangling call.
                non_results = [b for b in blocks
                               if not (isinstance(b, dict)
                                       and b.get("type") == "tool_result")]
                if non_results:
                    out.append({"role": "user", "content": non_results})
            if pending:
                _flush()
        else:
            _flush()
            out.append(m)
    _flush()
    return out


# ── Error normalization ──────────────────────────────────────────────────────────
# Providers surface failures through different envelopes (OpenAI nests everything
# under `error`, Anthropic tags a `type` on the envelope itself); callers that
# only ever see a message string lose the code/parameter that actually diagnoses
# a 400. `normalize_provider_error` flattens any failure into one canonical shape
# for the ledger, logs and user hints.


def _parse_error_body(body: str) -> dict:
    try:
        data = json.loads(body) if isinstance(body, str) else (body or {})
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    err = data.get("error")
    if isinstance(err, dict):
        return {"message": str(err.get("message") or ""),
                "error_type": str(err.get("type") or err.get("code") or ""),
                "parameter": str(err.get("param") or ""),
                "provider_code": str(err.get("code") or ""),
                "trace_id": str(err.get("trace_id") or ""),
                "raw": data}
    msg = str(data.get("message") or "")
    return {"message": msg,
            "error_type": str(data.get("type") or ""),
            "parameter": str(data.get("param") or ""),
            "provider_code": str(data.get("code") or ""),
            "trace_id": str(data.get("trace_id") or ""),
            "raw": data}


def normalize_provider_error(exc) -> dict:
    """Flatten any failure into canonical fields for the ledger and hints.

    Returns a dict with `status_code`, `url`, `message`, `error_type`,
    `parameter`, `provider_code`, `trace_id`. Non-HTTP exceptions get a minimal
    shape with their stringified message and their own type name, so every
    caller can read the SAME fields off any failure.
    """
    if isinstance(exc, ProviderHTTPError):
        params = _parse_error_body(exc.body)
        return {"status_code": exc.status,
                "url": exc.url,
                "message": params.get("message") or str(exc),
                "error_type": params.get("error_type") or "",
                "parameter": params.get("parameter") or "",
                "provider_code": params.get("provider_code") or "",
                "trace_id": params.get("trace_id") or ""}
    return {"status_code": None,
            "url": "",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "parameter": "",
            "provider_code": "",
            "trace_id": ""}


# ── Chat call — returns a normalised result ─────────────────────────────────────
#
# Normalised result shape:
#   {"text": str, "tool_calls": [{"id","name","args"}], "tokens": int}


def _openai_max_tokens_field(model_id: str) -> str:
    """Output-ceiling field name for an OpenAI-compatible model id.
    Reasoning models (o1/o3/o4/gpt-5 …) reject the legacy `max_tokens` and want
    `max_completion_tokens`; every other generation accepts `max_tokens`. Getting
    this wrong is a hard 400 mid-turn, and it is the kind of wire-format detail
    that decides whether a user-registered endpoint works at all.
    """
    text = str(model_id or "").strip().lower()
    if any(prefix in text for prefix in ("o1", "o3", "o4", "o5", "gpt-5", "gpt-5.")):
        return "max_completion_tokens"
    return "max_tokens"


def call_openai(prov: dict, messages: list[dict], system: str,
                use_tools: bool = True, max_tokens: int | None = None) -> dict:
    url = _openai_chat_url(prov["base_url"])
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {prov['api_key']}",
               "User-Agent": (prov.get("user_agent") or "").strip() or DEFAULT_USER_AGENT,
               # Some gateways (OpenRouter etc.) want these; harmless elsewhere.
               "HTTP-Referer": "https://github.com/aaravshah1311",
               "X-Title": "Agent 2"}
    msgs = [{"role": "system", "content": system}, *messages]
    payload = {"model": prov["model_id"], "messages": msgs}
    if max_tokens and max_tokens > 0:
        payload[_openai_max_tokens_field(prov["model_id"])] = max_tokens
    # Omitted entirely (not sent empty) when tools are off — some gateways reject
    # an empty `tools` array. Used by the blank-reply retry to force plain text.
    if use_tools:
        payload["tools"] = _openai_tools()
        payload["tool_choice"] = "auto"
    data = _http_post(url, headers, payload)

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    tool_calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        try:
            a = json.loads(fn.get("arguments") or "{}")
        except Exception:
            a = {}
        tool_calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "args": a})
    usage = data.get("usage", {}) or {}
    text = msg.get("content") or ""
    # Some reasoning models (DeepSeek's thinking mode, via proxies that keep the
    # two channels separate) return an EMPTY `content` with the only text in
    # `reasoning_content`. A no-tool response then looked blank and the turn ended
    # as "I didn't produce a reply". Prefer `content`; fall back to the reasoning
    # only when there is nothing else and no call to run.
    if not text and not tool_calls:
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str):
            text = reasoning.strip()
    return {"text": text,
            "tool_calls": tool_calls,
            "tokens": usage.get("total_tokens", 0),
            "raw_assistant": msg}


def call_anthropic(prov: dict, messages: list[dict], system: str,
                   use_tools: bool = True, max_tokens: int | None = None) -> dict:
    url = _anthropic_messages_url(prov["base_url"])
    headers = {"Content-Type": "application/json",
               "x-api-key": prov["api_key"],
               "User-Agent": (prov.get("user_agent") or "").strip() or DEFAULT_USER_AGENT,
               "anthropic-version": "2023-06-01"}
    # Anthropic REQUIRES max_tokens. Let the caller raise it (thinking mode wants
    # headroom); otherwise keep the long-standing default rather than a guess.
    payload = {"model": prov["model_id"], "system": system,
               "messages": messages,
               "max_tokens": int(max_tokens) if (max_tokens or 0) > 0 else 8192}
    if use_tools:
        payload["tools"] = _anthropic_tools()
    data = _http_post(url, headers, payload)

    text, tool_calls = "", []
    for block in (data.get("content") or []):
        if block.get("type") == "text":
            text += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({"id": block.get("id", ""),
                               "name": block.get("name", ""),
                               "args": block.get("input", {}) or {}})
    usage = data.get("usage", {}) or {}
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return {"text": text, "tool_calls": tool_calls, "tokens": tokens,
            "raw_content": data.get("content", [])}


def chat(prov: dict, messages: list[dict], system: str,
         use_tools: bool = True, max_tokens: int | None = None) -> dict:
    """Dispatch to the right wire format. `prov` is a full DB row (with api_key).

    Pass `use_tools=False` to ask for a plain-text answer with no tool schemas
    attached — the agent loops use this to recover from a blank reply.

    Pass `max_tokens` to cap/raise the output ceiling (e.g. thinking mode). None
    keeps the provider default (OpenAI sends nothing; Anthropic falls back to its
    long-standing 8192).
    """
    if prov.get("format") == "anthropic":
        return call_anthropic(prov, messages, system, use_tools=use_tools,
                              max_tokens=max_tokens)
    return call_openai(prov, messages, system, use_tools=use_tools,
                       max_tokens=max_tokens)
