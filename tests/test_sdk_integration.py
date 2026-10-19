"""The real anthropic SDK against a local HTTP server playing scripted API responses.

Every other test uses a fake client, which can't catch an SDK upgrade that rejects one
of our arguments or parses responses differently. These tests check the actual HTTP
requests we send and that real SDK response objects flow through the loop.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic
import duckdb
import pytest

from duckdb_agent.config import CONTEXT_MANAGEMENT, CONTEXT_MANAGEMENT_BETA, AgentConfig
from duckdb_agent.loop import run_agent
from duckdb_agent.tracing import init_traces_db


def message(stop_reason, content, usage=None, **extra):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 100, "output_tokens": 20,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        **extra,
    }


THINKING_BLOCK = {"type": "thinking", "thinking": "Count the rows first.", "signature": "sig_abc"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "query_database", "input": {"sql": "SELECT count(*) AS n FROM t"}}


class ScriptedAPI:
    """A local server that answers each POST with the next scripted (status, body) pair."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                api.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, reply = api.replies.pop(0)
                out = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("request-id", f"req_{len(api.requests)}")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def client(self, max_retries=0):
        return anthropic.Anthropic(
            api_key="test-key", base_url=f"http://127.0.0.1:{self.server.server_port}", max_retries=max_retries
        )

    def close(self):
        self.server.shutdown()


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(7)")
    return conn


@pytest.fixture
def traces_conn(tmp_path):
    return init_traces_db(str(tmp_path / "traces.duckdb"))


@pytest.fixture
def two_turn_api():
    api = ScriptedAPI(
        (200, message("tool_use", [THINKING_BLOCK, TOOL_USE],
                      usage={"input_tokens": 3000, "output_tokens": 150,
                             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2500})),
        (200, message("end_turn", [{"type": "text", "text": "t has 7 rows."}],
                      usage={"input_tokens": 200, "output_tokens": 40,
                             "cache_read_input_tokens": 5500, "cache_creation_input_tokens": 300},
                      context_management={"applied_edits": [
                          {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2, "cleared_input_tokens": 21000}
                      ]})),
    )
    yield api
    api.close()


def test_request_shape(conn, two_turn_api):
    run_agent(conn, "How many rows?", tables="t", client=two_turn_api.client())
    first = two_turn_api.requests[0]
    assert first["path"] == "/v1/messages?beta=true"
    assert first["headers"]["anthropic-beta"] == CONTEXT_MANAGEMENT_BETA
    body = first["body"]
    assert body["model"] == "claude-haiku-5-5"
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["context_management"] == CONTEXT_MANAGEMENT
    assert all(tool["strict"] is True for tool in body["tools"])
    assert not {"temperature", "top_p", "top_k"} & body.keys()


def test_second_request_replays_thinking_and_sends_tool_result(conn, two_turn_api):
    run_agent(conn, "How many rows?", tables="t", client=two_turn_api.client())
    messages = two_turn_api.requests[1]["body"]["messages"]
    assistant, tool_results = messages[1], messages[2]
    # The thinking block goes back exactly as received, signature included.
    assert assistant["content"][0] == THINKING_BLOCK
    assert assistant["content"][1]["id"] == "toolu_1"
    (result,) = tool_results["content"]
    assert result["tool_use_id"] == "toolu_1"
    assert "is_error" not in result
    assert json.loads(result["content"])["rows"] == [[7]]


def test_results_and_traces_from_real_responses(conn, two_turn_api, traces_conn):
    result = run_agent(conn, "How many rows?", tables="t", client=two_turn_api.client(), traces_conn=traces_conn)
    assert (result.status, result.text, result.api_calls) == ("completed", "t has 7 rows.", 2)
    assert result.usage == {
        "input_tokens": 3200, "cache_creation_input_tokens": 2800,
        "cache_read_input_tokens": 5500, "output_tokens": 190,
    }
    assert result.estimated_cost_usd > 0
    request_ids = traces_conn.execute(
        "SELECT request_id FROM api_calls WHERE run_id = ? ORDER BY id", [result.run_id]
    ).fetchall()
    assert request_ids == [("req_1",), ("req_2",)]
    actions = {row[0]: row[1] for row in traces_conn.execute(
        "SELECT action, coalesce(thinking, claude_response) FROM agent_traces WHERE run_id = ? "
        "AND action IN ('thinking', 'context_edit')", [result.run_id]
    ).fetchall()}
    assert actions["thinking"] == "Count the rows first."
    assert json.loads(actions["context_edit"])["cleared_tool_uses"] == 2


def test_bad_request_ends_the_run_with_request_id(conn, traces_conn):
    api = ScriptedAPI((400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad field"}}))
    try:
        result = run_agent(conn, "q", tables="t", client=api.client(), traces_conn=traces_conn)
    finally:
        api.close()
    assert result.status == "api_error"
    assert result.retryable is False
    assert "BadRequestError status=400 request_id=req_1" in result.error
    row = traces_conn.execute("SELECT request_id, error FROM api_calls WHERE run_id = ?", [result.run_id]).fetchone()
    assert row[0] == "req_1" and "bad field" in row[1]


def test_server_error_is_retried_by_the_sdk(conn):
    api = ScriptedAPI(
        (500, {"type": "error", "error": {"type": "api_error", "message": "try again"}}),
        (200, message("end_turn", [{"type": "text", "text": "ok"}])),
    )
    try:
        result = run_agent(conn, "q", tables="t", client=api.client(max_retries=1), config=AgentConfig())
    finally:
        api.close()
    assert result.status == "completed"
    assert len(api.requests) == 2  # one SDK retry, one run-level API call
    assert result.api_calls == 1
