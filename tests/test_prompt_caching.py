import copy
import json
from types import SimpleNamespace

import duckdb
import pytest

from duckdb_agent.config import AgentConfig, CACHE_CONTROL
from duckdb_agent.loop import run_agent
from duckdb_agent.prompts import SYSTEM_PROMPT
from duckdb_agent.usage import add_usage
from test_stop_reasons import FakeClient, response, text, tool_use


def usage(input=0, cache_read=None, cache_write=None, output=0):
    return SimpleNamespace(
        input_tokens=input,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        output_tokens=output,
    )


def with_usage(resp, **kwargs):
    resp.usage = usage(**kwargs)
    return resp


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT 1 AS a")
    return conn


class RecordingClient(FakeClient):
    """Also keeps each request's kwargs and a deep copy of its messages taken at send
    time, so later in-place edits to history show up as differences."""

    def __init__(self, *responses):
        super().__init__(*responses)
        self.requests = []
        self.snapshots = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        self.snapshots.append(copy.deepcopy(kwargs["messages"]))
        return super().create(**kwargs)


def test_system_prompt_has_cache_breakpoint(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="t", client=client)
    assert client.last_kwargs["system"] == [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": CACHE_CONTROL}
    ]


def test_automatic_caching_on_every_call(conn):
    client = RecordingClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="t", client=client)
    assert [request["cache_control"] for request in client.requests] == [CACHE_CONTROL, CACHE_CONTROL]


def test_wrap_up_call_keeps_caching(conn):
    client = RecordingClient(response("tool_use", tool_use("SELECT 1")), response("end_turn", text("summary")))
    run_agent(conn, "analyze", tables="t", client=client, config=AgentConfig(max_iterations=1))
    assert client.last_kwargs["tool_choice"] == {"type": "none"}
    assert client.last_kwargs["cache_control"] == CACHE_CONTROL


def test_caching_can_be_disabled(conn):
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="t", client=client, config=AgentConfig(prompt_caching=False))
    assert client.last_kwargs["system"] == SYSTEM_PROMPT
    assert "cache_control" not in client.last_kwargs


def test_at_most_four_breakpoints(conn):
    # The API rejects more than 4 cache_control breakpoints per request.
    client = FakeClient(response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="t", client=client, config=AgentConfig(structured_answer=True))
    kwargs = client.last_kwargs
    body = json.dumps({key: kwargs[key] for key in ("system", "tools", "messages")})
    breakpoints = body.count('"cache_control"') + ("cache_control" in kwargs)
    assert breakpoints <= 4


def test_cached_prefix_is_identical_across_runs(conn):
    # Tools + system render first; any byte difference between runs means a cache miss.
    first, second = FakeClient(response("end_turn", text("a"))), FakeClient(response("end_turn", text("b")))
    run_agent(conn, "first question", tables="t", client=first)
    run_agent(conn, "different question", client=second)
    for key in ("tools", "system"):
        assert json.dumps(first.last_kwargs[key]) == json.dumps(second.last_kwargs[key])


def comparable(messages):
    return json.dumps(messages, default=vars)


def test_history_is_append_only(conn):
    # Each request's messages must start with the previous request's messages,
    # unchanged, or the conversation cache misses from the first difference on.
    client = RecordingClient(
        response("tool_use", tool_use("SELECT 1")),
        response("tool_use", tool_use("SELECT 2", id="toolu_2")),
        response("end_turn", text("ok")),
    )
    run_agent(conn, "analyze", tables="t", client=client)
    sent = client.snapshots
    assert len(sent) == 3
    for previous, current in zip(sent, sent[1:]):
        assert comparable(current[:len(previous)]) == comparable(previous)


def test_wrap_up_only_appends_to_history(conn):
    # The wrap-up prompt joins the newest tool-results message, which no earlier
    # request has seen, so everything already sent stays byte-identical.
    client = RecordingClient(
        response("tool_use", tool_use("SELECT 1")),
        response("tool_use", tool_use("SELECT 2", id="toolu_2")),
        response("end_turn", text("summary")),
    )
    run_agent(conn, "analyze", tables="t", client=client, config=AgentConfig(max_iterations=2))
    sent = client.snapshots
    assert len(sent) == 3
    for previous, current in zip(sent, sent[1:]):
        assert comparable(current[:len(previous)]) == comparable(previous)


def test_usage_is_totalled_across_calls(conn):
    client = FakeClient(
        with_usage(response("tool_use", tool_use("SELECT 1")), input=5000, cache_write=2000, output=100),
        with_usage(response("end_turn", text("ok")), input=300, cache_read=7000, cache_write=400, output=200),
    )
    result = run_agent(conn, "analyze", tables="t", client=client)
    assert result.usage == {
        "input_tokens": 5300,
        "cache_creation_input_tokens": 2400,
        "cache_read_input_tokens": 7000,
        "output_tokens": 300,
    }


def test_usage_missing_is_empty(conn):
    result = run_agent(conn, "analyze", tables="t", client=FakeClient(response("end_turn", text("ok"))))
    assert result.usage == {}


def test_add_usage_treats_none_as_zero():
    totals = {}
    add_usage(totals, usage(input=10, output=5))
    assert totals == {
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 5,
    }
