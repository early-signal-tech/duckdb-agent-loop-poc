import dataclasses
import json

import duckdb
import pytest

from duckdb_agent.config import AgentConfig, DEFAULT_CONFIG
from duckdb_agent.loop import run_agent
from duckdb_agent.prompts import build_system_prompt, SYSTEM_PROMPT
from duckdb_agent.tools import build_tools, run_tool, SUBMIT_ANSWER_TOOL, TOOLS
from duckdb_agent.tracing import init_traces_db
from test_stop_reasons import FakeClient, response, text, tool_use


@pytest.fixture
def conn():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(50)")
    return conn


def query_description(tools):
    return next(tool["description"] for tool in tools if tool["name"] == "query_database")


# --- defaults ---

def test_defaults():
    config = AgentConfig()
    assert config.model == "claude-haiku-5-5"
    assert config.effort == "medium"
    assert config.max_tokens == 16_000
    assert config.max_iterations == 25
    assert (config.max_result_rows, config.max_result_chars, config.max_cell_chars) == (200, 20_000, 500)
    assert config.query_timeout_seconds == 30
    assert config.prompt_caching and config.context_editing
    assert not config.structured_answer


def test_default_prompt_and_tools_are_byte_identical_to_constants():
    # Any byte difference would split the prompt cache between runs.
    assert build_system_prompt(DEFAULT_CONFIG) == SYSTEM_PROMPT
    assert json.dumps(build_tools(DEFAULT_CONFIG)) == json.dumps(TOOLS)


def test_equal_configs_build_identical_prompts():
    a, b = AgentConfig(max_result_rows=50), AgentConfig(max_result_rows=50)
    assert build_system_prompt(a) == build_system_prompt(b)
    assert json.dumps(build_tools(a)) == json.dumps(build_tools(b))


# --- immutability and validation ---

def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.effort = "high"


def test_replace_makes_a_changed_copy():
    high = dataclasses.replace(DEFAULT_CONFIG, effort="high")
    assert high.effort == "high"
    assert DEFAULT_CONFIG.effort == "medium"


@pytest.mark.parametrize("settings", [
    {"effort": "extreme"},
    {"max_tokens": 0},
    {"max_iterations": -1},
    {"max_result_rows": 2.5},
    {"max_result_chars": None},
    {"query_timeout_seconds": 0},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        AgentConfig(**settings)


# --- settings reach the request and the tools ---

def test_model_effort_and_max_tokens_reach_the_request(conn):
    client = FakeClient(response("end_turn", text("ok")))
    config = AgentConfig(model="claude-opus-5-5", effort="high", max_tokens=32_000)
    run_agent(conn, "analyze", tables="t", config=config, client=client)
    assert client.last_kwargs["model"] == "claude-opus-5-5"
    assert client.last_kwargs["output_config"] == {"effort": "high"}
    assert client.last_kwargs["max_tokens"] == 32_000


def test_limits_are_stated_in_prompt_and_tool_description():
    config = AgentConfig(max_result_rows=50, max_result_chars=8_000, max_cell_chars=100, query_timeout_seconds=10)
    prompt = build_system_prompt(config)
    assert "50 rows" in prompt and "8,000 characters" in prompt and "10 seconds" in prompt
    description = query_description(build_tools(config))
    assert "50 rows" in description and "8,000 characters" in description
    assert "100 characters" in description and "10s" in description


def test_limits_are_enforced_on_query_results(conn):
    config = AgentConfig(max_result_rows=3)
    result = run_tool(conn, "query_database", {"sql": "FROM t"}, config)
    assert result["row_count"] == 3
    assert result["truncated"] is True
    assert run_tool(conn, "sample_rows", {"table": "t", "n": 10}, AgentConfig(max_result_rows=4))["row_count"] == 4


def test_cell_limit_is_enforced(conn):
    result = run_tool(conn, "query_database", {"sql": "SELECT repeat('x', 300) AS v"}, AgentConfig(max_cell_chars=50))
    assert result["rows"][0][0].startswith("x" * 50 + "…[clipped")


def test_limits_apply_inside_the_loop(conn):
    client = FakeClient(response("tool_use", tool_use("FROM t")), response("end_turn", text("ok")))
    run_agent(conn, "analyze", tables="t", config=AgentConfig(max_result_rows=2), client=client)
    tool_result = json.loads(client.sent_messages[1][-1]["content"][0]["content"])
    assert tool_result["row_count"] == 2


def test_structured_answer_adds_submit_tool():
    assert build_tools(AgentConfig(structured_answer=True))[-1] == SUBMIT_ANSWER_TOOL
    assert SUBMIT_ANSWER_TOOL not in build_tools(DEFAULT_CONFIG)


def test_custom_system_prompt_is_used_as_is():
    assert build_system_prompt(AgentConfig(system_prompt="Be brief.")) == "Be brief."


# --- recorded with each run ---

def test_config_json_hashes_the_prompt():
    settings = json.loads(DEFAULT_CONFIG.to_json())
    assert settings["model"] == "claude-haiku-5-5"
    assert settings["system_prompt"].startswith("sha256:")
    custom = json.loads(AgentConfig(system_prompt="Be brief.").to_json())
    assert custom["system_prompt"] != settings["system_prompt"]
    # A limit change changes the built prompt, so the hash shows it too.
    assert json.loads(AgentConfig(max_result_rows=10).to_json())["system_prompt"] != settings["system_prompt"]


def test_run_summary_records_model_and_config(conn, tmp_path):
    traces_conn = init_traces_db(str(tmp_path / "traces.duckdb"))
    config = AgentConfig(effort="high")
    result = run_agent(
        conn, "analyze", tables="t", config=config, traces_conn=traces_conn,
        client=FakeClient(response("end_turn", text("ok")))
    )
    model, recorded = traces_conn.execute(
        "SELECT model, config FROM agent_runs WHERE run_id = ?", [result.run_id]
    ).fetchone()
    assert model == "claude-haiku-5-5"
    assert json.loads(recorded) == json.loads(config.to_json())
    effort = traces_conn.execute(
        "SELECT config->>'effort' FROM agent_runs WHERE run_id = ?", [result.run_id]
    ).fetchone()[0]
    assert effort == "high"
