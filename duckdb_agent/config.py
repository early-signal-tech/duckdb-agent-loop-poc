"""Run settings: AgentConfig and the defaults behind it."""

import hashlib
from dataclasses import asdict, dataclass

from .serialization import json_dumps


# Result-size caps for what is sent back to Claude. Every tool result stays in the
# conversation and is resent on every later iteration, so large results compound.
MAX_RESULT_ROWS = 200


MAX_RESULT_CHARS = 20_000  # serialized JSON, roughly 6-9k tokens on Haiku 5.5's tokenizer


MAX_CELL_CHARS = 500  # long text values are clipped individually


# Queries running longer than this are cancelled and reported to Claude as an error.
QUERY_TIMEOUT_SECONDS = 30


# Claude Haiku 5.5 ($0.10 / $0.50 per MTok input / output for prompts up to 100K
# tokens). The agent relies on Haiku 5.5-era features (adaptive thinking with summaries,
# effort, mid-conversation system messages), so older models such as Haiku 4.5 won't work.
DEFAULT_MODEL = "claude-haiku-5-5"


# How much the model thinks: low | medium | high | xhigh | max. "medium" is Haiku 5.5's
# default, set explicitly so it's visible.
DEFAULT_EFFORT = "medium"


# Adaptive thinking (the model decides when and how much to think) with summarized
# thinking text, which is logged to agent_traces for debugging. Without "summarized",
# thinking blocks come back with empty text. Thinking blocks are passed back unchanged
# with tool results, because the full response.content is appended to history.
THINKING = {"type": "adaptive", "display": "summarized"}


# Per-response output cap, including thinking tokens. It's a ceiling, not a target: you
# pay only for tokens generated. Too low truncates answers (stop_reason "max_tokens"). The SDK refuses
# non-streaming requests above ~21k, so going higher means switching to streaming.
DEFAULT_MAX_TOKENS = 16_000


# Prompt caching, 5-minute TTL: loop turns are seconds apart, so every call refreshes
# the entry and the pricier 1-hour TTL would buy nothing. Two breakpoints:
#   1. on the system prompt -- caches tools + system (they render first), which are
#      identical on every call of every run, so even a run's first call can hit it
#      (if another run with the same model and settings ran in the last 5 minutes);
#   2. top-level (automatic) -- moves to the end of the conversation on each call,
#      so each iteration reads everything before it and writes only the new turn.
# Prefixes below the model's minimum (512 tokens on Haiku 5.5) silently don't cache;
# usage shows cache_creation/cache_read of 0.
CACHE_CONTROL = {"type": "ephemeral"}


# Server-side context editing: once a prompt passes the trigger, the API replaces old
# tool results with a placeholder before the model sees them. It must be server-side:
# editing history ourselves would invalidate Haiku 5.5's later thinking blocks (a 400),
# while server-side edits don't count, because the check runs on what we send.
#
# Why 80K: Haiku 5.5 input costs 5x more ($0.50 vs $0.10 per MTok) once a prompt is
# over 100K tokens, and the next turn's results add more on top of the trigger point.
# Each clear rewrites the cached conversation, so clears should be rare and large:
# clear_at_least skips a clear that wouldn't free enough to pay for that cache write.
# The SQL (tool inputs) stays visible, so Claude knows what it ran and can re-run it.
# Schema lookups are never cleared: they're small and stay relevant all run.
CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"


CONTEXT_MANAGEMENT = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 80_000},
            "keep": {"type": "tool_uses", "value": 8},
            "clear_at_least": {"type": "input_tokens", "value": 20_000},
            "exclude_tools": ["list_tables", "describe_table"],
        }
    ]
}


# SDK-level retries (exponential backoff, honors retry-after) for 408/409/429/5xx
# and connection errors. The SDK default is 2; agent runs are worth waiting for.
API_MAX_RETRIES = 4


# Tool-use rounds before the forced wrap-up call (see MAX_ITERATIONS_WRAP_UP_PROMPT).
DEFAULT_MAX_ITERATIONS = 25


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class AgentConfig:
    """Every setting that shapes a run, in one place. Defaults are the constants above.

    Frozen, so one config can be shared across runs safely, and recorded with each run
    in agent_runs.config. Change a field with dataclasses.replace(config, effort="high").

    The tool descriptions and system prompt state the result limits and timeout, so
    they're rebuilt from these values. With defaults they're byte-identical to TOOLS
    and SYSTEM_PROMPT, which keeps them cacheable across runs; any change makes a new
    cache entry.
    """
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    # Limits on what each query returns to Claude, and how long it may run.
    max_result_rows: int = MAX_RESULT_ROWS
    max_result_chars: int = MAX_RESULT_CHARS
    max_cell_chars: int = MAX_CELL_CHARS
    query_timeout_seconds: float = QUERY_TIMEOUT_SECONDS
    # Offer submit_answer; its input becomes AgentResult.answer.
    structured_answer: bool = False
    # Off switches for API features some gateways reject.
    prompt_caching: bool = True
    context_editing: bool = True
    # None builds the default prompt from these settings (build_system_prompt).
    system_prompt: str | None = None

    def __post_init__(self):
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}, not {self.effort!r}")
        for name in ("max_tokens", "max_iterations", "max_result_rows", "max_result_chars", "max_cell_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, not {value!r}")
        if not self.query_timeout_seconds or self.query_timeout_seconds <= 0:
            raise ValueError(f"query_timeout_seconds must be positive, not {self.query_timeout_seconds!r}")

    def to_json(self) -> str:
        """Settings as JSON for agent_runs.config. The system prompt actually sent (custom
        or built) is recorded as a short hash, so runs can be grouped by prompt version
        without storing the text each time."""
        # Imported here because prompts imports this module (for AgentConfig).
        from .prompts import build_system_prompt

        settings = asdict(self)
        prompt = build_system_prompt(self)
        settings["system_prompt"] = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return json_dumps(settings)


DEFAULT_CONFIG = AgentConfig()
