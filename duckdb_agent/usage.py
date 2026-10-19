"""Token totals and cost estimates."""


# USD per million tokens, for cost estimates in traces. Haiku 5.5 has two rate cards,
# picked per request by prompt size (input + cache write + cache read tokens): up to
# 100K tokens, and above. Cache reads cost 0.1x the input rate and 5-minute cache writes
# 1.25x. Output includes thinking tokens. Models not listed get no estimate (None).
# Estimates only: check the pricing page before budgeting on them.
MODEL_PRICING = {
    "claude-haiku-5-5": {
        "long_prompt_threshold": 100_000,
        "standard": {"input": 0.10, "output": 0.50},
        "long_prompt": {"input": 0.50, "output": 2.50},
    },
}


CACHE_READ_MULTIPLIER = 0.1


CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL (CACHE_CONTROL)


def estimate_cost(model: str, usage) -> float | None:
    """Estimated USD cost of one API call from its usage, or None for unpriced models."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    output_tokens = getattr(usage, "output_tokens", None) or 0
    prompt_tokens = input_tokens + cache_write + cache_read
    rates = pricing["long_prompt" if prompt_tokens > pricing["long_prompt_threshold"] else "standard"]
    input_rate = rates["input"] / 1_000_000
    return (
        input_tokens * input_rate
        + cache_write * input_rate * CACHE_WRITE_MULTIPLIER
        + cache_read * input_rate * CACHE_READ_MULTIPLIER
        + output_tokens * rates["output"] / 1_000_000
    )


USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def add_usage(totals: dict, usage) -> None:
    """Add one response's usage to running totals. Cache fields are None when unused."""
    for name in USAGE_FIELDS:
        totals[name] = totals.get(name, 0) + (getattr(usage, name, None) or 0)
