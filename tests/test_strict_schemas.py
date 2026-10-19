import pytest

from duckdb_agent.tools import SUBMIT_ANSWER_TOOL, TOOLS

ALL_TOOLS = TOOLS + [SUBMIT_ANSWER_TOOL]

# Numeric and string-length constraints aren't supported by strict mode. The SDK
# strips them for its typed helpers, but raw dict tools are sent as-is.
UNSUPPORTED_KEYWORDS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength",
}


def walk(schema, path="input_schema"):
    """Yield (path, subschema) for every schema node."""
    yield path, schema
    for name, sub in schema.get("properties", {}).items():
        yield from walk(sub, f"{path}.{name}")
    if "items" in schema:
        yield from walk(schema["items"], f"{path}[]")


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t["name"])
def test_tool_is_strict(tool):
    assert tool["strict"] is True


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t["name"])
def test_every_object_forbids_additional_properties(tool):
    for path, schema in walk(tool["input_schema"]):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False, path


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t["name"])
def test_no_unsupported_keywords(tool):
    for path, schema in walk(tool["input_schema"]):
        assert not UNSUPPORTED_KEYWORDS & schema.keys(), path


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t["name"])
def test_required_keys_are_declared(tool):
    schema = tool["input_schema"]
    assert set(schema.get("required", [])) <= schema["properties"].keys()


def test_tool_names_are_unique():
    names = [tool["name"] for tool in ALL_TOOLS]
    assert len(names) == len(set(names))
