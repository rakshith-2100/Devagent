from __future__ import annotations

from typing import Optional

from devagent.tools.base import ToolContext, ToolOutput, ToolRegistry, parse_docstring, schema_from_function, tool


@tool(tier=0)
def sample(ctx: ToolContext, path: str, count: int = 3, flag: bool = False, tag: Optional[str] = None, items: list[str] = None) -> ToolOutput:  # type: ignore[assignment]
    """Do a sample thing.

    Second paragraph of the description.

    Args:
        path: Where to look.
        count: How many. Defaults to 3.
        flag: Whether to flag
            with a continuation line.
        tag: Optional tag.
        items: Some items.
    """
    return ToolOutput(f"{path}:{count}:{flag}:{tag}:{items}")


def test_schema_generation() -> None:
    desc, params = schema_from_function(sample)
    assert desc.startswith("Do a sample thing.") and "Second paragraph" in desc
    props = params["properties"]
    assert props["path"] == {"type": "string", "description": "Where to look."}
    assert props["count"] == {"type": "integer", "description": "How many. Defaults to 3.", "default": 3}
    assert props["flag"]["type"] == "boolean" and "continuation line" in props["flag"]["description"]
    assert props["tag"] == {"type": "string", "description": "Optional tag."}
    assert props["items"]["type"] == "array" and props["items"]["items"] == {"type": "string"}
    assert params["required"] == ["path"]
    assert params["additionalProperties"] is False
    assert "ctx" not in props


def test_parse_docstring_without_args() -> None:
    assert parse_docstring("Just a line.") == ("Just a line.", {})
    assert parse_docstring(None) == ("", {})


def test_registry_subset_and_schemas() -> None:
    reg = ToolRegistry([sample])
    assert reg.names() == ["sample"]
    sub = reg.subset(["sample"])
    assert "sample" in sub and len(sub) == 1
    schema = reg.schemas()[0]
    assert set(schema) == {"name", "description", "input_schema"}
    try:
        reg.subset(["nope"])
    except KeyError as e:
        assert "Unknown tool" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_execute_coerces_and_validates() -> None:
    from devagent.core.types import ToolCall

    reg = ToolRegistry([sample])
    ctx = ToolContext(sandbox=None, config=None)
    res = reg.execute(ToolCall("1", "sample", {"path": "p", "count": "7", "flag": "true"}), ctx)
    assert res.ok and res.content.startswith("p:7:True")
    res = reg.execute(ToolCall("2", "sample", {"path": "p", "count": "seven"}), ctx)
    assert not res.ok and "should be integer" in res.content and "Expected arguments" in res.content
    res = reg.execute(ToolCall("3", "sample", {"path": "p", "count": True}), ctx)
    assert not res.ok and "got boolean" in res.content
