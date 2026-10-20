"""Tool protocol, registry, and JSON-schema generation from signatures.

A tool is a plain function whose first parameter is ``ctx: ToolContext``.
Its docstring (Google style, with an ``Args:`` section) becomes the description
the model sees. The decorator reads the signature with ``inspect`` and the
annotations with ``typing.get_type_hints`` and builds the schema; nothing is
hand-written, so documentation cannot drift from the implementation.

Error messages are instructions: every failure says what went wrong, the
expected shape, and the next action.
"""
from __future__ import annotations

import inspect
import json
import re
import time
import traceback
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Union, get_args, get_origin, get_type_hints

from ..core.types import ToolCall, ToolResult


class ToolError(Exception):
    """Raised by a tool when it cannot do what was asked. The message is for the model."""


@dataclass
class ToolOutput:
    content: str
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """What a tool may touch. Tools never import the session or the graph."""

    sandbox: Any
    config: Any
    runtime: Any = None  # AgentRuntime: plan, notes, tracer, delegate hook, test state
    approved: bool = False  # True only for the one call the user explicitly approved


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    tier: int = 0

    def schema(self) -> dict[str, Any]:
        """Anthropic-style tool definition; providers convert from this."""
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


# ------------------------------------------------------------- schema builder

_PRIMITIVES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _annotation_to_schema(ann: Any) -> dict[str, Any]:
    if ann is inspect.Parameter.empty or ann is Any:
        return {}
    if ann in _PRIMITIVES:
        return {"type": _PRIMITIVES[ann]}
    if ann is type(None):
        return {"type": "null"}
    origin = get_origin(ann)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}
    if origin in (list, typing.List, tuple, set):
        args = get_args(ann)
        items = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items}
    if origin in (dict, typing.Dict) or ann is dict:
        return {"type": "object"}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(get_args(ann))}
    if ann is list:
        return {"type": "array"}
    return {}


_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters):\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^\s*(Returns|Raises|Yields|Examples?|Notes?):\s*$", re.IGNORECASE)
_PARAM_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc: Optional[str]) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (description, {param: description})."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    desc_lines: list[str] = []
    params: dict[str, str] = {}
    i = 0
    while i < len(lines) and not _ARGS_HEADER.match(lines[i]):
        desc_lines.append(lines[i])
        i += 1
    i += 1  # skip the Args: header
    current: Optional[str] = None
    while i < len(lines):
        line = lines[i]
        if _SECTION_HEADER.match(line):
            break
        m = _PARAM_LINE.match(line)
        if m and not line.startswith(" " * 8):
            current = m.group(1).lstrip("*")
            params[current] = m.group(3).strip()
        elif current and line.strip():
            params[current] = (params[current] + " " + line.strip()).strip()
        i += 1
    return "\n".join(desc_lines).strip(), params


def _json_default(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def schema_from_function(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    description, param_docs = parse_docstring(fn.__doc__)
    try:
        hints = get_type_hints(fn)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for idx, (name, param) in enumerate(sig.parameters.items()):
        if idx == 0:  # ctx
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        schema = dict(_annotation_to_schema(hints.get(name, param.annotation)))
        if name in param_docs:
            schema["description"] = param_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            schema["default"] = _json_default(param.default)
        properties[name] = schema
    parameters: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        parameters["required"] = required
    return description, parameters


def tool(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None, tier: Optional[int] = None) -> Any:
    """Decorator: turn a function into a Tool (kept as ``fn.__devagent_tool__``).

    Works bare (``@tool``) or with arguments (``@tool(name="x")``).
    """

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or f.__name__
        description, parameters = schema_from_function(f)
        if tier is None:
            from ..policy.rules import TIERS

            t = TIERS.get(tool_name, 0)
        else:
            t = tier
        f.__devagent_tool__ = Tool(name=tool_name, description=description, parameters=parameters, fn=f, tier=t)  # type: ignore[attr-defined]
        return f

    return wrap(fn) if fn is not None else wrap


def as_tool(obj: Any) -> Tool:
    if isinstance(obj, Tool):
        return obj
    t = getattr(obj, "__devagent_tool__", None)
    if t is None:
        raise TypeError(f"{obj!r} is not a tool. Decorate it with @tool.")
    return t


# ------------------------------------------------------------------ registry

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


class ToolRegistry:
    def __init__(self, tools: Iterable[Any] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Any) -> Tool:
        tool_obj = as_tool(t)
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def subset(self, names: Iterable[str]) -> "ToolRegistry":
        chosen = []
        for n in names:
            if n not in self._tools:
                raise KeyError(f"Unknown tool '{n}'. Known tools: {', '.join(self.names())}")
            chosen.append(self._tools[n])
        return ToolRegistry(chosen)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    # ------------------------------------------------------------ execution
    def _validate(self, t: Tool, args: dict[str, Any]) -> Optional[str]:
        props = t.parameters.get("properties", {})
        required = t.parameters.get("required", [])
        unknown = sorted(set(args) - set(props))
        missing = [r for r in required if r not in args or args[r] is None]
        problems: list[str] = []
        if missing:
            problems.append(f"missing required argument(s): {', '.join(missing)}")
        if unknown:
            problems.append(f"unknown argument(s): {', '.join(unknown)}")
        for key, value in args.items():
            if key not in props or value is None:
                continue
            expected = props[key].get("type")
            if expected in ("integer", "number") and isinstance(value, bool):
                problems.append(f"'{key}' should be {expected}, got boolean")
                continue
            if expected in _JSON_TYPES and not isinstance(value, _JSON_TYPES[expected]):
                if expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                    continue
                if expected in ("integer", "number") and isinstance(value, str) and value.strip().lstrip("-").replace(".", "", 1).isdigit():
                    continue
                problems.append(f"'{key}' should be {expected}, got {type(value).__name__}")
        if not problems:
            return None
        sig = ", ".join(f"{k}{'' if k in required else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        return f"{t.name}: {'; '.join(problems)}. Expected arguments: {t.name}({sig})."

    @staticmethod
    def _coerce(t: Tool, args: dict[str, Any]) -> dict[str, Any]:
        props = t.parameters.get("properties", {})
        out: dict[str, Any] = {}
        for key, value in args.items():
            expected = props.get(key, {}).get("type")
            if isinstance(value, str):
                if expected == "boolean" and value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif expected == "integer" and value.strip().lstrip("-").isdigit():
                    value = int(value)
                elif expected == "number":
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            out[key] = value
        return out

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        t = self.get(call.name)
        started = time.monotonic()
        if t is None:
            return ToolResult(
                call_id=call.id,
                ok=False,
                content=f"Unknown tool '{call.name}'. Available tools: {', '.join(self.names())}.",
            )
        args = dict(call.args or {})
        problem = self._validate(t, args)
        if problem:
            return ToolResult(call_id=call.id, ok=False, content=problem, meta={"duration_ms": 0})
        args = self._coerce(t, args)
        try:
            out = t.fn(ctx, **args)
        except ToolError as e:
            return ToolResult(call_id=call.id, ok=False, content=str(e), meta={"duration_ms": _ms(started)})
        except Exception as e:  # noqa: BLE001 - tool failures become observations, never crashes
            from ..sandbox.local import SandboxError

            if isinstance(e, SandboxError):
                return ToolResult(call_id=call.id, ok=False, content=str(e), meta={"duration_ms": _ms(started)})
            tb = traceback.format_exc(limit=3)
            return ToolResult(
                call_id=call.id,
                ok=False,
                content=f"{t.name} failed with {type(e).__name__}: {e}. If the arguments look right, try a different approach.",
                meta={"duration_ms": _ms(started), "traceback": tb},
            )
        if isinstance(out, ToolOutput):
            content, truncated, meta = out.content, out.truncated, dict(out.meta)
        else:
            content, truncated, meta = str(out), False, {}
        meta["duration_ms"] = _ms(started)
        meta["bytes"] = len(content.encode("utf-8", "replace"))
        return ToolResult(call_id=call.id, ok=True, content=content, truncated=truncated, meta=meta)
