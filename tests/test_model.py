from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from devagent.core.config import Config
from devagent.core.model import (
    BedrockConverseChat,
    ModelError,
    ScriptedChatModel,
    build_chat_model,
    build_converse_body,
    messages_to_converse,
    parse_converse_response,
    usage_from_message,
)

TOOL = {"name": "read_file", "description": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}


def test_messages_to_converse_shapes() -> None:
    msgs = [
        SystemMessage(content="static"),
        SystemMessage(content="dynamic"),
        HumanMessage(content="fix it"),
        AIMessage(content="looking", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "t1", "type": "tool_call"}, {"name": "read_file", "args": {"path": "b.py"}, "id": "t2", "type": "tool_call"}]),
        ToolMessage(content="a contents", tool_call_id="t1", name="read_file"),
        ToolMessage(content="DENIED", tool_call_id="t2", name="read_file", status="error"),
        AIMessage(content="", tool_calls=[]),
    ]
    system, conv = messages_to_converse(msgs)
    assert system == [{"text": "static"}, {"text": "dynamic"}]
    assert conv[0] == {"role": "user", "content": [{"text": "fix it"}]}
    assert conv[1]["role"] == "assistant"
    assert conv[1]["content"][0] == {"text": "looking"}
    assert conv[1]["content"][1]["toolUse"] == {"toolUseId": "t1", "name": "read_file", "input": {"path": "a.py"}}
    # both tool results land in ONE user message
    assert conv[2]["role"] == "user" and len(conv[2]["content"]) == 2
    assert conv[2]["content"][1]["toolResult"]["status"] == "error"
    assert conv[3] == {"role": "assistant", "content": [{"text": "(no content)"}]}


def test_build_body_defaults_and_cache_points() -> None:
    msgs = [SystemMessage(content="s1"), SystemMessage(content="s2"), HumanMessage(content="hi")]
    body = build_converse_body(msgs, tools=[TOOL], max_tokens=123)
    assert body["inferenceConfig"] == {"maxTokens": 123}
    assert "temperature" not in body["inferenceConfig"]
    assert body["toolConfig"]["tools"][0]["toolSpec"]["name"] == "read_file"
    assert body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["required"] == ["path"]
    assert "additionalModelRequestFields" not in body
    cached = build_converse_body(msgs, tools=[TOOL], prompt_cache=True, temperature=0.2, extra_request_fields={"x": 1})
    assert cached["system"] == [{"text": "s1"}, {"cachePoint": {"type": "default"}}, {"text": "s2"}]
    assert cached["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
    assert cached["inferenceConfig"]["temperature"] == 0.2
    assert cached["additionalModelRequestFields"] == {"x": 1}


def test_parse_response() -> None:
    data = {
        "output": {"message": {"role": "assistant", "content": [{"text": "I will read it."}, {"toolUse": {"toolUseId": "tu1", "name": "read_file", "input": {"path": "a.py"}}}]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 3, "cacheWriteInputTokens": 0},
    }
    ai, usage, stop = parse_converse_response(data)
    assert ai.content == "I will read it."
    assert ai.tool_calls == [{"name": "read_file", "args": {"path": "a.py"}, "id": "tu1", "type": "tool_call"}]
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (10, 5, 3)
    assert stop == "tool_use"
    assert usage_from_message(ai).cache_read_tokens == 3


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BedrockConverseChat, "_backoff", staticmethod(lambda *a, **k: None))
    seen: list[dict] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer k123"
        assert "/model/anthropic.claude-opus-5-5/converse" in str(request.url)
        if calls["n"] == 1:
            return httpx.Response(429, json={"message": "ThrottlingException"})
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": "ok"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1}})

    model = BedrockConverseChat(model_id="anthropic.claude-opus-5-5", bearer_token="k123", max_retries=2)
    model._client = _client_with(handler)
    hooks: list[str] = []
    model.request_hook = lambda purpose, body: hooks.append("req:" + purpose)
    model.response_hook = lambda purpose, raw, ms: hooks.append("res:" + purpose)
    bound = model.bind_tools([TOOL])
    out = bound.invoke([SystemMessage(content="s"), HumanMessage(content="hi")], purpose="agent")
    assert out.content == "ok" and calls["n"] == 2
    assert seen[-1]["toolConfig"]["tools"][0]["toolSpec"]["name"] == "read_file"
    assert hooks == ["req:agent", "res:agent"]


def test_http_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BedrockConverseChat, "_backoff", staticmethod(lambda *a, **k: None))
    model = BedrockConverseChat(model_id="m", bearer_token="k", max_retries=1)
    model._client = _client_with(lambda req: httpx.Response(403, json={"message": "forbidden"}))
    with pytest.raises(ModelError, match="HTTP 403"):
        model.invoke([HumanMessage(content="hi")])


def test_missing_token_is_a_clear_error() -> None:
    model = BedrockConverseChat(model_id="m")
    with pytest.raises(ModelError, match="BEDROCK_API_KEY"):
        model.invoke([HumanMessage(content="hi")])


def test_build_chat_model_selection() -> None:
    cfg = Config()
    cfg.provider = "bedrock"
    with pytest.raises(ModelError, match="No Bedrock credentials"):
        build_chat_model(cfg)
    cfg.bearer_token = "abc"
    assert isinstance(build_chat_model(cfg), BedrockConverseChat)
    cfg.provider = "mock"
    assert isinstance(build_chat_model(cfg), ScriptedChatModel)


def test_openai_compatible_conversion_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from devagent.core.model import OpenAICompatibleChat, messages_to_openai, parse_openai_response, summarize_raw_response

    msgs = [
        SystemMessage(content="s"),
        HumanMessage(content="fix it"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "t1", "type": "tool_call"}]),
        ToolMessage(content="contents", tool_call_id="t1", name="read_file"),
    ]
    conv = messages_to_openai(msgs)
    assert conv[0] == {"role": "system", "content": "s"}
    assert conv[2]["content"] is None and conv[2]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.py"}'}
    assert conv[3] == {"role": "tool", "tool_call_id": "t1", "content": "contents"}

    ok_body = {
        "model": "x/y",
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "reasoning": "hmm", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "src/calc.py"}'}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0},
    }
    ai, usage, stop = parse_openai_response(ok_body)
    assert stop == "tool_use" and ai.tool_calls[0]["args"] == {"path": "src/calc.py"} and ai.response_metadata["cost_usd"] == 0.0
    assert usage.input_tokens == 10 and ai.additional_kwargs["reasoning"] == "hmm"
    text, calls, u, s = summarize_raw_response(ok_body)
    assert calls[0]["name"] == "read_file" and s == "tool_use" and u["input_tokens"] == 10

    monkeypatch.setattr(BedrockConverseChat, "_backoff", staticmethod(lambda *a, **k: None))
    calls_seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls_seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer or-key"
        if len(calls_seen) == 1:  # OpenRouter-style: HTTP 200 carrying an upstream 503
            return httpx.Response(200, json={"error": {"message": "Service temporarily overloaded", "code": 503}})
        return httpx.Response(200, json=ok_body)

    model = OpenAICompatibleChat(model_id="x/y", api_key="or-key", max_retries=2)
    model._client = httpx.Client(transport=httpx.MockTransport(handler))
    out = model.bind_tools([TOOL]).invoke([HumanMessage(content="hi")], purpose="agent")
    assert out.tool_calls[0]["name"] == "read_file" and len(calls_seen) == 2
    assert calls_seen[-1]["tools"][0]["function"]["name"] == "read_file" and calls_seen[-1]["tool_choice"] == "auto"
    assert "_call_no" not in calls_seen[-1]

    cfg = Config()
    cfg.provider = "openrouter"
    cfg.openrouter_api_key = "k"
    assert isinstance(build_chat_model(cfg), OpenAICompatibleChat)
    assert cfg.effective_model() == cfg.openrouter_model


def test_scripted_model() -> None:
    model = ScriptedChatModel(script=[{"text": "a", "tool_calls": [{"name": "read_file", "args": {"path": "x"}}]}, {"text": "b"}])
    first = model.bind_tools([TOOL]).invoke([HumanMessage(content="hi")])
    assert first.tool_calls[0]["name"] == "read_file" and first.tool_calls[0]["id"]
    second = model.invoke([HumanMessage(content="hi")])
    assert second.content == "b" and second.usage_metadata["input_tokens"] == 1000
    third = model.invoke([HumanMessage(content="hi")])
    assert "no more responses" in third.content
    assert len(model.calls) == 3
