"""One thin wrapper over the model API; no application logic here.

``BedrockConverseChat`` is a LangChain ``BaseChatModel`` that talks to the
Amazon Bedrock Converse API over HTTPS with a bearer token (a Bedrock API key
or a short-term token). It converts LangChain messages to Converse blocks and
back, handles retries, and reports usage. Everything else (policy, budgets,
tool execution) lives in the graph.

``OpenAICompatibleChat`` does the same for OpenAI-style ``/chat/completions``
endpoints such as OpenRouter, so any model reachable there can drive the agent.

``ScriptedChatModel`` returns canned responses so the whole system can be
exercised without an API key.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterator, Optional, Sequence
from urllib.parse import quote

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from .config import Config
from .types import Usage

RequestHook = Callable[[str, dict[str, Any]], None]
ResponseHook = Callable[[str, dict[str, Any], int], None]

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "InternalServerException", "ModelTimeoutException"}


class ModelError(RuntimeError):
    """Infrastructure failure talking to the model. The loop does not catch this."""


# ----------------------------------------------------------- message convert


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in (None, "text") and "text" in block:
                parts.append(str(block["text"]))
            elif block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p)


def tool_schema_to_converse(schema: dict[str, Any]) -> dict[str, Any]:
    """Accept Anthropic-style ({name, description, input_schema}) or OpenAI-style
    ({function: {name, description, parameters}}) definitions."""
    if "function" in schema:
        fn = schema["function"]
        name, desc, params = fn["name"], fn.get("description", ""), fn.get("parameters") or {"type": "object", "properties": {}}
    else:
        name = schema["name"]
        desc = schema.get("description", "")
        params = schema.get("input_schema") or schema.get("parameters") or {"type": "object", "properties": {}}
    return {"toolSpec": {"name": name, "description": desc or name, "inputSchema": {"json": params}}}


def messages_to_converse(messages: Sequence[BaseMessage]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (system_blocks, converse_messages). Consecutive same-role turns are merged."""
    system: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        if isinstance(m, SystemMessage):
            text = _text_of(m.content)
            if text:
                system.append({"text": text})
        elif isinstance(m, HumanMessage):
            text = _text_of(m.content)
            push("user", [{"text": text or "(empty)"}])
        elif isinstance(m, AIMessage):
            blocks: list[dict[str, Any]] = []
            text = _text_of(m.content)
            if text.strip():
                blocks.append({"text": text})
            for tc in m.tool_calls or []:
                blocks.append({"toolUse": {"toolUseId": tc["id"], "name": tc["name"], "input": tc.get("args") or {}}})
            if not blocks:
                blocks.append({"text": "(no content)"})
            push("assistant", blocks)
        elif isinstance(m, ToolMessage):
            text = _text_of(m.content)
            status = "error" if getattr(m, "status", "success") == "error" else "success"
            push(
                "user",
                [{"toolResult": {"toolUseId": m.tool_call_id, "content": [{"text": text or "(empty result)"}], "status": status}}],
            )
        else:  # pragma: no cover - other message types are not produced here
            push("user", [{"text": _text_of(m.content)}])
    return system, out


def build_converse_body(
    messages: Sequence[BaseMessage],
    *,
    tools: Optional[Sequence[dict[str, Any]]] = None,
    max_tokens: int = 8192,
    temperature: Optional[float] = None,
    prompt_cache: bool = False,
    extra_request_fields: Optional[dict[str, Any]] = None,
    stop: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    system, conv = messages_to_converse(messages)
    body: dict[str, Any] = {"messages": conv, "inferenceConfig": {"maxTokens": int(max_tokens)}}
    if temperature is not None:
        body["inferenceConfig"]["temperature"] = float(temperature)
    if stop:
        body["inferenceConfig"]["stopSequences"] = list(stop)
    if system:
        if prompt_cache:
            # Cache the static system prompt (first block); the dynamic block follows the cache point.
            system = [system[0], {"cachePoint": {"type": "default"}}, *system[1:]]
        body["system"] = system
    if tools:
        specs = [tool_schema_to_converse(t) if "toolSpec" not in t else t for t in tools]
        if prompt_cache:
            specs.append({"cachePoint": {"type": "default"}})
        body["toolConfig"] = {"tools": specs}
    if extra_request_fields:
        body["additionalModelRequestFields"] = dict(extra_request_fields)
    return body


def parse_converse_response(data: dict[str, Any]) -> tuple[AIMessage, Usage, str]:
    """Return (AIMessage, Usage, stop_reason) from a Converse response body."""
    message = ((data.get("output") or {}).get("message")) or {}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[str] = []
    for block in message.get("content") or []:
        if "text" in block:
            text_parts.append(str(block["text"]))
        elif "toolUse" in block:
            tu = block["toolUse"]
            args = tu.get("input")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append({"name": tu.get("name", ""), "args": args or {}, "id": tu.get("toolUseId", ""), "type": "tool_call"})
        elif "reasoningContent" in block:
            rc = block["reasoningContent"]
            txt = (rc.get("reasoningText") or {}).get("text") if isinstance(rc, dict) else None
            if txt:
                reasoning.append(str(txt))
    u = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(u.get("inputTokens", 0) or 0),
        output_tokens=int(u.get("outputTokens", 0) or 0),
        cache_read_tokens=int(u.get("cacheReadInputTokens", 0) or 0),
        cache_write_tokens=int(u.get("cacheWriteInputTokens", 0) or 0),
    )
    stop_reason = str(data.get("stopReason", ""))
    ai = AIMessage(
        content="\n".join(text_parts),
        tool_calls=tool_calls,
        usage_metadata=UsageMetadata(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            input_token_details={"cache_read": usage.cache_read_tokens, "cache_creation": usage.cache_write_tokens},  # type: ignore[typeddict-item]
        ),
        response_metadata={"stop_reason": stop_reason, "model_id": str(data.get("_model_id", "")), "latency_ms": int(data.get("_latency_ms", 0))},
        additional_kwargs={"reasoning": "\n".join(reasoning)} if reasoning else {},
    )
    return ai, usage, stop_reason


def usage_from_message(message: AIMessage) -> Usage:
    um = message.usage_metadata or {}
    details = um.get("input_token_details") or {}
    return Usage(
        input_tokens=int(um.get("input_tokens", 0) or 0),
        output_tokens=int(um.get("output_tokens", 0) or 0),
        cache_read_tokens=int(details.get("cache_read", 0) or 0),
        cache_write_tokens=int(details.get("cache_creation", 0) or 0),
    )


# ------------------------------------------------------------ chat models


class BedrockConverseChat(BaseChatModel):
    """Bedrock Converse API over httpx with ``Authorization: Bearer <token>``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    region: str = "us-east-1"
    bearer_token: Optional[str] = Field(default=None, repr=False, exclude=True)
    max_tokens: int = 8192
    temperature: Optional[float] = None
    timeout_s: float = 180.0
    max_retries: int = 3
    prompt_cache: bool = False
    extra_request_fields: dict[str, Any] = Field(default_factory=dict)
    request_hook: Optional[RequestHook] = Field(default=None, exclude=True)
    response_hook: Optional[ResponseHook] = Field(default=None, exclude=True)

    _client: Optional[httpx.Client] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "bedrock-converse-httpx"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "region": self.region}

    @property
    def endpoint(self) -> str:
        return f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{quote(self.model_id, safe='')}/converse"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Optional[str] = None, **kwargs: Any) -> Any:
        specs = []
        for t in tools:
            if isinstance(t, dict):
                specs.append(tool_schema_to_converse(t) if "toolSpec" not in t else t)
            else:
                from langchain_core.utils.function_calling import convert_to_openai_tool

                specs.append(tool_schema_to_converse(convert_to_openai_tool(t)))
        return self.bind(tools=specs, **kwargs)

    # -- transport -----------------------------------------------------------
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=20.0))
        return self._client

    def _send(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.bearer_token:
            raise ModelError(
                "No Bedrock credentials: set BEDROCK_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) in the environment or .env."
            )
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {self.bearer_token}"}
        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http().post(self.endpoint, json=body, headers=headers)
            except httpx.TimeoutException as e:
                last_error = f"timeout after {self.timeout_s}s: {e}"
                self._backoff(attempt)
                continue
            except httpx.HTTPError as e:
                last_error = f"connection error: {e}"
                self._backoff(attempt)
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise ModelError(f"Bedrock returned non-JSON body: {e}: {resp.text[:500]}") from e
            text = resp.text[:1000]
            code = ""
            try:
                code = str(resp.json().get("__type", "") or resp.json().get("message", ""))
            except ValueError:
                pass
            retryable = resp.status_code in RETRYABLE_STATUS or any(c in code for c in RETRYABLE_CODES)
            last_error = f"HTTP {resp.status_code} from Bedrock: {text}"
            if not retryable or attempt >= self.max_retries:
                raise ModelError(last_error + self._hint(resp.status_code))
            self._backoff(attempt, retry_after=resp.headers.get("retry-after"))
        raise ModelError(f"Bedrock request failed after {self.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _hint(status: int) -> str:
        if status in (401, 403):
            return " (check BEDROCK_API_KEY, the region, and that the model is enabled for your account)"
        if status == 404:
            return " (check BEDROCK_MODEL_ID; some models need a cross-region inference profile id such as 'us.anthropic....')"
        if status == 400:
            return " (request rejected: check max tokens, tool schemas, or unsupported fields like temperature for this model)"
        return ""

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str] = None) -> None:
        # 2, 4, 8, 16, 32, 60 s: shared free endpoints report "temporarily overloaded" for
        # tens of seconds at a time, and giving up after ten seconds wastes the whole run.
        delay = 2.0 * (2**attempt)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(min(delay, 60))

    # -- generation ----------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        purpose = str(kwargs.pop("purpose", "agent"))
        tools = kwargs.pop("tools", None)
        body = build_converse_body(
            messages,
            tools=tools,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            temperature=self.temperature,
            prompt_cache=self.prompt_cache,
            extra_request_fields=self.extra_request_fields,
            stop=stop,
        )
        if self.request_hook:
            self.request_hook(purpose, body)
        started = time.monotonic()
        data = self._send(body)
        latency_ms = int((time.monotonic() - started) * 1000)
        data["_latency_ms"] = latency_ms
        data["_model_id"] = self.model_id
        if self.response_hook:
            self.response_hook(purpose, data, latency_ms)
        ai, _usage, _stop = parse_converse_response(data)
        return ChatResult(generations=[ChatGeneration(message=ai)])


class BedrockBoto3Chat(BedrockConverseChat):
    """Same request/response shape, sent through boto3 (SigV4 credentials)."""

    aws_access_key_id: Optional[str] = Field(default=None, exclude=True)
    aws_secret_access_key: Optional[str] = Field(default=None, exclude=True)
    aws_session_token: Optional[str] = Field(default=None, exclude=True)
    _boto: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "bedrock-converse-boto3"

    def _send(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._boto is None:
            try:
                import boto3  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise ModelError("boto3 is not installed; pip install boto3, or set BEDROCK_API_KEY to use the HTTP path.") from e
            kwargs: dict[str, Any] = {"service_name": "bedrock-runtime", "region_name": self.region}
            if self.aws_access_key_id and self.aws_secret_access_key:
                kwargs["aws_access_key_id"] = self.aws_access_key_id
                kwargs["aws_secret_access_key"] = self.aws_secret_access_key
                if self.aws_session_token:
                    kwargs["aws_session_token"] = self.aws_session_token
            self._boto = boto3.client(**kwargs)
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._boto.converse(modelId=self.model_id, **body)
                resp.pop("ResponseMetadata", None)
                return resp
            except Exception as e:  # noqa: BLE001 - botocore error hierarchy is dynamic
                code = getattr(getattr(e, "response", {}), "get", lambda *_: {})("Error", {}).get("Code", "") if hasattr(e, "response") else ""
                last_error = f"{type(e).__name__}: {e}"
                if code in RETRYABLE_CODES and attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                raise ModelError(f"Bedrock (boto3) call failed: {last_error}") from e
        raise ModelError(f"Bedrock (boto3) call failed after retries: {last_error}")


# ------------------------------------------------ OpenAI-compatible endpoints

OPENAI_FINISH_TO_STOP = {"tool_calls": "tool_use", "function_call": "tool_use", "stop": "end_turn", "length": "max_tokens"}


def tool_schema_to_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Anthropic-style ({name, description, input_schema}) -> OpenAI function tool."""
    if "function" in schema:
        return schema
    params = schema.get("input_schema") or schema.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {"name": schema["name"], "description": schema.get("description") or schema["name"], "parameters": params},
    }


def messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": _text_of(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _text_of(m.content) or "(empty)"})
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": _text_of(m.content) or None}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False)},
                    }
                    for tc in m.tool_calls
                ]
            elif entry["content"] is None:
                entry["content"] = "(no content)"
            out.append(entry)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": _text_of(m.content) or "(empty result)"})
        else:  # pragma: no cover
            out.append({"role": "user", "content": _text_of(m.content)})
    return out


def parse_openai_response(data: dict[str, Any]) -> tuple[AIMessage, Usage, str]:
    """Return (AIMessage, Usage, stop_reason) from a chat.completions response body."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    text = _text_of(content) if isinstance(content, list) else str(content or "")
    tool_calls: list[dict[str, Any]] = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
        else:
            args = raw_args or {}
        if not isinstance(args, dict):
            args = {"_raw": args}
        tool_calls.append({"name": str(fn.get("name", "")), "args": args, "id": str(tc.get("id") or f"call_{i}"), "type": "tool_call"})
    u = data.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=int(u.get("prompt_tokens", 0) or 0),
        output_tokens=int(u.get("completion_tokens", 0) or 0),
        cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
    )
    finish = str(choice.get("finish_reason") or "")
    stop_reason = OPENAI_FINISH_TO_STOP.get(finish, finish)
    meta: dict[str, Any] = {"stop_reason": stop_reason, "model_id": str(data.get("model", "")), "latency_ms": int(data.get("_latency_ms", 0))}
    if u.get("cost") is not None:
        meta["cost_usd"] = float(u["cost"])  # OpenRouter reports the real price; the budget uses it when present
    extra: dict[str, Any] = {}
    if msg.get("reasoning"):
        extra["reasoning"] = str(msg["reasoning"])
    ai = AIMessage(
        content=text,
        tool_calls=tool_calls,
        usage_metadata=UsageMetadata(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            input_token_details={"cache_read": usage.cache_read_tokens, "cache_creation": 0},  # type: ignore[typeddict-item]
        ),
        response_metadata=meta,
        additional_kwargs=extra,
    )
    return ai, usage, stop_reason


def summarize_raw_response(raw: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, int], str]:
    """(text, tool_calls, usage, stop_reason) for the human-readable log, for either wire format."""
    if "choices" in raw:
        ai, usage, stop = parse_openai_response(raw)
    else:
        ai, usage, stop = parse_converse_response(raw)
    calls = [{"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in ai.tool_calls]
    return _text_of(ai.content), calls, usage.as_dict(), stop


class OpenAICompatibleChat(BaseChatModel):
    """``POST {base_url}/chat/completions`` with ``Authorization: Bearer <key>`` (OpenRouter, vLLM, etc.)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    api_key: Optional[str] = Field(default=None, repr=False, exclude=True)
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 8192
    temperature: Optional[float] = None
    timeout_s: float = 180.0
    max_retries: int = 3
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_request_fields: dict[str, Any] = Field(default_factory=dict)
    request_hook: Optional[RequestHook] = Field(default=None, exclude=True)
    response_hook: Optional[ResponseHook] = Field(default=None, exclude=True)

    _client: Optional[httpx.Client] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "base_url": self.base_url}

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Optional[str] = None, **kwargs: Any) -> Any:
        specs = []
        for t in tools:
            if isinstance(t, dict):
                specs.append(tool_schema_to_openai(t))
            else:
                from langchain_core.utils.function_calling import convert_to_openai_tool

                specs.append(convert_to_openai_tool(t))
        return self.bind(tools=specs, **kwargs)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=20.0))
        return self._client

    def _send(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ModelError("No API key for the OpenAI-compatible provider: set OPENROUTER_API_KEY in the environment or .env.")
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http().post(self.endpoint, json=body, headers=headers)
            except httpx.TimeoutException as e:
                last_error = f"timeout after {self.timeout_s}s: {e}"
                BedrockConverseChat._backoff(attempt)
                continue
            except httpx.HTTPError as e:
                last_error = f"connection error: {e}"
                BedrockConverseChat._backoff(attempt)
                continue
            try:
                data = resp.json()
            except ValueError:
                data = None
            if resp.status_code == 200 and isinstance(data, dict) and "choices" in data:
                return data
            # OpenRouter reports upstream failures as HTTP 200 with an "error" object.
            err = (data or {}).get("error") if isinstance(data, dict) else None
            code = int((err or {}).get("code") or resp.status_code or 0) if isinstance(err, dict) else resp.status_code
            message = (err or {}).get("message") if isinstance(err, dict) else resp.text[:500]
            last_error = f"HTTP {code} from {self.base_url}: {message}"
            retryable = code in RETRYABLE_STATUS or code == 200
            if not retryable or attempt >= self.max_retries:
                hint = " (check OPENROUTER_API_KEY)" if code in (401, 403) else (" (check OPENROUTER_MODEL)" if code == 404 else "")
                raise ModelError(last_error + hint)
            BedrockConverseChat._backoff(attempt, retry_after=resp.headers.get("retry-after"))
        raise ModelError(f"Request failed after {self.max_retries + 1} attempts: {last_error}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        purpose = str(kwargs.pop("purpose", "agent"))
        tools = kwargs.pop("tools", None)
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages_to_openai(messages),
            "max_tokens": int(kwargs.pop("max_tokens", self.max_tokens)),
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        if self.temperature is not None:
            body["temperature"] = float(self.temperature)
        if stop:
            body["stop"] = list(stop)
        if self.extra_request_fields:
            body.update(self.extra_request_fields)
        if self.request_hook:
            self.request_hook(purpose, body)
        started = time.monotonic()
        data = self._send(body)
        latency_ms = int((time.monotonic() - started) * 1000)
        data["_latency_ms"] = latency_ms
        if self.response_hook:
            self.response_hook(purpose, data, latency_ms)
        ai, _usage, _stop = parse_openai_response(data)
        return ChatResult(generations=[ChatGeneration(message=ai)])


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written responses in order. For tests and offline dry runs.

    Each script entry is either an ``AIMessage`` or a dict::

        {"text": "...", "tool_calls": [{"name": "read_file", "args": {"path": "x"}}]}
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[Any] = Field(default_factory=list)
    model_id: str = "scripted"
    request_hook: Optional[RequestHook] = Field(default=None, exclude=True)
    response_hook: Optional[ResponseHook] = Field(default=None, exclude=True)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Optional[str] = None, **kwargs: Any) -> Any:
        return self.bind(tools=[t if isinstance(t, dict) else {"name": getattr(t, "name", str(t))} for t in tools], **kwargs)

    def _next(self) -> AIMessage:
        if self._cursor >= len(self.script):
            return AIMessage(content="(scripted model has no more responses) Done.")
        entry = self.script[self._cursor]
        self._cursor += 1
        if isinstance(entry, AIMessage):
            return entry
        tool_calls = []
        for i, tc in enumerate(entry.get("tool_calls", []) or []):
            tool_calls.append(
                {"name": tc["name"], "args": tc.get("args", {}), "id": tc.get("id") or f"call_{self._cursor}_{i}", "type": "tool_call"}
            )
        n_in = int(entry.get("input_tokens", 1000))
        n_out = int(entry.get("output_tokens", 100))
        return AIMessage(
            content=entry.get("text", ""),
            tool_calls=tool_calls,
            usage_metadata=UsageMetadata(input_tokens=n_in, output_tokens=n_out, total_tokens=n_in + n_out),
            response_metadata={"stop_reason": "tool_use" if tool_calls else "end_turn", "model_id": self.model_id, "latency_ms": 1},
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        purpose = str(kwargs.pop("purpose", "agent"))
        self._calls.append(list(messages))
        if self.request_hook:
            body = build_converse_body(messages, tools=kwargs.get("tools"))
            self.request_hook(purpose, body)
        ai = self._next()
        if self.response_hook:
            raw = {"output": {"message": {"role": "assistant", "content": [{"text": ai.content}] + [{"toolUse": {"toolUseId": tc["id"], "name": tc["name"], "input": tc["args"]}} for tc in ai.tool_calls]}}, "stopReason": ai.response_metadata.get("stop_reason"), "usage": {"inputTokens": (ai.usage_metadata or {}).get("input_tokens", 0), "outputTokens": (ai.usage_metadata or {}).get("output_tokens", 0)}}
            self.response_hook(purpose, raw, 1)
        return ChatResult(generations=[ChatGeneration(message=ai)])


def load_script(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("responses", [])
    return list(data)


def build_chat_model(
    config: Config,
    *,
    request_hook: Optional[RequestHook] = None,
    response_hook: Optional[ResponseHook] = None,
    script: Optional[list[Any]] = None,
) -> BaseChatModel:
    if config.provider == "mock" or script is not None:
        return ScriptedChatModel(script=list(script or []), request_hook=request_hook, response_hook=response_hook)
    if config.provider == "openrouter":
        return OpenAICompatibleChat(
            model_id=config.openrouter_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            max_tokens=config.max_output_tokens,
            temperature=config.temperature,
            timeout_s=config.request_timeout_s,
            max_retries=config.max_retries,
            extra_headers={"HTTP-Referer": "https://github.com/devagent-cli", "X-Title": "devagent"},
            extra_request_fields=dict(config.extra_request_fields),
            request_hook=request_hook,
            response_hook=response_hook,
        )
    if config.provider != "bedrock":
        raise ModelError(f"Unknown DEVAGENT_MODEL_PROVIDER '{config.provider}'. Use 'bedrock', 'openrouter' or 'mock'.")
    common: dict[str, Any] = dict(
        model_id=config.model_id,
        region=config.aws_region,
        max_tokens=config.max_output_tokens,
        temperature=config.temperature,
        timeout_s=config.request_timeout_s,
        max_retries=config.max_retries,
        prompt_cache=config.prompt_cache,
        extra_request_fields=dict(config.extra_request_fields),
        request_hook=request_hook,
        response_hook=response_hook,
    )
    if config.bearer_token:
        return BedrockConverseChat(bearer_token=config.bearer_token, **common)
    if config.aws_access_key_id and config.aws_secret_access_key:
        return BedrockBoto3Chat(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
            **common,
        )
    raise ModelError(
        "No Bedrock credentials found. Set BEDROCK_API_KEY (preferred) or AWS_BEARER_TOKEN_BEDROCK, "
        "or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (needs boto3). Use DEVAGENT_MODEL_PROVIDER=mock for an offline dry run."
    )


def iter_text_blocks(message: BaseMessage) -> Iterator[str]:
    text = _text_of(message.content)
    if text:
        yield text
