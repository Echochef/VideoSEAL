from __future__ import annotations

"""Provider adapters for a unified multimodal chat interface.

Adapters standardize payload building across providers:
  - OpenAI-compatible chat.completions (images via data URL)
  - DashScope Qwen MultiModal
  - Volcengine Ark chat.completions

Each adapter implements:
  - generate(content, model, **kwargs) -> str
  - generate_chat(messages, model, **kwargs) -> str
"""

import base64
import os
from typing import Any, Dict, List, Optional

from .error_utils import extract_request_id_from_headers


def _to_data_url(b64: str, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{b64}"


class BaseAdapter:
    def __init__(self, *, uploader=None, prefer_url: bool = False) -> None:
        self.uploader = uploader
        self.prefer_url = prefer_url
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_response: Optional[Dict[str, Any]] = None

    def generate(self, content: List[Dict[str, Any]], model: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> str:
        raise NotImplementedError


class OpenAIChatAdapter(BaseAdapter):
    """OpenAI-compatible chat.completions adapter."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, *, uploader=None, prefer_url: bool = False):
        super().__init__(uploader=uploader, prefer_url=prefer_url)
        self.base_url = base_url or os.getenv("MLLM_API_BASE", "http://localhost:23333/v1")
        if (not api_key) and self.base_url and ("dashscope.aliyuncs.com" in self.base_url):
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("MLLM_API_KEY", "")
        self.api_key = api_key or os.getenv("MLLM_API_KEY", "")
        self._raw_base_endpoint = bool(os.getenv("OPENAI_FORCE_RAW_BASE", "0").lower() in ("1", "true", "yes", "on"))

    def _build_user_content(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for it in content:
            t = it.get("type")
            if t == "text":
                items.append({"type": "text", "text": str(it.get("text", ""))})
            elif t == "image_b64":
                items.append({"type": "image_url", "image_url": {"url": _to_data_url(str(it.get("b64", ""))), "detail": "low"}})
            elif t == "image_url":
                items.append({"type": "image_url", "image_url": {"url": str(it.get("url", "")), "detail": "low"}})
            elif t == "image_path":
                path = str(it.get("path"))
                if self.prefer_url and self.uploader is not None:
                    url = self.uploader.upload_file(path)
                    items.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
                else:
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    items.append({"type": "image_url", "image_url": {"url": _to_data_url(b64), "detail": "low"}})
            elif t == "video_frames":
                for fp in it.get("frames") or []:
                    with open(fp, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    items.append({"type": "image_url", "image_url": {"url": _to_data_url(b64), "detail": "low"}})
            elif t in ("video_path", "video_url"):
                raise NotImplementedError("OpenAI chat.completions does not support raw video; pass frames instead")
        return items

    def generate_chat(self, messages: List[Dict[str, Any]], model: str, **kwargs) -> str:
        import requests

        temperature = float(kwargs.get("temperature", 0.0))
        max_tok_val = int(kwargs.get("max_tokens", 800))
        timeout = float(kwargs.get("timeout", 60))

        model_lower = (model or "").lower()
        env_max_field = (os.getenv("OPENAI_MAX_TOKENS_FIELD", "") or "").strip()
        if env_max_field:
            max_field = env_max_field
        elif model_lower.startswith(("gpt-5", "o1", "o3", "o4")):
            max_field = "max_completion_tokens"
        else:
            max_field = "max_tokens"

        # Normalize messages for OpenAI-compatible servers.
        norm_msgs: List[Dict[str, Any]] = []
        for i, m in enumerate(messages or []):
            role = str(m.get("role") or "").strip()
            if not role:
                continue
            content = m.get("content") if ("content" in m) else str(m.get("text") or "")
            msg_obj: Dict[str, Any] = {"role": role, "content": content}
            if role == "assistant" and "tool_calls" in m:
                msg_obj["tool_calls"] = m.get("tool_calls")
            if role == "tool" and "tool_call_id" not in m:
                msg_obj["tool_call_id"] = f"tool_{i}"
            elif "tool_call_id" in m:
                msg_obj["tool_call_id"] = m.get("tool_call_id")
            norm_msgs.append(msg_obj)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": norm_msgs,
            "temperature": temperature,
            max_field: max_tok_val,
        }
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs.get("tools")
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs.get("tool_choice")
        if kwargs.get("parallel_tool_calls") is not None:
            payload["parallel_tool_calls"] = bool(kwargs.get("parallel_tool_calls"))
        if kwargs.get("top_p") is not None:
            payload["top_p"] = float(kwargs.get("top_p"))
        if kwargs.get("top_k") is not None:
            payload["top_k"] = int(kwargs.get("top_k"))
        if kwargs.get("n") is not None:
            payload["n"] = int(kwargs.get("n"))
        if kwargs.get("stop") is not None:
            payload["stop"] = kwargs.get("stop")
        if kwargs.get("response_json"):
            payload["response_format"] = {"type": "json_object"}

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.base_url and "dashscope.aliyuncs.com" in self.base_url:
            headers["X-DashScope-DataInspection"] = '{"input":"disable","output":"disable"}'
        if os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")  # noqa: N815
        if os.getenv("OPENROUTER_X_TITLE"):
            headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE")

        base = self.base_url.rstrip("/") if self.base_url else ""
        url_ep = base
        if not self._raw_base_endpoint:
            url_ep = f"{base}/chat/completions" if not base.endswith("/chat/completions") else base
        if base.endswith("/responses"):
            url_ep = base

        try:
            resp = requests.post(url_ep, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            try:
                if getattr(e, "response", None) is not None:
                    rid = extract_request_id_from_headers(getattr(e.response, "headers", None))
                    rid_part = f" request_id={rid}" if rid else ""
                    raise RuntimeError(f"OpenAI-compatible error {e.response.status_code}{rid_part}: {e.response.text[:5000]}") from e
            except Exception:
                pass
            raise

        self.last_response = data if isinstance(data, dict) else {"raw": data}
        try:
            self.last_usage = data.get("usage") if isinstance(data, dict) else None
        except Exception:
            self.last_usage = None

        try:
            choice0 = (data.get("choices") or [])[0] if isinstance(data, dict) else None
            msg = (choice0 or {}).get("message") if isinstance(choice0, dict) else None
            if isinstance(msg, dict):
                return str(msg.get("content") or "")
        except Exception:
            pass
        return ""

    def generate(self, content: List[Dict[str, Any]], model: str, **kwargs) -> str:
        system_prompt = str(kwargs.pop("system_prompt", "") or kwargs.pop("system_message", "") or "").strip()
        user_content = self._build_user_content(content)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return self.generate_chat(messages, model, **kwargs)


class ArkChatAdapter(OpenAIChatAdapter):
    """Volc Ark adapter (OpenAI-compatible schema with different auth)."""

    def __init__(self, api_key: Optional[str] = None, *, uploader=None, prefer_url: bool = True):
        base_url = os.getenv("ARK_API_BASE") or os.getenv("MLLM_API_BASE") or "https://ark.cn-beijing.volces.com/api/v3"
        api_key = api_key or os.getenv("ARK_API_KEY") or os.getenv("SEED_ARK_API_KEY") or ""
        super().__init__(base_url=base_url, api_key=api_key, uploader=uploader, prefer_url=prefer_url)


def build_adapter(backend: str, *, base_url: Optional[str], api_key: Optional[str], uploader=None, prefer_url: bool = False) -> BaseAdapter:
    b = (backend or "").lower()
    if b in ("openai", "openai_compat", "oai"):
        return OpenAIChatAdapter(base_url=base_url, api_key=api_key, uploader=uploader, prefer_url=prefer_url)
    if b in ("ark", "volc", "bytedance"):
        return ArkChatAdapter(api_key=api_key, uploader=uploader, prefer_url=True)
    if b in ("dashscope", "qwen"):
        # DashScope compatible-mode is still OpenAI schema; use OpenAI adapter with special headers.
        return OpenAIChatAdapter(base_url=base_url, api_key=api_key, uploader=uploader, prefer_url=prefer_url)
    return OpenAIChatAdapter(base_url=base_url, api_key=api_key, uploader=uploader, prefer_url=prefer_url)
