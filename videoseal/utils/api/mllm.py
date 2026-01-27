from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .error_utils import extract_request_id
from .router import build_adapter
from .uploader import build_uploader


def _b64_of_image_bytes(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def to_data_url(b64: str, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{b64}"


class MLLMClient:
    """Unified multimodal client with pluggable backends (OpenAI-compatible by default)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.model = model or os.getenv("MLLM_MODEL", "gpt-4o-mini")
        self.backend = (backend or os.getenv("MLLM_BACKEND", "openai")).lower()

        prefer_url = bool(int(os.getenv("MLLM_UPLOAD_PREFER_URL", os.getenv("UPLOAD_PREFER_URL", "0"))))
        uploader = build_uploader(None)

        default_base = os.getenv("MLLM_API_BASE")
        default_key: Optional[str] = None
        if self.backend in ("ark", "volc", "bytedance"):
            default_key = os.getenv("ARK_API_KEY") or os.getenv("SEED_ARK_API_KEY")
            default_base = base_url or None
        elif self.backend in ("dashscope", "qwen", "qwen_mm"):
            default_key = os.getenv("DASHSCOPE_API_KEY")
            default_base = base_url or None
        else:
            default_base = base_url or default_base
            if (default_base and "dashscope.aliyuncs.com" in default_base and os.getenv("DASHSCOPE_API_KEY")):
                default_key = os.getenv("DASHSCOPE_API_KEY")
            else:
                default_key = os.getenv("MLLM_API_KEY")

        self._adapter = build_adapter(
            self.backend,
            base_url=default_base,
            api_key=api_key or default_key,
            uploader=uploader,
            prefer_url=prefer_url,
        )

        self._temperature = float(os.getenv("MLLM_TEMPERATURE", "0.0"))
        try:
            self._max_tokens = int(os.getenv("MLLM_MAX_TOKENS", "800"))
        except Exception:
            self._max_tokens = 800
        try:
            self._timeout = int(os.getenv("MLLM_TIMEOUT", "60"))
        except Exception:
            self._timeout = 60
        try:
            self._retry_times = int(os.getenv("MLLM_RETRY_TIMES", "5"))
        except Exception:
            self._retry_times = 5
        try:
            self._retry_delay = float(os.getenv("MLLM_RETRY_DELAY", "5"))
        except Exception:
            self._retry_delay = 5.0

    def _with_retry(self, fn):
        last_exc: Optional[Exception] = None
        for i in range(max(1, self._retry_times)):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if i < self._retry_times - 1:
                    try:
                        base_url = getattr(self._adapter, "base_url", None)
                        rid = extract_request_id(e)
                        rid_part = f" request_id={rid}" if rid else ""
                        print(
                            f"[MLLM][retry {i+1}/{self._retry_times}] backend={self.backend} base={base_url} model={self.model} "
                            f"{type(e).__name__}{rid_part}: {e} — sleep {self._retry_delay}s",
                            flush=True,
                        )
                    except Exception:
                        pass
                    time.sleep(self._retry_delay)
                    continue
                raise last_exc

    def _normalize_usage(self, usage: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not usage:
            return None, None, None
        in_tok = usage.get("prompt_tokens") or usage.get("input_tokens")
        out_tok = usage.get("completion_tokens") or usage.get("output_tokens")
        tot = usage.get("total_tokens") or usage.get("total") or None
        try:
            if tot is None and in_tok is not None and out_tok is not None:
                tot = int(in_tok) + int(out_tok)
        except Exception:
            tot = None
        try:
            in_tok = int(in_tok) if in_tok is not None else None
        except Exception:
            in_tok = None
        try:
            out_tok = int(out_tok) if out_tok is not None else None
        except Exception:
            out_tok = None
        try:
            tot = int(tot) if tot is not None else None
        except Exception:
            tot = None
        return in_tok, out_tok, tot

    def _maybe_with_usage(self, text: Any, *, with_usage: bool) -> Any:
        if not with_usage:
            return text
        usage = getattr(self._adapter, "last_usage", None)
        in_tok, out_tok, tot = self._normalize_usage(usage if isinstance(usage, dict) else None)
        return {"text": text, "usage": {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": tot}}

    def generate_text(self, prompt: str, *, response_json: bool = False, with_usage: bool = False, **kwargs) -> Any:
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        kwargs.setdefault("timeout", self._timeout)
        text = self._with_retry(lambda: self._adapter.generate([{"type": "text", "text": prompt}], self.model, response_json=response_json, **kwargs))
        return self._maybe_with_usage(text, with_usage=with_usage)

    def generate_chat(self, messages: List[Dict[str, Any]], *, response_json: bool = False, with_usage: bool = False, **kwargs) -> Any:
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        kwargs.setdefault("timeout", self._timeout)
        if not hasattr(self._adapter, "generate_chat"):
            raise NotImplementedError(f"Adapter {type(self._adapter).__name__} does not support generate_chat().")
        text = self._with_retry(lambda: self._adapter.generate_chat(messages, self.model, response_json=response_json, **kwargs))
        return self._maybe_with_usage(text, with_usage=with_usage)

    def get_last_usage(self) -> Optional[Dict[str, Any]]:
        try:
            return getattr(self._adapter, "last_usage", None)
        except Exception:
            return None

    def get_last_response(self) -> Optional[Dict[str, Any]]:
        try:
            v = getattr(self._adapter, "last_response", None)
            return v if isinstance(v, dict) else None
        except Exception:
            return None

    def generate_images_b64(self, images_b64: List[str], prompt: str, *, response_json: bool = False, with_usage: bool = False, **kwargs) -> Any:
        content: List[Dict[str, Any]] = [{"type": "image_b64", "b64": b} for b in images_b64]
        content.append({"type": "text", "text": prompt})
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        kwargs.setdefault("timeout", self._timeout)
        text = self._with_retry(lambda: self._adapter.generate(content, self.model, response_json=response_json, **kwargs))
        return self._maybe_with_usage(text, with_usage=with_usage)

    def generate_images_paths(self, image_paths: List[str], prompt: str, *, response_json: bool = False, timestamps: Optional[List[float]] = None, with_usage: bool = False, **kwargs) -> Any:
        content: List[Dict[str, Any]] = []

        def _ts_from_name(path: str) -> Optional[float]:
            import re

            m = re.search(r"frame_(\\d+\\.\\d+)", path)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None

        for i, p in enumerate(image_paths):
            ts = None
            if timestamps is not None and i < len(timestamps):
                ts = timestamps[i]
            else:
                ts = _ts_from_name(p)
            if ts is not None:
                content.append({"type": "text", "text": f"[{ts:.3f} second]"})
            content.append({"type": "image_path", "path": p})
        content.append({"type": "text", "text": prompt})
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        kwargs.setdefault("timeout", self._timeout)
        text = self._with_retry(lambda: self._adapter.generate(content, self.model, response_json=response_json, **kwargs))
        return self._maybe_with_usage(text, with_usage=with_usage)

    def generate_caption(self, images_b64: List[str], prompt: str, response_json: bool = True, timeout: int = 60, with_usage: bool = False) -> Any:
        return self.generate_images_b64(images_b64, prompt, response_json=response_json, timeout=timeout, with_usage=with_usage)
