from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import ipaddress

from videoseal.tools.base import Tool, ToolOutput
from videoseal.utils.agent.env import env_bool_strict, env_flag, require_env, require_float_env, require_int_env
from videoseal.utils.agent.tool_agent_parsing import extract_between, parse_thinking, parse_tool_call_qwen
from videoseal.utils.agent.tool_schema import filter_args_for_forward, sanitize_args_against_schema
from videoseal.utils.agent.trajectory import Step, Trajectory
from videoseal.utils.api.mllm import MLLMClient
from videoseal.utils.video.io import get_video_duration
from videoseal.utils.video.time import sec_to_hhmmss


def _is_local_or_private_api_base(api_base: str) -> bool:
    base = (api_base or "").strip()
    if not base:
        return False
    try:
        u = urlparse(base if "://" in base else f"http://{base}")
        host = (u.hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_unspecified)


def _bool_env_or_default(name: str, default: bool) -> bool:
    if name in os.environ:
        return env_bool_strict(name, default=default)
    return default


def _system_prompt() -> str:
    return """You are a helper that answers multi-step video questions by sequentially invoking functions. Your ONLY job is to retrieve and refine candidate time spans; the final answer content must come from visual_inspect, not from your own imagination.

Tool outputs are lossy and may miss details. Prefer improving coverage by:
- trying different queries,
- inspecting new / non-overlapping time spans,
rather than re-running the same tool on the same spans.

If you cannot find strong anchors after several retrieval attempts, you may switch to an exploration mode:
- randomly sample a few large, non-overlapping time spans across the video,
- send them to visual_inspect to discover where relevant evidence might be.

Each turn, do exactly one of the following:

* If further clues and evidence are needed:
  Output EXACTLY one tool call:
  <tool_call>{"name":"tool_name","arguments":{}}</tool_call>

* If ready to finalize:
  You may enter this branch ONLY if the immediately previous turn called visual_inspect and its output contains a clear, reliable verdict.
  Output:
  <final>final answer</final>

Final Format
- Multiple-choice -> ONLY the uppercase option letter(s), for example C. No extra words.

Rules
- One tool call only per turn. Exactly one tool call per turn, except final turn (no tool call).
- You are a retriever, not an answerer: never invent answers; never override visual_inspect.
- You MUST NOT conclude from any non-visual_inspect tool alone.
- You may output <final> ONLY if the immediately previous turn was a visual_inspect call AND it returned a clear, reliable verdict.
- If the last tool call is NOT visual_inspect, you MUST NOT output <final>; keep searching and then call visual_inspect.
- The final answer must match the most recent reliable visual_inspect verdict (prefer the latest decisive visual_inspect when multiple exist).
- Do NOT re-run the same tool on the same time spans just to get more info. Instead, broaden coverage via different queries and new spans, or use exploration mode with large non-overlapping spans.
- If visual_inspect explicitly indicates that more search is needed, you MUST NOT finalize.
- A good pattern is to alternate tools, for example: visual_retrieve -> visual_inspect -> visual_retrieve with a different query -> visual_inspect on different time spans -> final.
- As soon as you have plausible spans, it is better to call visual_inspect again, possibly with more detailed context or slightly refined spans, than to keep doing blind retrieval. Multiple visual_inspect calls with different time spans are encouraged.
- Always zero-pad time fields (HH:MM:SS).
- Hard constraint: end_time must be strictly greater than start_time.

Tool Calling Conventions
- visual_retrieve: {"name":"visual_retrieve","arguments":{"query":"..."}}.
  Use free-form natural language or short visual phrases related to the question to retrieve visually/scene-relevant regions and coarse time anchors from the unified visual index.
- visual_inspect: {"name":"visual_inspect","arguments":{"spans":[{"start_time":"HH:MM:SS","end_time":"HH:MM:SS"}],"context":"Restate the original question and the visual sub-questions. Explain briefly why these spans were chosen, and provide a short look-for checklist of disambiguating cues to verify within these spans."}}.
  Constraints: each span must satisfy end_time > start_time.
"""


def _parse_final_answer(text: str) -> Optional[str]:
    fin = extract_between(text or "", "<final>", "</final>")
    if fin:
        return fin.strip()
    return None


def _tool_required_reminder(tool_names: List[str], *, api_messages: bool) -> str:
    tools = ", ".join(tool_names) if tool_names else "the available tools"
    if api_messages:
        return (
            "Your previous response was invalid: it did not contain a parsed tool call and did not contain <final>...</final>.\n"
            f"You must call one available tool now using the tool-calling interface: {tools}. Do not answer in plain text."
        )
    return (
        "Your previous response was invalid: it did not contain a parsed <tool_call>...</tool_call> and did not contain <final>...</final>.\n"
        f"You must use one available tool now: {tools}.\n"
        'Return exactly one tool call in this format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>. Do not answer in plain text.'
    )


class SimpleToolAgent:
    def __init__(
        self,
        *,
        tools: Dict[str, type[Tool]],
        llm_backend: str = "api",
        local_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.tools = tools
        self.system_prompt_override = system_prompt
        self.llm_backend = (os.getenv("AGENT_LLM_BACKEND") or llm_backend).lower()

        self.local_model = local_model
        if self.llm_backend == "vllm":
            self.local_model = self.local_model or os.getenv("VLLM_MODEL")
            if not self.local_model or not str(self.local_model).strip():
                raise RuntimeError("AGENT_LLM_BACKEND=vllm requires VLLM_MODEL (or pass local_model=...).")

        self.llm: Optional[MLLMClient] = None
        if self.llm_backend != "vllm":
            agent_api_model = require_env("AGENT_LLM_MODEL")
            agent_api_base = require_env("AGENT_LLM_API_BASE")
            agent_api_key = require_env("AGENT_LLM_API_KEY")
            agent_backend = (os.getenv("AGENT_MLLM_BACKEND") or "openai").strip()
            self.llm = MLLMClient(base_url=agent_api_base, api_key=agent_api_key, model=agent_api_model, backend=agent_backend)

        self._vllm = None
        self._tokenizer = None
        self._model_config = None

        self.traj: Optional[Trajectory] = None
        self._traj_start_monotonic: float | None = None

        # API path: for local/private vLLM, messages mode is typically safe and enables OpenAI tool calling.
        if "AGENT_API_USE_MESSAGES" in os.environ:
            self._api_use_messages = env_bool_strict("AGENT_API_USE_MESSAGES", default=False)
        else:
            base = os.getenv("AGENT_LLM_API_BASE", "")
            self._api_use_messages = _is_local_or_private_api_base(base)

    def _ensure_vllm(self) -> None:
        if self._vllm is not None:
            return
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("vLLM backend requires transformers.") from exc
        try:
            from vllm import LLM  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("vLLM backend requires vllm.") from exc

        model = str(self.local_model or "").strip()
        if not model:
            raise RuntimeError("Missing VLLM_MODEL for local vLLM backend.")

        tp = int(os.getenv("VLLM_TENSOR_PARALLEL", "1"))
        gpu_mem = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9"))
        self._vllm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._model_config = getattr(self._vllm, "llm_engine", None) and getattr(self._vllm.llm_engine, "model_config", None)

    def _inject_system_args(self, tool_name: str, model_args: Dict[str, Any]) -> Dict[str, Any]:
        injected = dict(model_args or {})
        # minimal injection for our two tools
        if tool_name == "visual_retrieve":
            injected.setdefault("video_id", getattr(self.traj, "video_id", "") if self.traj else "")
            injected.setdefault("original_question", getattr(self.traj, "question", "") if self.traj else "")
            vis = getattr(self, "_visual_index", None)
            if vis:
                injected.setdefault("index_path", vis)
        if tool_name == "visual_inspect":
            injected.setdefault("video_path", getattr(self, "_video_path", ""))
            injected.setdefault("questions", getattr(self.traj, "question", "") if self.traj else "")
            injected.setdefault("prompt_type", getattr(self, "_prompt_type", 0))
        return injected

    def _extract_tool_calls_from_last_response(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if self.llm is None:
            return {}, []
        data = self.llm.get_last_response()
        if not isinstance(data, dict):
            return {}, []
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {}, []
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return {}, []
        msg = choice0.get("message")
        if not isinstance(msg, dict):
            return {}, []
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        return msg, tool_calls

    def _save_trajectory(self, save_dir: str) -> None:
        if self.traj is None:
            return
        if self._traj_start_monotonic is not None:
            self.traj.elapsed_sec = round(time.monotonic() - float(self._traj_start_monotonic), 6)
        self.traj.finished_at = datetime.now(timezone.utc).isoformat()
        out_dir = Path(save_dir) / self.traj.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trajectory.json").write_text(json.dumps(asdict(self.traj), ensure_ascii=False, indent=2), encoding="utf-8")

    def _fallback_answer_from_last_tool(self) -> Optional[str]:
        if self.traj is None:
            return None
        for st in reversed(self.traj.steps):
            obs = st.observation
            if not isinstance(obs, dict) or obs.get("ok") is not True:
                continue
            tool_name = str(obs.get("name") or "").strip() or (st.action.get("name") if isinstance(st.action, dict) else None) or ""
            if tool_name != "visual_inspect":
                continue
            out = obs.get("output")
            if isinstance(out, dict):
                ans_text = str(out.get("answer") or "").strip()
            else:
                ans_text = str(out or "").strip()
            if not ans_text:
                continue
            return ans_text
        return None

    def _extract_answer_text(self, raw: object) -> str:
        if isinstance(raw, dict):
            raw = raw.get("answer")
        text = str(raw or "").strip()
        if not text:
            return ""
        unwrapped = _parse_final_answer(text)
        if unwrapped:
            return unwrapped.strip()
        return text

    def _forced_full_video_visual_inspect(self) -> Optional[str]:
        if self.traj is None:
            return None
        tool_cls = self.tools.get("visual_inspect")
        if tool_cls is None:
            return None

        video_path = str(getattr(self, "_video_path", "") or "").strip()
        if not video_path:
            raise RuntimeError("Forced visual_inspect fallback requires video_path.")

        duration = float(get_video_duration(video_path))
        if duration <= 0.0:
            raise RuntimeError(f"Forced visual_inspect fallback requires a valid video duration (video={video_path!r}).")

        max_total_raw = (os.getenv("INSPECT_MAX_TOTAL_IMAGES") or "").strip() or "64"
        max_total = int(max_total_raw)
        if max_total <= 0:
            raise ValueError(f"INSPECT_MAX_TOTAL_IMAGES must be a positive integer, got {max_total_raw!r}.")

        # Uniform sampling across the full video by setting fps so that duration * fps ~= max_total.
        fps = float(max_total) / float(duration)

        spans = [{"start_time": "00:00:00", "end_time": sec_to_hhmmss(int(math.ceil(duration)))}]

        args: Dict[str, Any] = {
            "video_path": video_path,
            "spans": spans,
            "questions": [self.traj.question],
            "prompt_type": int(getattr(self, "_prompt_type", 0)),
        }

        st = Step(chat_prompt=self.traj.question, model_response="", usage=None)
        st.action = {"name": "visual_inspect", "arguments": {"spans": spans}}

        restore_mode = os.getenv("VISUAL_INSPECT_PROMPT_MODE")
        restore_fps = os.getenv("INSPECT_FPS")
        try:
            os.environ["VISUAL_INSPECT_PROMPT_MODE"] = (os.getenv("AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE") or "mcq").strip() or "mcq"
            os.environ["INSPECT_FPS"] = str(fps)
            out = tool_cls()(**args)
        finally:
            if restore_mode is None:
                os.environ.pop("VISUAL_INSPECT_PROMPT_MODE", None)
            else:
                os.environ["VISUAL_INSPECT_PROMPT_MODE"] = restore_mode
            if restore_fps is None:
                os.environ.pop("INSPECT_FPS", None)
            else:
                os.environ["INSPECT_FPS"] = restore_fps

        ok = out.error is None
        payload = {
            "name": "visual_inspect",
            "ok": ok,
            "output": out.output if ok else None,
            "error": out.error,
            "forced": True,
            "mode": "full_video",
        }
        st.observation = payload
        self.traj.steps.append(st)

        if not ok:
            self.traj.note = f"max steps reached; forced full-video visual_inspect failed: {out.error}"
            return None

        return self._extract_answer_text(out.output)

    def run(
        self,
        *,
        question: str,
        uid: Optional[str] = None,
        time_reference: Optional[str] = None,
        video_id: str,
        video_path: str,
        visual_index: Optional[str] = None,
        groundtruth: Optional[str] = None,
        max_steps: int = 8,
        save_dir: Optional[str] = None,
        prompt_type: int = 0,
    ) -> Dict[str, Any]:
        self._traj_start_monotonic = time.monotonic()
        self._video_path = str(video_path or "")
        self._visual_index = str(visual_index or "") if visual_index else None
        self._prompt_type = int(prompt_type)

        system = (self.system_prompt_override or _system_prompt()).strip()
        tool_names = list((self.tools or {}).keys())

        tool_schemas = [cls().json for cls in (self.tools or {}).values()]
        tools_schema_text = json.dumps(tool_schemas, ensure_ascii=False)

        self.traj = Trajectory(
            system_prompt=system,
            tools_schema=tools_schema_text,
            question=str(question or "").strip(),
            time_reference=(str(time_reference).strip() if time_reference else None),
            uid=uid,
            groundtruth=(str(groundtruth).strip() if groundtruth else None),
            video_id=str(video_id or ""),
        )

        # History
        tool_history_blocks: List[str] = []
        api_history: List[Dict[str, Any]] = [{"role": "system", "content": system}]

        user_base = str(question or "").strip()
        if time_reference:
            user_base = f"[Time reference: {time_reference}]\n{user_base}"
        if self.llm_backend != "vllm" and self._api_use_messages:
            api_history.append({"role": "user", "content": user_base})

        for step_idx in range(1, int(max_steps) + 1):
            if self.llm_backend == "vllm":
                self._ensure_vllm()
                from vllm import SamplingParams  # type: ignore

                msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
                if tool_history_blocks:
                    msgs.append({"role": "user", "content": "\n".join(tool_history_blocks) + "\n\n" + user_base})
                else:
                    msgs.append({"role": "user", "content": user_base})
                prompt_text = self._tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # type: ignore[union-attr]
                max_new = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
                temperature = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
                outputs = self._vllm.generate([prompt_text], SamplingParams(max_tokens=max_new, temperature=temperature, top_p=0.95, n=1))
                resp_text = outputs[0].outputs[0].text
                usage = None
            else:
                if self.llm is None:
                    raise RuntimeError("Agent API client is not initialized.")

                tool_choice = None
                if max_steps > 1 and step_idx == max_steps - 1 and env_flag("AGENT_FORCE_LAST_STEP_VISUAL_INSPECT", default=False):
                    tool_choice = {"type": "function", "function": {"name": "visual_inspect"}}

                if self._api_use_messages:
                    resp_text = self.llm.generate_chat(
                        api_history,
                        response_json=False,
                        max_tokens=require_int_env("AGENT_LLM_MAX_TOKENS"),
                        temperature=require_float_env("AGENT_LLM_TEMPERATURE"),
                        timeout=require_float_env("AGENT_LLM_TIMEOUT"),
                        tools=tool_schemas,
                        tool_choice=(tool_choice or "auto"),
                    )
                    usage = self.llm.get_last_usage() or None
                    msg, tool_calls = self._extract_tool_calls_from_last_response()
                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.get("content", resp_text)}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    api_history.append(assistant_msg)

                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                            tool_name = str(fn.get("name") or "").strip()
                            call_id = str(tc.get("id") or str(uuid.uuid4()))
                            arg_s = fn.get("arguments")
                            if isinstance(arg_s, dict):
                                args = arg_s
                            elif isinstance(arg_s, str):
                                if not arg_s.strip():
                                    args = {}
                                else:
                                    try:
                                        raw_args = json.loads(arg_s)
                                    except json.JSONDecodeError as exc:
                                        raise RuntimeError(f"Invalid tool call arguments JSON for {tool_name!r}: {exc}") from exc
                                    if not isinstance(raw_args, dict):
                                        raise RuntimeError(
                                            f"Tool call arguments for {tool_name!r} must be a JSON object, got: {type(raw_args).__name__}"
                                        )
                                    args = raw_args
                            else:
                                args = {}

                            st = Step(chat_prompt=user_base, model_response=json.dumps({"tool_calls": tool_calls}, ensure_ascii=False), usage=usage)
                            st.thinking = parse_thinking(resp_text or "")
                            st.action = {"name": tool_name, "arguments": args}

                            tool_cls = self.tools.get(tool_name)
                            if tool_cls is None:
                                payload = {"name": tool_name, "ok": False, "output": None, "error": f"unknown tool: {tool_name}"}
                                api_history.append({"role": "tool", "tool_call_id": call_id, "content": f"Error: {payload['error']}"})
                                st.observation = payload
                                self.traj.steps.append(st)
                                continue

                            tool = tool_cls()
                            injected = self._inject_system_args(tool_name, args)
                            injected = sanitize_args_against_schema(tool, injected)
                            injected = filter_args_for_forward(tool, injected)

                            # Optional prompt-mode override for last-step visual_inspect.
                            _set_vis_mode = (
                                tool_name == "visual_inspect"
                                and max_steps > 1
                                and step_idx == max_steps - 1
                                and env_flag("AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK", default=False)
                            )
                            _restore_vis_mode = None
                            if _set_vis_mode:
                                _restore_vis_mode = os.getenv("VISUAL_INSPECT_PROMPT_MODE")
                                os.environ["VISUAL_INSPECT_PROMPT_MODE"] = (os.getenv("AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE") or "mcq").strip()
                            try:
                                out = tool(**injected)
                            finally:
                                if _set_vis_mode:
                                    if _restore_vis_mode is None:
                                        os.environ.pop("VISUAL_INSPECT_PROMPT_MODE", None)
                                    else:
                                        os.environ["VISUAL_INSPECT_PROMPT_MODE"] = _restore_vis_mode

                            ok = out.error is None
                            payload = {"name": tool_name, "ok": ok, "output": out.output if ok else None, "error": out.error, "metadata": out.metadata}
                            visible_txt = out.to_string()
                            api_history.append({"role": "tool", "tool_call_id": call_id, "content": visible_txt})
                            st.observation = payload
                            self.traj.steps.append(st)

                        # continue after tool execution
                        continue

                    # No OpenAI tool calls. Accept only a tagged final answer; otherwise remind and retry.
                    st = Step(chat_prompt=user_base, model_response=resp_text or "", usage=usage)
                    st.thinking = parse_thinking(resp_text or "")
                    self.traj.steps.append(st)
                    ans = _parse_final_answer(resp_text or "")
                    if ans:
                        self.traj.answer = ans.strip()
                        if save_dir:
                            self._save_trajectory(save_dir)
                        return {"final": self.traj.answer, "answer": self.traj.answer, "steps": step_idx, "run_id": self.traj.run_id}
                    self.traj.note = "invalid response; reminded model to use a tool"
                    api_history.append(
                        {
                            "role": "user",
                            "content": _tool_required_reminder(tool_names, api_messages=True),
                        }
                    )
                    continue

                # Legacy mode: tag-based tool calling (<tool_call>...</tool_call>).
                history_txt = "\n".join(tool_history_blocks) if tool_history_blocks else ""
                tools_block = "<tools>\n" + "\n".join(json.dumps(s, ensure_ascii=False) for s in tool_schemas) + "\n</tools>"
                if history_txt:
                    user_prompt = tools_block + "\n\n" + history_txt + "\n\n" + user_base
                else:
                    user_prompt = tools_block + "\n\n" + user_base
                resp_text = self.llm.generate_text(
                    user_prompt,
                    response_json=False,
                    system_message=system.strip(),
                    max_tokens=require_int_env("AGENT_LLM_MAX_TOKENS"),
                    temperature=require_float_env("AGENT_LLM_TEMPERATURE"),
                    timeout=require_float_env("AGENT_LLM_TIMEOUT"),
                )
                usage = self.llm.get_last_usage() or None

            # Shared parsing for vllm backend and legacy API backend.
            resp_text = str(resp_text or "")
            st = Step(chat_prompt=user_base, model_response=resp_text, usage=usage)
            st.thinking = parse_thinking(resp_text)

            call = parse_tool_call_qwen(resp_text)
            if call:
                tool_name = str(call.get("name") or "").strip()
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                st.action = {"name": tool_name, "arguments": args}

                tool_cls = self.tools.get(tool_name)
                if tool_cls is None:
                    payload = {"name": tool_name, "ok": False, "output": None, "error": f"unknown tool: {tool_name}"}
                    st.observation = payload
                    self.traj.steps.append(st)
                    tool_history_blocks.append(f"<tool_response>{json.dumps(payload, ensure_ascii=False)}</tool_response>")
                    continue

                tool = tool_cls()
                injected = self._inject_system_args(tool_name, args)
                injected = sanitize_args_against_schema(tool, injected)
                injected = filter_args_for_forward(tool, injected)
                out = tool(**injected)
                ok = out.error is None
                payload = {"name": tool_name, "ok": ok, "output": out.output if ok else None, "error": out.error, "metadata": out.metadata}
                st.observation = payload
                self.traj.steps.append(st)
                tool_history_blocks.append(f"<tool_response>{ToolOutput(name=tool_name, output=payload.get('output') if ok else out.to_string()).to_string()}</tool_response>")
                continue

            self.traj.steps.append(st)
            ans = _parse_final_answer(resp_text)
            if ans:
                self.traj.answer = ans.strip()
                if save_dir:
                    self._save_trajectory(save_dir)
                return {"final": self.traj.answer, "answer": self.traj.answer, "steps": step_idx, "run_id": self.traj.run_id}
            self.traj.note = "invalid response; reminded model to use a tool"
            tool_history_blocks.append(
                "<format_feedback>"
                + _tool_required_reminder(tool_names, api_messages=False)
                + "</format_feedback>"
            )
            continue

        if env_flag("AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK", default=False):
            forced = self._forced_full_video_visual_inspect()
            if forced:
                self.traj.answer = forced
                self.traj.note = "max steps reached; forced full-video visual_inspect fallback"
                if save_dir:
                    self._save_trajectory(save_dir)
                return {"final": self.traj.answer, "answer": self.traj.answer, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}
            if save_dir:
                self._save_trajectory(save_dir)
            return {"answer": None, "steps": int(max_steps), "note": self.traj.note or "max steps reached; forced visual_inspect fallback failed", "run_id": self.traj.run_id}

        fallback = self._fallback_answer_from_last_tool()
        if fallback:
            self.traj.answer = str(fallback).strip()
            self.traj.note = "max steps reached (fallback from last visual_inspect)"
            if save_dir:
                self._save_trajectory(save_dir)
            return {"final": self.traj.answer, "answer": self.traj.answer, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}

        self.traj.note = "max steps reached"
        if save_dir:
            self._save_trajectory(save_dir)
        return {"answer": None, "steps": int(max_steps), "note": self.traj.note, "run_id": self.traj.run_id}
