import json
import os
import queue
import warnings
from pathlib import Path
from typing import Any

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool


class ToolEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_hhmmss_to_seconds(value: Any) -> float | None:
        try:
            s = str(value or "").strip()
            if not s:
                return None
            parts = s.split(":")
            if len(parts) == 3:
                h, m, sec = parts
            elif len(parts) == 2:
                h = "0"
                m, sec = parts
            else:
                h = "0"
                m = "0"
                sec = parts[0]
            return float(h) * 3600.0 + float(m) * 60.0 + float(sec)
        except Exception:
            return None

    @classmethod
    def _spans_overlap(cls, a: dict[str, Any], b: dict[str, Any]) -> bool:
        a0 = cls._parse_hhmmss_to_seconds(a.get("start_time"))
        a1 = cls._parse_hhmmss_to_seconds(a.get("end_time"))
        b0 = cls._parse_hhmmss_to_seconds(b.get("start_time"))
        b1 = cls._parse_hhmmss_to_seconds(b.get("end_time"))
        if None in (a0, a1, b0, b1):
            return False
        lo_a, hi_a = (a0, a1) if a0 <= a1 else (a1, a0)
        lo_b, hi_b = (b0, b1) if b0 <= b1 else (b1, b0)
        return not (hi_a <= lo_b or hi_b <= lo_a)

    def _collect_previous_visual_inspect_spans(self) -> list[dict[str, str]]:
        spans: list[dict[str, str]] = []
        for item in self.transcript:
            if not isinstance(item, dict):
                continue
            tool_calls = item.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                if call.get("function", {}).get("name") != "visual_inspect":
                    continue
                args = call.get("function", {}).get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = None
                if not isinstance(args, dict):
                    continue
                sp = args.get("spans")
                if not isinstance(sp, list):
                    continue
                for s in sp:
                    if not isinstance(s, dict):
                        continue
                    st = s.get("start_time")
                    et = s.get("end_time")
                    if not st or not et:
                        continue
                    spans.append({"start_time": str(st), "end_time": str(et)})
        return spans

    def _apply_last_step_visual_inspect_fallback(self, tool_calls: list[dict[str, Any]]) -> None:
        if not self._enable_last_step_visual_inspect_fallback:
            return
        if self.max_steps < 2:
            return
        if self.step_count != self.max_steps - 1:
            return

        prev_spans = self._collect_previous_visual_inspect_spans()
        prompt_file = str(self._last_step_visual_inspect_prompt_file)

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if fn.get("name") != "visual_inspect":
                continue

            raw = fn.get("arguments", "{}")
            args = None
            if isinstance(raw, str):
                try:
                    args = json.loads(raw)
                except Exception:
                    args = None
            elif isinstance(raw, dict):
                args = dict(raw)

            if not isinstance(args, dict):
                continue

            cur_spans = args.get("spans")
            if isinstance(cur_spans, list):
                merged: list[dict[str, Any]] = []
                for s in cur_spans:
                    if isinstance(s, dict) and s.get("start_time") and s.get("end_time"):
                        merged.append({"start_time": str(s["start_time"]), "end_time": str(s["end_time"])})
                for s in prev_spans:
                    if len(merged) >= 10:
                        break
                    if any(self._spans_overlap(s, m) for m in merged):
                        continue
                    merged.append({"start_time": str(s["start_time"]), "end_time": str(s["end_time"])})
                if merged:
                    args["spans"] = merged[:10]

            # Hidden per-call override: force a different visual_inspect prompt template on this step.
            args["_prompt_file"] = prompt_file

            fn["arguments"] = json.dumps(args, ensure_ascii=False)
            tc["function"] = fn

    def _redact_task_for_agent(self, task):
        """Strip ground-truth style labels before exposing the task to the agent."""
        if not isinstance(task, dict):
            return task

        def _strip(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    key_lower = k.lower()
                    if key_lower in {"ground_truth", "answer", "answers", "label", "labels"}:
                        continue
                    out[k] = _strip(v)
                return out
            if isinstance(obj, list):
                return [_strip(x) for x in obj]
            return obj

        return _strip(task)

    def __init__(
        self,
        task: dict | None = None,
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        reward_fn: RewardFunction | None = None,
        max_steps=10,
        task_defaults: dict[str, Any] | None = None,
    ):
        """
        Initialize the ToolEnvironment.

        Args:
            task: Task information for the environment.
            tools: List of tool names to look up in the registry (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.step_count = 0
        self.max_steps = max_steps

        # Initialize MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        self.task_defaults: dict[str, Any] = dict(task_defaults) if isinstance(task_defaults, dict) else {}
        self.task = task if task is not None else (dict(self.task_defaults) if self.task_defaults else None)
        self.transcript: list[dict[str, Any]] = []
        self.tool_calls_total: int = 0
        self._enable_last_step_visual_inspect_fallback = self._env_flag("RLLM_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK", default=False)
        default_prompt_file = Path(__file__).resolve().parents[2] / "agents" / "visual_inspect_last_step_mcq_prompt.txt"
        self._last_step_visual_inspect_prompt_file = Path(os.environ.get("RLLM_LAST_STEP_VISUAL_INSPECT_PROMPT_FILE", str(default_prompt_file)))
        if self._enable_last_step_visual_inspect_fallback and not self._last_step_visual_inspect_prompt_file.exists():
            raise FileNotFoundError(
                "RLLM_LAST_STEP_VISUAL_INSPECT_PROMPT_FILE not found: "
                f"{self._last_step_visual_inspect_prompt_file}"
            )
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward
        else:
            self.reward_fn = reward_fn

    def reset(self, task: dict | None = None):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        self.transcript = []
        self.tool_calls_total = 0
        if task is not None:
            if self.task_defaults and isinstance(task, dict):
                merged = dict(self.task_defaults)
                merged.update(task)
                self.task = merged
            else:
                self.task = task

        return self._redact_task_for_agent(self.task), {}

    def step(self, action: list[dict] | str | dict):
        """
        Take a step in the environment based on the action.

        Args:
            actions: List containing a single action string from the agent

        Returns:
            next_observations, rewards, terminateds, infos
        """
        if action is None:
            action = []

        if isinstance(action, dict):
            action = [action]
        self.step_count += 1
        if isinstance(action, list):
            self.tool_calls_total += len(action)

        reward = 0
        terminal_answer_from_spans = False
        answer_from_spans_call_id = ""
        # Check if we should terminate
        done = self.step_count >= self.max_steps or isinstance(action, str)
        # Check if action contains a "finish" tool call
        if isinstance(action, list) and action:
            has_finish = False
            for tool_call in action:
                name = tool_call.get("function", {}).get("name")
                if name == "finish":
                    has_finish = True
                elif name == "answer_from_spans":
                    terminal_answer_from_spans = True
                    answer_from_spans_call_id = str(tool_call.get("id") or "")
            if has_finish:
                done = True
            # Allow answer_from_spans to run even on the last allowed step.
            if terminal_answer_from_spans and not isinstance(action, str):
                done = False
        if done:
            # Cannot find tool calls which means the agent is not using the tool and is done.
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                # Find the finish tool call
                finish_action = None
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    # arguments may be a dict or a JSON-encoded string; normalize to dict.
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                            if isinstance(parsed, dict):
                                arguments = parsed
                        except Exception:
                            pass
                    llm_response = arguments.get("response", "") if isinstance(arguments, dict) else ""
                else:
                    # No finish tool call found, use the action itself
                    llm_response = str(action)

            try:
                if isinstance(llm_response, str) and llm_response.strip():
                    self.transcript.append({"assistant_response": llm_response})
            except Exception:
                pass

            task_info = self.task if self.task is not None else {}
            try:
                task_info = dict(task_info)
                task_info["transcript"] = list(self.transcript)
                task_info["tool_calls_total"] = int(self.tool_calls_total)
            except Exception:
                pass
            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata, "is_correct": reward_output.is_correct}

        tool_calls = action
        assert isinstance(tool_calls, list)
        # "finish" is an environment control action, not an actual tool to execute.
        if terminal_answer_from_spans:
            tool_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") != "finish"]
        tool_calls_exec = tool_calls
        max_calls_raw = os.environ.get("RLLM_MAX_TOOL_CALLS_PER_STEP", "16")
        try:
            max_calls = int(str(max_calls_raw).strip())
        except Exception:
            max_calls = 16
        excess_calls: list[dict[Any, Any]] = []
        if max_calls > 0 and len(tool_calls_exec) > max_calls:
            excess_calls = list(tool_calls_exec[max_calls:])
            tool_calls_exec = list(tool_calls_exec[:max_calls])

        self._apply_last_step_visual_inspect_fallback(tool_calls_exec)

        # Last-step visual_inspect fallback: force the MCQ-style visual_inspect prompt mode on the
        # penultimate step (max_steps - 1), matching Videoseal's inference runner defaults.
        _set_vis_mode = (
            self._enable_last_step_visual_inspect_fallback
            and self.max_steps > 1
            and self.step_count == self.max_steps - 1
            and any((tc.get("function") or {}).get("name") == "visual_inspect" for tc in tool_calls_exec)
        )
        _restore_vis_mode = None
        if _set_vis_mode:
            _restore_vis_mode = os.getenv("VISUAL_INSPECT_PROMPT_MODE")
            mode = (os.getenv("RLLM_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE") or os.getenv("AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE") or "mcq").strip() or "mcq"
            os.environ["VISUAL_INSPECT_PROMPT_MODE"] = mode
        try:
            tool_outputs = self._execute_tool_calls(tool_calls_exec)
        finally:
            if _set_vis_mode:
                if _restore_vis_mode is None:
                    os.environ.pop("VISUAL_INSPECT_PROMPT_MODE", None)
                else:
                    os.environ["VISUAL_INSPECT_PROMPT_MODE"] = _restore_vis_mode
        if excess_calls:
            for tc in excess_calls:
                call_id = str(tc.get("id") or "")
                if not call_id:
                    continue
                tool_outputs[call_id] = f"Error: TooManyToolCalls: max {max_calls} tool calls per step"
        try:
            self.transcript.append({"tool_calls": tool_calls, "tool_outputs": tool_outputs})
        except Exception:
            pass
        if terminal_answer_from_spans:
            llm_response = ""
            if answer_from_spans_call_id and answer_from_spans_call_id in tool_outputs:
                llm_response = str(tool_outputs.get(answer_from_spans_call_id) or "")
            elif tool_outputs:
                # Best-effort fallback: use the first (or only) tool output string.
                try:
                    llm_response = str(next(iter(tool_outputs.values())) or "")
                except Exception:
                    llm_response = ""

            try:
                if isinstance(llm_response, str) and llm_response.strip():
                    self.transcript.append({"assistant_response": llm_response})
            except Exception:
                pass

            task_info = self.task if self.task is not None else {}
            try:
                task_info = dict(task_info)
                task_info["transcript"] = list(self.transcript)
                task_info["tool_calls_total"] = int(self.tool_calls_total)
            except Exception:
                pass

            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            done = True
            return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata, "is_correct": reward_output.is_correct}
        next_obs = {"tool_outputs": tool_outputs}

        # Return results as lists with single items to maintain batch structure
        return next_obs, reward, done, {"response": action, "metadata": {}}

    def _execute_tool_calls(self, tool_calls: list[dict[Any, Any]]) -> dict[str, str]:
        import threading

        # Create a dictionary to store results in order
        tool_outputs: dict[str, str] = {}
        output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        threads = []

        def inject_task_context(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
            # Video-style tasks often keep resources in task fields or task["extra_info"].
            if not isinstance(self.task, dict):
                return tool_args

            task = self.task
            extra = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}

            def get_first(*keys: str):
                for k in keys:
                    if k in task and task[k] is not None:
                        return task[k]
                for k in keys:
                    if k in extra and extra[k] is not None:
                        return extra[k]
                return None

            question = get_first("question", "prompt", "QUERY", "Question", "QUESTION")
            video_path = get_first("video_path", "VIDEO_PATH", "video", "VIDEO")
            video_id = get_first("video_id", "VIDEO_ID", "vid", "VID")
            image_index_dir = get_first("IMAGE_INDEX_DIR", "image_index_dir", "image_index", "IMAGE_INDEX", "image_index_path")
            visual_index_dir = get_first("VISUAL_INDEX_DIR", "visual_index_dir", "semantic_index", "SEMANTIC_INDEX", "semantic_index_dir")
            video_duration_sec = get_first("video_duration_sec", "VIDEO_DURATION_SEC", "duration_sec", "video_duration", "VIDEO_DURATION")
            summary_dir = get_first("SUMMARY_DIR", "summary_dir")
            summary_file = get_first("SUMMARY_FILE", "summary_file")

            # Inject common hidden fields. Resource paths should be treated as
            # task-owned (do not let the agent override them when the task provides them).
            if tool_name in {"visual_inspect", "visual_scout", "image_retrieve", "visual_retrieve", "answer_from_spans"}:
                if video_path:
                    tool_args["video_path"] = video_path
                if video_id:
                    tool_args["video_id"] = video_id
                if question:
                    tool_args["original_question"] = question
                if tool_name == "image_retrieve" and image_index_dir:
                    tool_args["index_path"] = image_index_dir
                if tool_name == "visual_retrieve" and visual_index_dir:
                    tool_args["index_path"] = visual_index_dir
                # Inject duration when available so tools can avoid probing.
                if video_duration_sec is not None and "video_duration_sec" not in tool_args:
                    try:
                        tool_args["video_duration_sec"] = float(video_duration_sec)
                    except Exception:
                        pass

            # Video task convention: inject the original question for visual_inspect.
            if tool_name in {"visual_inspect", "visual_scout", "answer_from_spans"}:
                if question:
                    tool_args["questions"] = question

            # Summary reader tool (DVAgent compatibility)
            if tool_name in {"overview_read", "read_summary"}:
                if video_id:
                    tool_args["video_id"] = video_id
                if summary_dir:
                    tool_args["summary_dir"] = summary_dir
                if summary_file:
                    tool_args["summary_file"] = summary_file

            return tool_args

        def execute_tool(tool_call):
            tool_call_id = str(tool_call.get("id") or "")
            try:
                fn = tool_call.get("function") or {}
                tool_name = fn.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise ValueError("missing tool name")

                args_raw = fn.get("arguments", "{}")
                tool_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                if not isinstance(tool_args, dict):
                    raise ValueError("tool arguments must be a JSON object")

                tool_args = inject_task_context(tool_name, tool_args)
                tool_output = self.tools(tool_name=tool_name, **tool_args)
                tool_output_str = tool_output.to_string()
            except Exception as exc:
                tool_output_str = f"Error: {type(exc).__name__}: {exc}"

            if tool_call_id:
                output_queue.put((tool_call_id, tool_output_str))

        # Create and start a thread for each tool call
        for idx, tool_call in enumerate(tool_calls):
            thread = threading.Thread(target=execute_tool, args=(tool_call,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results and store in order
        while not output_queue.empty():
            tool_call_id, output_str = output_queue.get()
            tool_outputs[tool_call_id] = output_str

        return tool_outputs

    @staticmethod
    def from_dict(env_args: dict) -> "ToolEnvironment":
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        return ToolEnvironment(task=env_args, tools=tools, tool_map=tool_map, max_steps=max_steps, reward_fn=reward_fn)
