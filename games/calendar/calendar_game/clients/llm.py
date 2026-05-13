"""LLM-backed client wrapping a2a_engine provider clients."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable

try:
    from json_repair import repair_json as _repair_json
except ImportError:  # pragma: no cover
    _repair_json = None

from calendar_game.agents import BaseClient, DecideResult, GameConfig, TokenUsage, TurnResult
from calendar_game.prompts import (
    build_decision_message,
    build_retry_message,
    build_round_start_message,
    build_system_prompt,
    build_turn_message,
    build_voluntary_reschedule_message,
)
from calendar_game.privacy import hydrate_calendar_render_for_llm, hydrate_meeting_for_llm

SystemPromptBuilder = Callable[[dict], str]


def _parse_response(text: str) -> tuple[list[dict], str | None]:
    """Parse model output into (tool_calls, thinking).

    Accepts either the new {"thinking": "...", "actions": [...]} object format
    or the legacy bare-list format. Falls back through fence-stripping, json_repair,
    and regex extraction.
    """
    if not text:
        return [], None

    def _extract(parsed: object) -> tuple[list[dict], str | None] | None:
        if isinstance(parsed, dict) and "actions" in parsed:
            actions = parsed["actions"]
            if isinstance(actions, list):
                return [a for a in actions if isinstance(a, dict)], parsed.get("thinking") or None
        if isinstance(parsed, list):
            return [a for a in parsed if isinstance(a, dict)], None
        return None

    stripped = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()
    for candidate in (text.strip(), stripped):
        try:
            result = _extract(json.loads(candidate))
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass
    if _repair_json is not None:
        try:
            result = _extract(_repair_json(stripped, return_objects=True))
            if result is not None:
                return result
        except Exception:
            pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, stripped, re.DOTALL)
        if m and _repair_json is not None:
            try:
                result = _extract(_repair_json(m.group(), return_objects=True))
                if result is not None:
                    return result
            except Exception:
                pass
    return [], None


def _make_usage(result: dict) -> TokenUsage | None:
    pt = result.get("prompt_tokens")
    ct = result.get("completion_tokens")
    tt = result.get("total_tokens")
    if pt is None and ct is None:
        return None
    return TokenUsage(
        prompt_tokens=pt or 0,
        completion_tokens=ct or 0,
        total_tokens=tt or (pt or 0) + (ct or 0),
        reasoning_tokens=result.get("reasoning_tokens"),
        cached_prompt_tokens=result.get("cached_prompt_tokens"),
    )


class LLMClient(BaseClient):
    """LLM client wrapping an underlying provider client (OpenAI, Anthropic, etc.)."""

    def __init__(self, llm_client: object, system_prompt_builder: SystemPromptBuilder = build_system_prompt) -> None:
        self._llm = llm_client
        self._system_prompt_builder = system_prompt_builder
        self.agent_id: int = -1
        self._system_prompt: str = ""
        self._history: list[dict] = []

    def _call(self, user_message: str) -> dict:
        self._history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self._system_prompt}] + self._history
        try:
            result = self._llm.streaming_with_retry(messages)
            if result is None:
                raise RuntimeError("LLM call exhausted retries")
        except Exception as exc:
            logging.getLogger("calendar_game.llm").error(
                "LLM call failed (agent %d): %s: %s", self.agent_id, type(exc).__name__, exc
            )
            self._history.pop()
            return {"text": None, "prompt_tokens": None, "completion_tokens": None,
                    "total_tokens": None, "duration_s": None, "finish_reason": "error",
                    "_error": str(exc)}
        assistant_text = result.get("text") or ""
        self._history.append({"role": "assistant", "content": assistant_text})
        return result

    def _make_turn_result(self, result: dict) -> TurnResult:
        text = result.get("text") or None
        tool_calls, model_thinking = _parse_response(text or "")
        return TurnResult(
            tool_calls=tool_calls,
            text=text,
            thinking=model_thinking or result.get("reasoning") or None,
            usage=_make_usage(result),
            latency_ms=(result.get("duration_s") or 0) * 1000 or None,
            raw=result.get("_raw_response") or result,
        )

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self.agent_id = agent_id
        self._system_prompt = self._system_prompt_builder(dataclasses.asdict(game_config))
        self._history = []
        self._round_meeting: dict | None = None
        self._round_calendar: str = ""
        self._round_num: int = 0
        self._incurred_penalty: int = 0
        self._first_turn: bool = True
        self._communication_protocol = game_config.communication_protocol

    def observe_penalty(self, incurred_penalty: int) -> None:
        self._incurred_penalty = incurred_penalty

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self._round_meeting = hydrate_meeting_for_llm(
            meeting,
            stable_key=f"agent:{self.agent_id}:round:{round_num}",
        )
        self._round_calendar = hydrate_calendar_render_for_llm(
            calendar_render,
            stable_key=f"agent:{self.agent_id}:round:{round_num}",
        )
        self._round_num = round_num
        self._first_turn = True

    def observe_calendar(self, calendar_render: str) -> None:
        if self._round_meeting is None:
            self._round_calendar = hydrate_calendar_render_for_llm(
                calendar_render,
                stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
            )

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        if self._first_turn:
            self._first_turn = False
            if self._round_meeting is None:
                user_msg = (
                    f"=== YOUR CALENDAR ===\n{self._round_calendar}\n\n"
                    f"{build_turn_message(messages, turn_index, max_turns_per_round, self._communication_protocol)}"
                )
            else:
                user_msg = build_round_start_message(
                    self._round_meeting,
                    self._round_calendar,
                    self._round_num,
                    incurred_penalty=self._incurred_penalty,
                    turn_index=turn_index,
                    max_turns_per_round=max_turns_per_round,
                    communication_protocol=self._communication_protocol,
                )
                if messages:
                    user_msg += "\n\n" + build_turn_message(
                        messages,
                        turn_index,
                        max_turns_per_round,
                        self._communication_protocol,
                    )
        else:
            user_msg = build_turn_message(
                messages,
                turn_index,
                max_turns_per_round,
                self._communication_protocol,
            )
        result = self._call(user_msg)
        return self._make_turn_result(result)

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        hydrated_meeting = hydrate_meeting_for_llm(
            meeting,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        hydrated_calendar = hydrate_calendar_render_for_llm(
            calendar_render,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        msg = build_decision_message(hydrated_meeting, hydrated_calendar)
        result = self._call(msg)
        tr = self._make_turn_result(result)
        return DecideResult(
            tool_calls=tr.tool_calls, text=tr.text, thinking=tr.thinking,
            usage=tr.usage, latency_ms=tr.latency_ms, raw=tr.raw, retry_count=0,
        )

    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        msg = build_retry_message(attempt, max_attempts, conflict)
        result = self._call(msg)
        tr = self._make_turn_result(result)
        return DecideResult(
            tool_calls=tr.tool_calls, text=tr.text, thinking=tr.thinking,
            usage=tr.usage, latency_ms=tr.latency_ms, raw=tr.raw, retry_count=attempt,
        )

    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        hydrated_meeting = hydrate_meeting_for_llm(
            meeting,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        hydrated_calendar = hydrate_calendar_render_for_llm(
            calendar_render,
            stable_key=f"agent:{self.agent_id}:round:{self._round_num}",
        )
        msg = build_voluntary_reschedule_message(hydrated_meeting, hydrated_calendar)
        result = self._call(msg)
        tr = self._make_turn_result(result)
        return DecideResult(
            tool_calls=tr.tool_calls, text=tr.text, thinking=tr.thinking,
            usage=tr.usage, latency_ms=tr.latency_ms, raw=tr.raw, retry_count=0,
        )
