"""Calendar scheduling game for a2a-engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import uuid
from pathlib import Path

from a2a_engine import EventLog, GameConfigBase, GameTraceBase, register_game
from pydantic import Field

from calendar_game.agents import Agent, BaseClient, GameConfig
from calendar_game.clients import (
    DSMClient,
    DSPyClient,
    IncrementalMAPClient,
    LLMClient,
    PaperDSMClient,
    PrivateDSMClient,
    SDClient,
    ScriptedClient,
)
from calendar_game.prompts import (
    build_decision_message,
    build_round_start_message,
    build_system_prompt,
    build_turn_message,
    build_voluntary_reschedule_message,
)
from calendar_game.privacy import hydrate_calendar_render_for_llm, hydrate_meeting_for_llm
from calendar_game.calendar import Calendar, apply_batch, validate_batch
from calendar_game.fallback import FallbackDepthExceeded, FallbackImpossible, find_fallback_slot
from calendar_game.scenario import generate_scenario
from calendar_game.solver import solve_greedy, solve_optimal
from a2a_engine.llm.factory import make_llm_client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CalendarGameConfig(GameConfigBase):
    """Config for the calendar scheduling benchmark."""

    game_name: str = "calendar"
    num_agents: int = 2
    num_slots: int = 16
    density: float = 0.5
    pref_level: int = 1
    num_meetings: int = 1
    num_participants: int | None = None  # agents per meeting; None means all agents participate
    max_turns_per_round: int = 20
    dm_cap: int = 1_000_000
    decision_retries: int = 3
    errand_cost_level: int = 10
    meeting_cost_level: int = 1
    enable_fallback: bool = True
    fallback_max_depth: int = 3
    task_path: str | None = None
    task_id: str | None = None
    dsm_num_proposals: int = 4
    dsm_cascade_depth: int = 1
    dsm_displacement_targets: int = 4
    dsm_exhaustive_search: bool = True
    dsm_stop_on_perfect: bool = True
    nosy_agent_ids: list[int] = Field(default_factory=list)
    dsm_lmin: int = 1
    dsm_lmax: int | None = None
    dsm_beta: float = 1.0
    dsm_theta: float = 0.0
    dsm_social_welfare_weight: float = 1.0
    dsm_privacy_unit_cost: float = 1.0
    dsm_initial_budget: int = 100
    sd_model: dict[int, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# CalendarGame
# ---------------------------------------------------------------------------


class CalendarGame:
    """Calendar scheduling benchmark game."""

    def __init__(self, config: dict | CalendarGameConfig, dry_run: bool = False) -> None:
        self.config = config if isinstance(config, CalendarGameConfig) else CalendarGameConfig(**config)
        self.dry_run = dry_run
        self.events = EventLog()

    @staticmethod
    def _meeting_participants(scenario: dict) -> dict[int, list[int]]:
        participants = {
            int(meeting["id"]): list(meeting.get("participants", []))
            for meeting in scenario.get("meetings", [])
        }
        for prior in scenario.get("prior_meetings", []):
            participants[int(prior["id"])] = list(prior.get("participants", []))
        return participants

    @staticmethod
    def _speaker_order(meeting: dict) -> list[int]:
        participants = list(meeting.get("participants", []))
        order = list(meeting.get("speaker_order") or participants)
        if sorted(order) != sorted(participants):
            return participants
        return order

    def _annotate_calendars(self, agents: list[Agent], scenario: dict) -> None:
        meeting_participants = self._meeting_participants(scenario)
        for agent in agents:
            agent.calendar.meeting_participants = meeting_participants

    def _prompt_meeting_for_agent(self, agent: Agent, meeting: dict, round_num: int) -> dict:
        if isinstance(agent.client, LLMClient):
            return hydrate_meeting_for_llm(
                meeting,
                stable_key=f"agent:{agent.agent_id}:round:{round_num}",
            )
        return meeting

    def _prompt_calendar_for_agent(self, agent: Agent, calendar_render: str, round_num: int) -> str:
        if isinstance(agent.client, LLMClient):
            return hydrate_calendar_render_for_llm(
                calendar_render,
                stable_key=f"agent:{agent.agent_id}:round:{round_num}",
            )
        return calendar_render

    @staticmethod
    def _balanced_participant_lists(
        seed: int | None,
        total_agents: int,
        subset_size: int,
        num_meetings: int,
    ) -> list[list[int]]:
        rng = random.Random(seed)
        if subset_size >= total_agents:
            return [list(range(total_agents)) for _ in range(num_meetings)]

        total_appearances = subset_size * num_meetings
        if total_appearances % total_agents != 0:
            return [
                sorted(rng.sample(range(total_agents), subset_size))
                for _ in range(num_meetings)
            ]

        target = total_appearances // total_agents
        combos: list[tuple[int, ...]] = []

        def build(start: int, current: list[int]) -> None:
            if len(current) == subset_size:
                combos.append(tuple(current))
                return
            for agent_id in range(start, total_agents):
                current.append(agent_id)
                build(agent_id + 1, current)
                current.pop()

        build(0, [])
        rng.shuffle(combos)
        counts = {agent_id: 0 for agent_id in range(total_agents)}
        chosen: list[tuple[int, ...]] = []

        def search() -> bool:
            if len(chosen) == num_meetings:
                return all(count == target for count in counts.values())
            viable = [
                combo for combo in combos
                if all(counts[agent_id] < target for agent_id in combo)
            ]
            viable.sort(key=lambda combo: (sum(counts[agent_id] for agent_id in combo), rng.random()))
            for combo in viable:
                chosen.append(combo)
                for agent_id in combo:
                    counts[agent_id] += 1
                if search():
                    return True
                for agent_id in combo:
                    counts[agent_id] -= 1
                chosen.pop()
            return False

        if not search():
            return [
                sorted(rng.sample(range(total_agents), subset_size))
                for _ in range(num_meetings)
            ]
        return [list(combo) for combo in chosen]

    def _ensure_speaker_orders(self, scenario: dict) -> None:
        if all("speaker_order" in meeting for meeting in scenario.get("meetings", [])):
            return
        from calendar_game.taskgen import assign_balanced_speaker_orders
        participant_lists = [
            list(meeting.get("participants", []))
            for meeting in scenario.get("meetings", [])
        ]
        speaker_orders = assign_balanced_speaker_orders(
            int(scenario.get("seed") or self.config.seed or 0),
            participant_lists,
            self.config.num_agents,
        )
        for meeting, speaker_order in zip(scenario.get("meetings", []), speaker_orders, strict=True):
            meeting["speaker_order"] = speaker_order

    def _invalid_tool_call(
        self,
        *,
        round_num: int,
        turn: int,
        phase: str,
        agent_id: int,
        tool: object,
        reason: str,
    ) -> None:
        self.events.append("invalid_tool_call", data={
            "round": round_num,
            "turn": turn,
            "phase": phase,
            "agent_id": agent_id,
            "tool_call": tool,
            "reason": reason,
        })

    @staticmethod
    def _blocked_slots_by_agent(scenario: dict) -> dict[int, set[int]]:
        blocked: dict[int, set[int]] = {}
        for agent_id, calendar in enumerate(scenario.get("calendars", [])):
            blocked[agent_id] = {
                slot_idx
                for slot_idx, slot_val in enumerate(calendar)
                if isinstance(slot_val, dict) and bool(slot_val.get("blocked"))
            }
        return blocked

    @staticmethod
    def _blocked_reschedule_violations(
        calendar: Calendar,
        actions: list[dict],
        *,
        agent_id: int,
        phase: str,
        attempt: int,
    ) -> list[dict]:
        violations: list[dict] = []
        for action in actions:
            if action.get("type") != "reschedule":
                continue
            from_slot = action.get("from_slot")
            if not isinstance(from_slot, int) or from_slot < 0 or from_slot >= calendar.num_slots:
                continue
            slot_val = calendar.get(from_slot)
            if isinstance(slot_val, dict) and slot_val.get("blocked"):
                violations.append({
                    "agent_id": agent_id,
                    "phase": phase,
                    "attempt": attempt,
                    "kind": "blocked_reschedule",
                    "slot": from_slot,
                    "item_id": action.get("item_id"),
                })
        return violations

    def _decision_actions(self, tool_calls: list[dict], meeting: dict) -> list[dict]:
        actions: list[dict] = []
        for tool in tool_calls:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") not in ("schedule", "reschedule"):
                continue
            if tool.get("type") == "schedule" and tool.get("meeting_id") == meeting["id"]:
                actions.append({**tool, "cost": meeting.get("cost", 1)})
            else:
                actions.append(tool)
        return actions

    def generate_scenario(self) -> dict:
        """Generate a scenario from this game's config. Can be inspected or modified before run_with_scenario()."""
        if self.config.task_path:
            return self._load_task_scenario()

        from calendar_game.taskgen import assign_balanced_speaker_orders, select_participants
        participant_lists = [list(range(self.config.num_agents)) for _ in range(self.config.num_meetings)]
        if self.config.num_participants is not None:
            k = max(1, min(self.config.num_participants, self.config.num_agents))
            if self.config.num_meetings == 1:
                participant_lists = [select_participants(self.config.seed, self.config.num_agents, k)]
            else:
                participant_lists = self._balanced_participant_lists(
                    self.config.seed,
                    self.config.num_agents,
                    k,
                    self.config.num_meetings,
                )
        speaker_orders = assign_balanced_speaker_orders(
            self.config.seed,
            participant_lists,
            self.config.num_agents,
        )
        return generate_scenario(
            self.config.seed,
            self.config.num_agents,
            self.config.num_slots,
            self.config.density,
            self.config.pref_level,
            self.config.num_meetings,
            participant_lists=participant_lists,
            speaker_orders=speaker_orders,
            errand_cost_level=self.config.errand_cost_level,
            meeting_cost_level=self.config.meeting_cost_level,
        )

    def _load_task_scenario(self) -> dict:
        if not self.config.task_id:
            raise ValueError("task_id is required when task_path is set")

        task_path = Path(self.config.task_path)
        if not task_path.is_absolute():
            candidates = [
                Path.cwd() / task_path,
                Path(__file__).resolve().parents[1] / task_path,
            ]
            task_path = next((path for path in candidates if path.exists()), candidates[0])

        with task_path.open() as f:
            for line in f:
                task = json.loads(line)
                if task.get("task_id") != self.config.task_id:
                    continue

                calendars = task["calendars"]
                meetings = task["meetings"]
                if len(calendars) != self.config.num_agents:
                    raise ValueError(
                        f"task {self.config.task_id!r} has {len(calendars)} calendars, "
                        f"but config num_agents={self.config.num_agents}"
                    )
                if any(len(calendar) != self.config.num_slots for calendar in calendars):
                    raise ValueError(
                        f"task {self.config.task_id!r} calendar length does not match "
                        f"config num_slots={self.config.num_slots}"
                    )
                return {
                    "seed": task.get("seed"),
                    "num_agents": len(calendars),
                    "num_slots": self.config.num_slots,
                    "calendars": calendars,
                    "meetings": meetings,
                    "witness_solution": task.get("witness_solution", {}),
                    "optimal": task.get("optimal", {}),
                    "greedy": task.get("greedy", {}),
                    "prior_meetings": task.get("prior_meetings", []),
                    "feasible": task.get("feasible", True),
                    "task_id": task.get("task_id"),
                }

        raise ValueError(f"task_id {self.config.task_id!r} not found in {task_path}")

    def run(self) -> GameTraceBase:
        return self.run_with_scenario(self.generate_scenario())

    def run_with_scenario(self, scenario: dict) -> GameTraceBase:
        return asyncio.run(self._run_async(scenario))

    def _build_agents(self, scenario: dict) -> list[Agent]:
        """Construct and calendar-initialize agents from scenario. Separated for testability."""
        agents: list[Agent] = []
        for agent_id in range(self.config.num_agents):
            if self.dry_run:
                client: BaseClient = ScriptedClient()
            else:
                spec = self.config.agents[agent_id] if agent_id < len(self.config.agents) else {"model": "gpt-4o-mini"}
                cfg = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
                agent_type = cfg.get("type", "llm")
                if agent_type == "dsm":
                    client = DSMClient()
                elif agent_type == "paper_dsm":
                    client = PaperDSMClient()
                elif agent_type == "private_dsm":
                    client = PrivateDSMClient()
                elif agent_type in {"imap", "incremental_map"}:
                    client = IncrementalMAPClient()
                elif agent_type in {"sd", "scheduling_difficulty"}:
                    client = SDClient()
                elif agent_type == "dspy":
                    prompt_variant = cfg.get("prompt_variant") or cfg.get("extra", {}).get("prompt_variant")
                    prompt_variant_dir = cfg.get("prompt_variant_dir") or cfg.get("extra", {}).get("prompt_variant_dir")
                    client = DSPyClient(
                        make_llm_client(cfg),
                        prompt_variant=prompt_variant,
                        prompt_variant_dir=prompt_variant_dir,
                    )
                else:
                    client = LLMClient(make_llm_client(cfg))
            agent = Agent(client)
            cal = Calendar(num_slots=self.config.num_slots)
            cal.slots = list(scenario["calendars"][agent_id])
            agent.calendar = cal
            agents.append(agent)
        self._annotate_calendars(agents, scenario)
        return agents

    def _nosy_agent_ids(self) -> list[int]:
        ids: set[int] = set()
        for raw_agent_id in self.config.nosy_agent_ids:
            try:
                agent_id = int(raw_agent_id)
            except (TypeError, ValueError):
                continue
            if 0 <= agent_id < self.config.num_agents:
                ids.add(agent_id)
        return sorted(ids)

    def _run_with_agents(self, agents: list[Agent], scenario: dict) -> GameTraceBase:
        """Run the full game loop with a pre-built agent list. Exposed for testing."""
        return asyncio.run(self._run_async(scenario, agents=agents))

    async def _run_async(self, scenario: dict, agents: list[Agent] | None = None) -> GameTraceBase:
        self._ensure_speaker_orders(scenario)
        optimal = (
            scenario.get("optimal")
            if scenario.get("optimal", {}).get("cost") is not None
            else solve_optimal(scenario["calendars"], scenario["meetings"], self.config.num_slots)
        )
        greedy = (
            scenario.get("greedy")
            if scenario.get("greedy", {}).get("cost") is not None
            else solve_greedy(scenario["calendars"], scenario["meetings"], self.config.num_slots)
        )

        # 2. Build agents (or use injected ones for testing)
        if agents is None:
            agents = self._build_agents(scenario)
        else:
            self._annotate_calendars(agents, scenario)
        nosy_agent_ids = self._nosy_agent_ids()
        blocked_slots_by_agent = self._blocked_slots_by_agent(scenario)

        # 3. game_start event — emitted before agent registration so it is always first
        self.events.append("game_start", data={
            "round": -1, "turn": -1, "phase": "GAME_START", "agent_id": None,
            "scenario_seed": self.config.seed,
            "num_agents": self.config.num_agents,
            "num_slots": self.config.num_slots,
            "optimal_cost": optimal.get("cost"),
            "greedy_cost": greedy.get("cost"),
            "nosy_agent_ids": nosy_agent_ids,
        })

        # 4. Register all agents
        all_agent_ids = list(range(self.config.num_agents))
        for agent_id, agent in enumerate(agents):
            dsm_prior_meetings = [
                {
                    "id": int(prior["id"]),
                    "participants": list(prior.get("participants", [])),
                    "slot": prior.get("slot"),
                }
                for prior in scenario.get("prior_meetings", [])
                if agent_id in prior.get("participants", [])
            ]
            game_config = GameConfig(
                num_agents=self.config.num_agents,
                num_slots=self.config.num_slots,
                agent_id=agent_id,
                all_agent_ids=all_agent_ids,
                dm_cap=self.config.dm_cap,
                decision_retries=self.config.decision_retries,
                dsm_num_proposals=self.config.dsm_num_proposals,
                dsm_cascade_depth=self.config.dsm_cascade_depth,
                dsm_displacement_targets=self.config.dsm_displacement_targets,
                dsm_exhaustive_search=self.config.dsm_exhaustive_search,
                dsm_stop_on_perfect=self.config.dsm_stop_on_perfect,
                dsm_prior_meetings=dsm_prior_meetings,
                dsm_lmin=self.config.dsm_lmin,
                dsm_lmax=self.config.dsm_lmax,
                dsm_beta=self.config.dsm_beta,
                dsm_theta=self.config.dsm_theta,
                dsm_social_welfare_weight=self.config.dsm_social_welfare_weight,
                dsm_privacy_unit_cost=self.config.dsm_privacy_unit_cost,
                dsm_initial_budget=self.config.dsm_initial_budget,
                sd_model={int(k): float(v) for k, v in self.config.sd_model.items()},
            )
            agent.register(agent_id, game_config)
            system_prompt_text = getattr(agent.client, "_system_prompt", None) or build_system_prompt(
                dataclasses.asdict(game_config)
            )
            self.events.append("agent_registered", data={
                "round": -1, "turn": -1, "phase": "GAME_START", "agent_id": agent_id,
                "is_nosy_agent": agent_id in nosy_agent_ids,
                "nosy_agent_ids": nosy_agent_ids,
                "system_prompt": system_prompt_text,
                "calendar_render": agent.calendar.render(),
            })

        # 5. Per-game accumulators
        displacement_cost: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        fallback_displacement_cost: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        total_client_calls: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        total_dms_sent: int = 0
        total_dm_chars: int = 0
        max_dm_chars: int = 0
        round_outcomes: list[dict] = []
        registered_meetings: dict[int, dict] = {
            int(meeting["id"]): meeting
            for meeting in scenario["meetings"]
        }
        for prior in scenario.get("prior_meetings", []):
            registered_meetings[int(prior["id"])] = {
                "id": int(prior["id"]),
                "participants": list(prior.get("participants", [])),
                "duration": int(prior.get("duration", 1)),
                "cost": int(prior.get("cost", 1)),
            }
        meeting_registry: dict[int, int] = {
            int(prior["id"]): int(prior["slot"])
            for prior in scenario.get("prior_meetings", [])
            if prior.get("slot") is not None
        }  # meeting_id → canonical_slot

        # 6. Main loop — one round per meeting
        for round_num, meeting in enumerate(scenario["meetings"]):
            speaker_order = self._speaker_order(meeting)
            self.events.append("round_start", data={
                "round": round_num, "turn": 0, "phase": "CHEAP_TALK", "agent_id": None,
                "meeting": meeting,
                "speaker_order": speaker_order,
            })

            # call start_round for all participants
            for agent_id in speaker_order:
                agents[agent_id].start_round(meeting, round_num, incurred_penalty=displacement_cost[agent_id])

            # per-round state
            already_queued: set[int] = set()
            turn_index = 0
            blocked_slot_violations: list[dict] = []

            # --- CHEAP_TALK PHASE ---
            has_activity = True
            while has_activity and turn_index < self.config.max_turns_per_round:
                has_activity = False

                # participants act
                for agent_id in speaker_order:
                    agent = agents[agent_id]
                    inbox_snapshot = list(agents[agent_id].inbox_queue)
                    calendar_render = agents[agent_id].calendar.render()
                    prompt_calendar_render = self._prompt_calendar_for_agent(agent, calendar_render, round_num)
                    prompt_meeting = self._prompt_meeting_for_agent(agent, meeting, round_num)
                    turn_prompt = (
                        build_round_start_message(
                            prompt_meeting,
                            prompt_calendar_render,
                            round_num,
                            incurred_penalty=displacement_cost[agent_id],
                            turn_index=turn_index,
                            max_turns_per_round=self.config.max_turns_per_round,
                        )
                        if turn_index == 0
                        else build_turn_message(
                            inbox_snapshot,
                            turn_index,
                            self.config.max_turns_per_round,
                        )
                    )
                    self.events.append("turn_start", data={
                        "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                        "agent_id": agent_id,
                        "inbox_drained": inbox_snapshot,
                        "calendar_render": prompt_calendar_render,
                        "prompt_sent": turn_prompt,
                    })
                    result = agents[agent_id].turn(turn_index, self.config.max_turns_per_round)
                    total_client_calls[agent_id] += 1
                    self.events.append("turn_end", data={
                        "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                        "agent_id": agent_id,
                        "tool_calls": result.tool_calls,
                        "text": result.text,
                        "thinking": result.thinking,
                        "usage": result.usage.__dict__ if result.usage else None,
                        "latency_ms": result.latency_ms,
                        "raw_api_response": result.raw,
                    })

                    for tool in result.tool_calls:
                        if not isinstance(tool, dict):
                            self._invalid_tool_call(
                                round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                                agent_id=agent_id, tool=tool, reason="tool call is not an object",
                            )
                            continue
                        if tool.get("type") != "dm":
                            continue
                        try:
                            to = int(tool["to"])
                        except (KeyError, TypeError, ValueError):
                            self._invalid_tool_call(
                                round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                                agent_id=agent_id, tool=tool, reason="dm tool missing integer 'to'",
                            )
                            continue
                        # deliver DM
                        msg = {"from": agent_id, "meeting_id": meeting["id"], "content": str(tool.get("content", ""))}
                        dm_chars = len(msg["content"])
                        agents[to].inbox_queue.append(msg)
                        total_dms_sent += 1
                        total_dm_chars += dm_chars
                        max_dm_chars = max(max_dm_chars, dm_chars)
                        has_activity = True
                        self.events.append("dm_sent", data={
                            "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                            "agent_id": agent_id,
                            "from_agent": agent_id, "to_agent": to,
                            "meeting_id": msg["meeting_id"], "content": msg["content"],
                            "content_chars": dm_chars,
                        })
                        # queue non-participants
                        if to not in meeting["participants"] and to not in already_queued:
                            already_queued.add(to)

                # drain unique_queue (non-participants who got DMs)
                queue = list(already_queued - set(meeting["participants"]))
                for agent_id in queue:
                    agent = agents[agent_id]
                    inbox_snapshot = list(agents[agent_id].inbox_queue)
                    calendar_render = agents[agent_id].calendar.render()
                    prompt_calendar_render = self._prompt_calendar_for_agent(agent, calendar_render, round_num)
                    turn_prompt = build_turn_message(
                        inbox_snapshot,
                        turn_index,
                        self.config.max_turns_per_round,
                    )
                    self.events.append("turn_start", data={
                        "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                        "agent_id": agent_id,
                        "inbox_drained": inbox_snapshot,
                        "calendar_render": prompt_calendar_render,
                        "prompt_sent": turn_prompt,
                    })
                    result = agents[agent_id].turn(turn_index, self.config.max_turns_per_round)
                    total_client_calls[agent_id] += 1
                    self.events.append("turn_end", data={
                        "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                        "agent_id": agent_id,
                        "tool_calls": result.tool_calls,
                        "text": result.text, "thinking": result.thinking,
                        "usage": result.usage.__dict__ if result.usage else None,
                        "latency_ms": result.latency_ms, "raw_api_response": result.raw,
                    })
                    for tool in result.tool_calls:
                        if not isinstance(tool, dict):
                            self._invalid_tool_call(
                                round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                                agent_id=agent_id, tool=tool, reason="tool call is not an object",
                            )
                            continue
                        if tool.get("type") != "dm":
                            continue
                        try:
                            to = int(tool["to"])
                        except (KeyError, TypeError, ValueError):
                            self._invalid_tool_call(
                                round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                                agent_id=agent_id, tool=tool, reason="dm tool missing integer 'to'",
                            )
                            continue
                        msg = {"from": agent_id, "meeting_id": meeting["id"], "content": str(tool.get("content", ""))}
                        dm_chars = len(msg["content"])
                        agents[to].inbox_queue.append(msg)
                        total_dms_sent += 1
                        total_dm_chars += dm_chars
                        max_dm_chars = max(max_dm_chars, dm_chars)
                        has_activity = True
                        self.events.append("dm_sent", data={
                            "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                            "agent_id": agent_id,
                            "from_agent": agent_id, "to_agent": to,
                            "meeting_id": msg["meeting_id"], "content": msg["content"],
                            "content_chars": dm_chars,
                        })
                        if to not in already_queued:
                            already_queued.add(to)

                turn_index += 1

            # --- VOLUNTARY RESCHEDULE PHASE (non-participants who received DMs) ---
            for agent_id in sorted(already_queued - set(meeting["participants"])):
                agent = agents[agent_id]
                calendar_render = agents[agent_id].calendar.render()
                prompt_calendar_render = self._prompt_calendar_for_agent(agent, calendar_render, round_num)
                prompt_meeting = self._prompt_meeting_for_agent(agent, meeting, round_num)
                self.events.append("decide_start", data={
                    "round": round_num, "turn": turn_index, "phase": "VOLUNTARY",
                    "agent_id": agent_id,
                    "calendar_render": prompt_calendar_render,
                    "prompt_sent": build_voluntary_reschedule_message(prompt_meeting, prompt_calendar_render),
                })
                result = agents[agent_id].voluntary_decide(meeting)
                total_client_calls[agent_id] += 1
                self.events.append("decide_end", data={
                    "round": round_num, "turn": turn_index, "phase": "VOLUNTARY",
                    "agent_id": agent_id,
                    "tool_calls": result.tool_calls, "text": result.text,
                    "thinking": result.thinking,
                    "usage": result.usage.__dict__ if result.usage else None,
                    "latency_ms": result.latency_ms, "raw_api_response": result.raw,
                })
                actions = [
                    a for a in result.tool_calls
                    if isinstance(a, dict) and a.get("type") == "reschedule"
                ]
                for attempt in range(self.config.decision_retries + 1):
                    blocked_slot_violations.extend(self._blocked_reschedule_violations(
                        agents[agent_id].calendar,
                        actions,
                        agent_id=agent_id,
                        phase="VOLUNTARY",
                        attempt=attempt,
                    ))
                    ok, conflict = validate_batch(agents[agent_id].calendar, actions, require_schedule=False)
                    if ok:
                        for action in actions:
                            from_slot = int(action["from_slot"])
                            item = agents[agent_id].calendar.get(from_slot)
                            if isinstance(item, dict) and "cost" in item:
                                displacement_cost[agent_id] += int(item["cost"])
                        apply_batch(agents[agent_id].calendar, actions)
                        self.events.append("batch_applied", data={
                            "round": round_num, "turn": turn_index, "phase": "VOLUNTARY",
                            "agent_id": agent_id,
                            "actions": actions,
                            "calendar_render_after": agents[agent_id].calendar.render(),
                        })
                        break
                    else:
                        self.events.append("batch_rejected", data={
                            "round": round_num, "turn": turn_index, "phase": "VOLUNTARY",
                            "agent_id": agent_id,
                            "attempt": attempt, "conflict_description": conflict, "actions": actions,
                        })
                        if attempt < self.config.decision_retries:
                            retry_result = agents[agent_id].client.retry_decide(attempt + 1, self.config.decision_retries, conflict)
                            total_client_calls[agent_id] += 1
                            actions = [
                                a for a in retry_result.tool_calls
                                if isinstance(a, dict) and a.get("type") == "reschedule"
                            ]
                        else:
                            break

            # --- DECISION PHASE ---
            staged_decisions: dict[int, tuple[Calendar, list[dict], int]] = {}
            decision_phase_failed = False
            for agent_id in speaker_order:
                agent = agents[agent_id]
                snapshot_render = agents[agent_id].calendar.snapshot().render()
                prompt_snapshot_render = self._prompt_calendar_for_agent(agent, snapshot_render, round_num)
                prompt_meeting = self._prompt_meeting_for_agent(agent, meeting, round_num)
                decision_prompt = build_decision_message(prompt_meeting, prompt_snapshot_render)
                self.events.append("decide_start", data={
                    "round": round_num, "turn": turn_index, "phase": "DECISION",
                    "agent_id": agent_id,
                    "calendar_snapshot_render": prompt_snapshot_render,
                    "prompt_sent": decision_prompt,
                })
                result = agents[agent_id].decide(meeting)
                total_client_calls[agent_id] += 1
                self.events.append("decide_end", data={
                    "round": round_num, "turn": turn_index, "phase": "DECISION",
                    "agent_id": agent_id,
                    "tool_calls": result.tool_calls, "text": result.text,
                    "thinking": result.thinking,
                    "usage": result.usage.__dict__ if result.usage else None,
                    "latency_ms": result.latency_ms, "raw_api_response": result.raw,
                    "retry_count": result.retry_count, "status": "pending",
                })

                # validate and apply batch with retries
                # Inject meeting cost into schedule actions so apply_batch can store it
                actions = self._decision_actions(result.tool_calls, meeting)
                for attempt in range(self.config.decision_retries + 1):
                    blocked_slot_violations.extend(self._blocked_reschedule_violations(
                        agents[agent_id].calendar,
                        actions,
                        agent_id=agent_id,
                        phase="DECISION",
                        attempt=attempt,
                    ))
                    ok, conflict = validate_batch(agents[agent_id].calendar, actions)
                    if ok:
                        staged_calendar = agents[agent_id].calendar.snapshot()
                        pending_displacement_cost = 0
                        for action in actions:
                            if action.get("type") == "reschedule":
                                from_slot = int(action["from_slot"])
                                item = agents[agent_id].calendar.get(from_slot)
                                if isinstance(item, dict) and "cost" in item:
                                    pending_displacement_cost += int(item["cost"])
                        apply_batch(staged_calendar, actions)
                        staged_decisions[agent_id] = (
                            staged_calendar,
                            actions,
                            pending_displacement_cost,
                        )
                        break
                    else:
                        self.events.append("batch_rejected", data={
                            "round": round_num, "turn": turn_index, "phase": "DECISION",
                            "agent_id": agent_id,
                            "attempt": attempt, "conflict_description": conflict, "actions": actions,
                        })
                        if attempt < self.config.decision_retries:
                            retry_result = agents[agent_id].client.retry_decide(attempt + 1, self.config.decision_retries, conflict)
                            total_client_calls[agent_id] += 1
                            actions = self._decision_actions(retry_result.tool_calls, meeting)
                        else:
                            self.events.append("decision_failed", data={
                                "round": round_num, "turn": turn_index, "phase": "DECISION",
                                "agent_id": agent_id,
                                "attempts_exhausted": self.config.decision_retries + 1,
                            })
                            decision_phase_failed = True
                            break

            if (
                not decision_phase_failed
                and not blocked_slot_violations
                and len(staged_decisions) == len(speaker_order)
            ):
                for agent_id in speaker_order:
                    staged_calendar, actions, pending_displacement_cost = staged_decisions[agent_id]
                    agents[agent_id].calendar.slots = staged_calendar.slots
                    agents[agent_id].calendar.meeting_participants = staged_calendar.meeting_participants
                    displacement_cost[agent_id] += pending_displacement_cost
                    self.events.append("batch_applied", data={
                        "round": round_num, "turn": turn_index, "phase": "DECISION",
                        "agent_id": agent_id,
                        "actions": actions,
                        "calendar_render_after": agents[agent_id].calendar.render(),
                    })
            elif staged_decisions:
                rollback_reason = (
                    "blocked slot violation"
                    if blocked_slot_violations
                    else "another participant failed decision validation"
                )
                for agent_id, (_staged_calendar, actions, _pending_cost) in staged_decisions.items():
                    self.events.append("batch_rolled_back", data={
                        "round": round_num, "turn": turn_index, "phase": "DECISION",
                        "agent_id": agent_id,
                        "actions": actions,
                        "reason": rollback_reason,
                        "calendar_render_after": agents[agent_id].calendar.render(),
                    })

            # --- RESOLUTION PHASE ---
            per_agent_slot: dict[str, int | None] = {}
            for agent_id in meeting["participants"]:
                found_slot = None
                for slot_idx, slot_val in enumerate(agents[agent_id].calendar.slots):
                    if isinstance(slot_val, dict) and slot_val.get("meeting_id") == meeting["id"]:
                        found_slot = slot_idx
                        break
                per_agent_slot[str(agent_id)] = found_slot

            slots_chosen = [s for s in per_agent_slot.values() if s is not None]
            coordinated = len(slots_chosen) == len(meeting["participants"]) and len(set(slots_chosen)) == 1

            # slot conflicts: any agent with >1 meeting in same slot
            slot_conflicts: dict[str, list[int]] = {}
            for agent_id in range(self.config.num_agents):
                seen: dict[int, list] = {}
                for slot_idx, slot_val in enumerate(agents[agent_id].calendar.slots):
                    if isinstance(slot_val, dict) and "meeting_id" in slot_val:
                        seen.setdefault(slot_idx, []).append(f"M{slot_val['meeting_id']}")
                conflicts = [slot_idx for slot_idx, vals in seen.items() if len(vals) > 1]
                if conflicts:
                    slot_conflicts[str(agent_id)] = conflicts

            for agent_id, blocked_slots in blocked_slots_by_agent.items():
                for slot_idx in blocked_slots:
                    slot_val = agents[agent_id].calendar.get(slot_idx)
                    if isinstance(slot_val, dict) and "meeting_id" in slot_val:
                        blocked_slot_violations.append({
                            "agent_id": agent_id,
                            "phase": "RESOLUTION",
                            "kind": "meeting_on_blocked_slot",
                            "slot": slot_idx,
                            "meeting_id": slot_val.get("meeting_id"),
                        })

            # constraint B: all participants of each registered meeting must agree on slot
            consistency_violated_ids: set[int] = set()
            for reg_mid, canonical_slot in meeting_registry.items():
                reg_meeting = registered_meetings.get(reg_mid)
                if reg_meeting is None:
                    continue
                agent_reg_slots: dict[int, int | None] = {}
                for pid in reg_meeting["participants"]:
                    found = None
                    for s_idx, s_val in enumerate(agents[pid].calendar.slots):
                        if isinstance(s_val, dict) and s_val.get("meeting_id") == reg_mid:
                            found = s_idx
                            break
                    agent_reg_slots[pid] = found
                slot_set = set(agent_reg_slots.values())
                if len(slot_set) > 1 or None in slot_set:
                    consistency_violated_ids.add(reg_mid)
                    for pid, s in agent_reg_slots.items():
                        self.events.append("consistency_violation", data={
                            "round": round_num, "turn": turn_index, "phase": "RESOLUTION",
                            "agent_id": pid,
                            "meeting_id": reg_mid,
                            "canonical_slot": canonical_slot,
                            "agent_slot": s,
                        })

            if consistency_violated_ids:
                coordinated = False
            if blocked_slot_violations:
                coordinated = False

            # update registry: consistent moves update canonical slot
            for reg_mid in list(meeting_registry):
                if reg_mid in consistency_violated_ids:
                    continue
                reg_meeting = registered_meetings.get(reg_mid)
                if reg_meeting is None:
                    continue
                new_slots: set[int] = set()
                for pid in reg_meeting["participants"]:
                    for s_idx, s_val in enumerate(agents[pid].calendar.slots):
                        if isinstance(s_val, dict) and s_val.get("meeting_id") == reg_mid:
                            new_slots.add(s_idx)
                            break
                if len(new_slots) == 1:
                    meeting_registry[reg_mid] = next(iter(new_slots))

            if coordinated:
                meeting_registry[meeting["id"]] = slots_chosen[0]

            self.events.append("resolution", data={
                "round": round_num, "turn": turn_index, "phase": "RESOLUTION",
                "agent_id": None,
                "meeting_id": meeting["id"],
                "per_agent_slot": per_agent_slot,
                "coordinated": coordinated,
                "slot_conflicts": slot_conflicts,
                "consistency_violated_meeting_ids": sorted(consistency_violated_ids),
                "blocked_slot_violations": blocked_slot_violations,
            })

            # --- FALLBACK PHASE ---
            if not coordinated and not blocked_slot_violations and self.config.enable_fallback and not self.dry_run:
                self.events.append("fallback_start", data={
                    "round": round_num, "turn": turn_index, "phase": "FALLBACK",
                    "agent_id": None,
                    "meeting_id": meeting["id"],
                    "participants": meeting["participants"],
                })
                try:
                    chosen_slot, disp_plan = find_fallback_slot(
                        agents, meeting, self.config.num_slots,
                        meeting_registry, list(registered_meetings.values()),
                        max_depth=self.config.fallback_max_depth,
                    )
                    # Apply displacement actions grouped by agent
                    registry_cascades: list[int] = []
                    for action in disp_plan:
                        aid = action["agent_id"]
                        fs, ts = action["from_slot"], action["to_slot"]
                        item = agents[aid].calendar.get(fs)
                        if isinstance(item, dict) and "cost" in item:
                            fallback_displacement_cost[aid] += int(item["cost"])
                        agents[aid].calendar.slots[ts] = agents[aid].calendar.slots[fs]
                        agents[aid].calendar.slots[fs] = None
                        if action.get("is_meeting_cascade"):
                            mid_c = action["item_id"]
                            if mid_c not in registry_cascades:
                                registry_cascades.append(mid_c)
                    # Clear any stale placement of this meeting from the failed round
                    for pid in meeting["participants"]:
                        for s_idx, s_val in enumerate(agents[pid].calendar.slots):
                            if isinstance(s_val, dict) and s_val.get("meeting_id") == meeting["id"]:
                                agents[pid].calendar.slots[s_idx] = None
                    # Place meeting on all participants' calendars at chosen_slot
                    for pid in meeting["participants"]:
                        agents[pid].calendar.slots[chosen_slot] = {
                            "meeting_id": meeting["id"],
                            "cost": meeting.get("cost", 1),
                        }
                    # Update registry for cascaded meetings and the new meeting
                    for mid_c in registry_cascades:
                        reg_m = registered_meetings.get(mid_c)
                        if reg_m:
                            cascade_slots: set[int] = set()
                            for pid in reg_m["participants"]:
                                for s_idx, s_val in enumerate(agents[pid].calendar.slots):
                                    if isinstance(s_val, dict) and s_val.get("meeting_id") == mid_c:
                                        cascade_slots.add(s_idx)
                                        break
                            if len(cascade_slots) == 1:
                                meeting_registry[mid_c] = next(iter(cascade_slots))
                    meeting_registry[meeting["id"]] = chosen_slot
                    coordinated = True
                    per_agent_slot = {str(pid): chosen_slot for pid in meeting["participants"]}
                    self.events.append("fallback_applied", data={
                        "round": round_num, "turn": turn_index, "phase": "FALLBACK",
                        "agent_id": None,
                        "meeting_id": meeting["id"],
                        "chosen_slot": chosen_slot,
                        "displacement_plan": disp_plan,
                        "fallback_displacement_cost": sum(fallback_displacement_cost.values()),
                        "registry_cascades": registry_cascades,
                        "calendar_renders_after": {
                            str(pid): agents[pid].calendar.render()
                            for pid in meeting["participants"]
                        },
                    })
                except FallbackDepthExceeded as exc:
                    self.events.append("fallback_error", data={
                        "round": round_num, "turn": turn_index, "phase": "FALLBACK",
                        "agent_id": None,
                        "meeting_id": meeting["id"],
                        "reason": "depth_exceeded",
                        "depth": exc.depth,
                        "detail": str(exc),
                    })
                except FallbackImpossible as exc:
                    self.events.append("fallback_error", data={
                        "round": round_num, "turn": turn_index, "phase": "FALLBACK",
                        "agent_id": None,
                        "meeting_id": meeting["id"],
                        "reason": "no_feasible_slot",
                        "detail": str(exc),
                    })

            round_outcomes.append({
                "meeting_id": meeting["id"],
                "coordinated": coordinated,
                "per_agent_slot": per_agent_slot,
                "slot_conflicts": slot_conflicts,
                "consistency_violated_meeting_ids": sorted(consistency_violated_ids),
                "blocked_slot_violations": blocked_slot_violations,
            })

        # 7. Compute final metrics
        total_meetings = len(scenario["meetings"])
        coordinated_meetings = sum(1 for o in round_outcomes if o["coordinated"])
        coordination_rate = coordinated_meetings / total_meetings if total_meetings > 0 else 1.0

        agents_with_conflicts = sum(
            1 for agent_id in range(self.config.num_agents)
            if any(str(agent_id) in o["slot_conflicts"] for o in round_outcomes)
        )
        slot_conflict_rate = agents_with_conflicts / self.config.num_agents if self.config.num_agents > 0 else 0.0

        realized_cost = sum(displacement_cost.values())
        total_fallback_cost = sum(fallback_displacement_cost.values())
        optimal_cost = optimal.get("cost") or 0

        # efficiency: avg DMs sent per meeting scheduled (lower = more efficient)
        efficiency = total_dms_sent / coordinated_meetings if coordinated_meetings > 0 else float("inf")
        avg_dm_chars = total_dm_chars / total_dms_sent if total_dms_sent > 0 else 0.0
        dm_chars_per_meeting = total_dm_chars / coordinated_meetings if coordinated_meetings > 0 else float("inf")

        per_agent_cost_list = [displacement_cost[i] for i in range(self.config.num_agents)]
        per_agent_fallback_cost_list = [fallback_displacement_cost[i] for i in range(self.config.num_agents)]
        max_cost = max(per_agent_cost_list) if per_agent_cost_list else 0
        fairness = min(per_agent_cost_list) / max_cost if max_cost > 0 else 1.0

        metrics = {
            "coordination_rate": coordination_rate,
            "slot_conflict_rate": slot_conflict_rate,
            "efficiency": efficiency,
            "fairness": fairness,
            "meetings_scheduled": coordinated_meetings,
            "total_dms_sent": total_dms_sent,
            "total_dm_chars": total_dm_chars,
            "avg_dm_chars": avg_dm_chars,
            "max_dm_chars": max_dm_chars,
            "dm_chars_per_meeting": dm_chars_per_meeting,
            "realized_cost": realized_cost,
            "fallback_displacement_cost": total_fallback_cost,
            "optimal_cost": optimal_cost,
            "nosy_agent_count": len(nosy_agent_ids),
        }

        self.events.append("game_end", data={
            "round": len(scenario["meetings"]), "turn": 0, "phase": "GAME_END", "agent_id": None,
            **metrics,
        })

        return GameTraceBase(
            game_id=str(uuid.uuid4()),
            config=self.config,
            events=self.events.all(),
            final_state={
                "calendars": [agent.calendar.slots for agent in agents],
                "per_agent_cost": per_agent_cost_list,
                "per_agent_fallback_cost": per_agent_fallback_cost_list,
                "round_outcomes": round_outcomes,
                "nosy_agent_ids": nosy_agent_ids,
            },
            metrics=metrics,
        )


register_game("calendar", CalendarGame)
