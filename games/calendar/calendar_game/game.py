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
from calendar_game.solver import cost_by_agent_for_assignments, solve_greedy, solve_optimal
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
    agent_densities: list[float] | dict[int, float] | dict[str, float] | None = None
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
    communication_protocol: str | dict[str, bool] = "dm"
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
    representation_elo_base: float = 1500.0
    representation_elo_scale: float = 400.0


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

    def _communication_channels(self) -> set[str]:
        raw_protocol = self.config.communication_protocol or "dm"
        aliases = {
            "direct": {"dm"},
            "direct_message": {"dm"},
            "private": {"dm"},
            "participant_groupchat": {"participant_groupchat"},
            "participant_chat": {"participant_groupchat"},
            "meeting_groupchat": {"participant_groupchat"},
            "meeting_chat": {"participant_groupchat"},
            "group": {"all_groupchat"},
            "groupchat": {"all_groupchat"},
            "group_chat": {"all_groupchat"},
            "all_groupchat": {"all_groupchat"},
            "all_agent_groupchat": {"all_groupchat"},
            "all_agent_chat": {"all_groupchat"},
            "dm_and_groupchat": {"dm", "all_groupchat"},
            "dm_and_all_groupchat": {"dm", "all_groupchat"},
            "dm_and_participant_groupchat": {"dm", "participant_groupchat"},
            "both": {"dm", "all_groupchat"},
            "mixed": {"dm", "all_groupchat"},
            "all": {"dm", "participant_groupchat", "all_groupchat"},
        }
        valid = {"dm", "participant_groupchat", "all_groupchat"}
        if isinstance(raw_protocol, dict):
            channels: set[str] = set()
            for key, enabled in raw_protocol.items():
                if not enabled:
                    continue
                normalized = str(key).lower()
                channels.update(aliases.get(normalized, {normalized}))
        else:
            protocol = str(raw_protocol).lower()
            channels = set(aliases.get(protocol, {protocol}))
        if not channels or any(channel not in valid for channel in channels):
            raise ValueError(
                "communication_protocol must enable one or more of: "
                "dm, participant_groupchat, all_groupchat"
            )
        return channels

    def _communication_protocol(self) -> str:
        channels = self._communication_channels()
        order = ["dm", "participant_groupchat", "all_groupchat"]
        return "+".join(channel for channel in order if channel in channels)

    def _allows_channel(self, channel: str) -> bool:
        return channel in self._communication_channels()

    def _allows_dm(self) -> bool:
        return self._allows_channel("dm")

    def _allows_participant_groupchat(self) -> bool:
        return self._allows_channel("participant_groupchat")

    def _allows_all_groupchat(self) -> bool:
        return self._allows_channel("all_groupchat")

    def _allows_groupchat(self) -> bool:
        return self._allows_participant_groupchat() or self._allows_all_groupchat()

    def _canonical_tool_type(self, tool_type: object) -> str:
        protocol = str(tool_type or "").lower()
        aliases = {
            "groupchat": "all_groupchat",
            "group_chat": "all_groupchat",
            "group": "all_groupchat",
            "all_agent_groupchat": "all_groupchat",
            "all_agent_chat": "all_groupchat",
            "participant_chat": "participant_groupchat",
            "meeting_groupchat": "participant_groupchat",
            "meeting_chat": "participant_groupchat",
        }
        return aliases.get(protocol, protocol)

    def _agent_spec_for(self, agent_id: int) -> dict:
        if agent_id < len(self.config.agents):
            spec = self.config.agents[agent_id]
            return spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
        return {"type": "llm", "model": "gpt-4o-mini"}

    def _agent_team_metadata(self) -> dict:
        specs = [self._agent_spec_for(agent_id) for agent_id in range(self.config.num_agents)]
        models = [str(spec.get("model") or spec.get("type") or "unknown") for spec in specs]
        types = [str(spec.get("type") or "unknown") for spec in specs]
        model_counts: dict[str, int] = {}
        for model in models:
            model_counts[model] = model_counts.get(model, 0) + 1
        return {
            "agent_models": models,
            "agent_types": types,
            "team_model_counts": model_counts,
            "team_model_label": " vs ".join(dict.fromkeys(models)),
            "is_heterogeneous_team": len(set(models)) > 1 or len(set(types)) > 1,
        }

    @staticmethod
    def _actual_calendar_densities(calendars: list[list[object]]) -> list[float]:
        densities: list[float] = []
        for calendar in calendars:
            if not calendar:
                densities.append(0.0)
                continue
            occupied = sum(1 for slot in calendar if slot is not None)
            densities.append(occupied / len(calendar))
        return densities

    def _representation_elo_by_agent(self, calendar_densities: list[float]) -> list[float]:
        if not calendar_densities:
            return []
        mean_density = sum(calendar_densities) / len(calendar_densities)
        return [
            round(
                self.config.representation_elo_base
                + self.config.representation_elo_scale * (density - mean_density),
                3,
            )
            for density in calendar_densities
        ]

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
            agent_densities=self.config.agent_densities,
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
                    "agent_densities": task.get("params", {}).get("agent_densities"),
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
                cfg = self._agent_spec_for(agent_id)
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
        communication_protocol = self._communication_protocol()
        team_metadata = self._agent_team_metadata()
        initial_calendar_densities = self._actual_calendar_densities(scenario["calendars"])
        representation_elo_by_agent = self._representation_elo_by_agent(initial_calendar_densities)

        # 3. game_start event — emitted before agent registration so it is always first
        self.events.append("game_start", data={
            "round": -1, "turn": -1, "phase": "GAME_START", "agent_id": None,
            "scenario_seed": self.config.seed,
            "num_agents": self.config.num_agents,
            "num_slots": self.config.num_slots,
            "optimal_cost": optimal.get("cost"),
            "greedy_cost": greedy.get("cost"),
            "nosy_agent_ids": nosy_agent_ids,
            "communication_protocol": communication_protocol,
            "agent_densities": scenario.get("agent_densities") or self.config.agent_densities,
            "initial_calendar_density_by_agent": initial_calendar_densities,
            "representation_elo_by_agent": representation_elo_by_agent,
            **team_metadata,
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
                communication_protocol=communication_protocol,
            )
            agent.register(agent_id, game_config)
            system_prompt_text = getattr(agent.client, "_system_prompt", None) or build_system_prompt(
                dataclasses.asdict(game_config)
            )
            self.events.append("agent_registered", data={
                "round": -1, "turn": -1, "phase": "GAME_START", "agent_id": agent_id,
                "is_nosy_agent": agent_id in nosy_agent_ids,
                "nosy_agent_ids": nosy_agent_ids,
                "model": team_metadata["agent_models"][agent_id],
                "agent_type": team_metadata["agent_types"][agent_id],
                "initial_calendar_density": initial_calendar_densities[agent_id],
                "representation_elo": representation_elo_by_agent[agent_id],
                "system_prompt": system_prompt_text,
                "calendar_render": agent.calendar.render(),
            })

        # 5. Per-game accumulators
        displacement_cost: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        fallback_displacement_cost: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        total_client_calls: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        total_dms_sent: int = 0
        total_participant_groupchat_messages_sent: int = 0
        total_all_groupchat_messages_sent: int = 0
        total_dm_chars: int = 0
        total_participant_groupchat_chars: int = 0
        total_all_groupchat_chars: int = 0
        max_dm_chars: int = 0
        max_participant_groupchat_chars: int = 0
        max_all_groupchat_chars: int = 0
        messages_sent_by_agent: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
        messages_received_by_agent: dict[int, int] = {i: 0 for i in range(self.config.num_agents)}
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

        def deliver_cheap_talk_tool(
            *,
            tool: dict,
            agent_id: int,
            meeting: dict,
            round_num: int,
            turn_index: int,
            already_queued: set[int],
        ) -> bool:
            nonlocal total_dms_sent
            nonlocal total_participant_groupchat_messages_sent
            nonlocal total_all_groupchat_messages_sent
            nonlocal total_dm_chars
            nonlocal total_participant_groupchat_chars
            nonlocal total_all_groupchat_chars
            nonlocal max_dm_chars
            nonlocal max_participant_groupchat_chars
            nonlocal max_all_groupchat_chars

            tool_type = self._canonical_tool_type(tool.get("type"))
            if tool_type == "dm":
                if not self._allows_dm():
                    self._invalid_tool_call(
                        round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                        agent_id=agent_id, tool=tool, reason="dm tool is disabled by communication_protocol",
                    )
                    return False
                try:
                    to = int(tool["to"])
                except (KeyError, TypeError, ValueError):
                    self._invalid_tool_call(
                        round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                        agent_id=agent_id, tool=tool, reason="dm tool missing integer 'to'",
                    )
                    return False
                if to < 0 or to >= self.config.num_agents:
                    self._invalid_tool_call(
                        round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                        agent_id=agent_id, tool=tool, reason="dm recipient is out of range",
                    )
                    return False
                msg = {
                    "from": agent_id,
                    "to": to,
                    "channel": "dm",
                    "meeting_id": meeting["id"],
                    "content": str(tool.get("content", "")),
                }
                dm_chars = len(msg["content"])
                agents[to].inbox_queue.append(msg)
                messages_sent_by_agent[agent_id] += 1
                messages_received_by_agent[to] += 1
                total_dms_sent += 1
                total_dm_chars += dm_chars
                max_dm_chars = max(max_dm_chars, dm_chars)
                self.events.append("dm_sent", data={
                    "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                    "agent_id": agent_id,
                    "from_agent": agent_id, "to_agent": to,
                    "meeting_id": msg["meeting_id"], "content": msg["content"],
                    "content_chars": dm_chars,
                    "channel": "dm",
                })
                if to not in meeting["participants"] and to not in already_queued:
                    already_queued.add(to)
                return True

            if tool_type in {"participant_groupchat", "all_groupchat"}:
                if not self._allows_channel(tool_type):
                    self._invalid_tool_call(
                        round_num=round_num, turn=turn_index, phase="CHEAP_TALK",
                        agent_id=agent_id, tool=tool,
                        reason=f"{tool_type} tool is disabled by communication_protocol",
                    )
                    return False
                content = str(tool.get("content", ""))
                msg_chars = len(content)
                if tool_type == "participant_groupchat":
                    recipients = [
                        to for to in meeting["participants"]
                        if to != agent_id
                    ]
                    event_type = "participant_groupchat_sent"
                else:
                    recipients = [to for to in all_agent_ids if to != agent_id]
                    event_type = "all_groupchat_sent"
                for to in recipients:
                    agents[to].inbox_queue.append({
                        "from": agent_id,
                        "to": None,
                        "channel": tool_type,
                        "meeting_id": meeting["id"],
                        "content": content,
                    })
                    messages_received_by_agent[to] += 1
                    if tool_type == "all_groupchat" and to not in meeting["participants"]:
                        already_queued.add(to)
                messages_sent_by_agent[agent_id] += 1
                if tool_type == "participant_groupchat":
                    total_participant_groupchat_messages_sent += 1
                    total_participant_groupchat_chars += msg_chars
                    max_participant_groupchat_chars = max(max_participant_groupchat_chars, msg_chars)
                else:
                    total_all_groupchat_messages_sent += 1
                    total_all_groupchat_chars += msg_chars
                    max_all_groupchat_chars = max(max_all_groupchat_chars, msg_chars)
                self.events.append(event_type, data={
                    "round": round_num, "turn": turn_index, "phase": "CHEAP_TALK",
                    "agent_id": agent_id,
                    "from_agent": agent_id,
                    "to_agents": recipients,
                    "meeting_id": meeting["id"],
                    "content": content,
                    "content_chars": msg_chars,
                    "channel": tool_type,
                })
                return True

            return False

        # 6. Main loop — one round per meeting
        for round_num, meeting in enumerate(scenario["meetings"]):
            speaker_order = self._speaker_order(meeting)
            self.events.append("round_start", data={
                "round": round_num, "turn": 0, "phase": "CHEAP_TALK", "agent_id": None,
                "meeting": meeting,
                "speaker_order": speaker_order,
                "communication_protocol": communication_protocol,
            })

            active_agent_ids: set[int] = set(speaker_order)
            active_agent_order = list(speaker_order)

            # call start_round for all active cheap-talk agents
            for agent_id in active_agent_order:
                agents[agent_id].start_round(meeting, round_num, incurred_penalty=displacement_cost[agent_id])

            # per-round state
            already_queued: set[int] = set()
            turn_index = 0
            blocked_slot_violations: list[dict] = []

            # --- CHEAP_TALK PHASE ---
            has_activity = True
            while has_activity and turn_index < self.config.max_turns_per_round:
                has_activity = False

                # active cheap-talk agents act
                for agent_id in list(active_agent_order):
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
                            communication_protocol=communication_protocol,
                        )
                        if turn_index == 0
                        else build_turn_message(
                            inbox_snapshot,
                            turn_index,
                            self.config.max_turns_per_round,
                            communication_protocol=communication_protocol,
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
                        if deliver_cheap_talk_tool(
                            tool=tool,
                            agent_id=agent_id,
                            meeting=meeting,
                            round_num=round_num,
                            turn_index=turn_index,
                            already_queued=already_queued,
                        ):
                            has_activity = True

                # drain unique_queue (non-participants who got DMs or all-agent groupchat)
                queue = sorted(already_queued - active_agent_ids)
                for agent_id in queue:
                    active_agent_ids.add(agent_id)
                    active_agent_order.append(agent_id)
                    agents[agent_id].start_round(
                        meeting,
                        round_num,
                        incurred_penalty=displacement_cost[agent_id],
                    )
                    agent = agents[agent_id]
                    inbox_snapshot = list(agents[agent_id].inbox_queue)
                    calendar_render = agents[agent_id].calendar.render()
                    prompt_calendar_render = self._prompt_calendar_for_agent(agent, calendar_render, round_num)
                    turn_prompt = build_turn_message(
                        inbox_snapshot,
                        turn_index,
                        self.config.max_turns_per_round,
                        communication_protocol=communication_protocol,
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
                        if deliver_cheap_talk_tool(
                            tool=tool,
                            agent_id=agent_id,
                            meeting=meeting,
                            round_num=round_num,
                            turn_index=turn_index,
                            already_queued=already_queued,
                        ):
                            has_activity = True

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
        replayed_oracle_cost, oracle_per_agent_cost = cost_by_agent_for_assignments(
            scenario["calendars"],
            scenario["meetings"],
            optimal.get("assignments", {}),
        )
        if replayed_oracle_cost is None:
            oracle_per_agent_cost = [0 for _ in range(self.config.num_agents)]
        per_agent_cost_list = [displacement_cost[i] for i in range(self.config.num_agents)]
        per_agent_fallback_cost_list = [fallback_displacement_cost[i] for i in range(self.config.num_agents)]
        per_agent_excess_burden = [
            per_agent_cost_list[i] - oracle_per_agent_cost[i]
            for i in range(self.config.num_agents)
        ]

        # efficiency: avg DMs sent per meeting scheduled (lower = more efficient)
        efficiency = total_dms_sent / coordinated_meetings if coordinated_meetings > 0 else float("inf")
        avg_dm_chars = total_dm_chars / total_dms_sent if total_dms_sent > 0 else 0.0
        dm_chars_per_meeting = total_dm_chars / coordinated_meetings if coordinated_meetings > 0 else float("inf")

        max_cost = max(per_agent_cost_list) if per_agent_cost_list else 0
        fairness = min(per_agent_cost_list) / max_cost if max_cost > 0 else 1.0
        participant_rounds = {i: 0 for i in range(self.config.num_agents)}
        coordinated_participant_rounds = {i: 0 for i in range(self.config.num_agents)}
        for outcome, meeting in zip(round_outcomes, scenario["meetings"], strict=False):
            for agent_id in meeting["participants"]:
                participant_rounds[agent_id] += 1
                if outcome["coordinated"]:
                    coordinated_participant_rounds[agent_id] += 1

        total_groupchat_messages_sent = (
            total_participant_groupchat_messages_sent
            + total_all_groupchat_messages_sent
        )
        total_cheap_talk_messages = total_dms_sent + total_groupchat_messages_sent
        total_groupchat_chars = total_participant_groupchat_chars + total_all_groupchat_chars
        max_groupchat_chars = max(max_participant_groupchat_chars, max_all_groupchat_chars)
        max_representation_elo = max(representation_elo_by_agent) if representation_elo_by_agent else 1.0
        contribution_scores: list[dict] = []
        for agent_id in range(self.config.num_agents):
            participation_count = participant_rounds[agent_id]
            coordination_rate_for_agent = (
                coordinated_participant_rounds[agent_id] / participation_count
                if participation_count > 0
                else 0.0
            )
            communication_share = (
                messages_sent_by_agent[agent_id] / total_cheap_talk_messages
                if total_cheap_talk_messages > 0
                else 0.0
            )
            cost_efficiency = (
                1.0 - (per_agent_cost_list[agent_id] / max_cost)
                if max_cost > 0
                else 1.0
            )
            raw_score = (
                100.0
                * (
                    0.60 * coordination_rate_for_agent
                    + 0.25 * cost_efficiency
                    + 0.15 * communication_share
                )
            )
            representation_weight = (
                representation_elo_by_agent[agent_id] / max_representation_elo
                if max_representation_elo > 0
                else 1.0
            )
            model = team_metadata["agent_models"][agent_id]
            contribution_scores.append({
                "agent_id": agent_id,
                "model": model,
                "agent_type": team_metadata["agent_types"][agent_id],
                "participant_rounds": participation_count,
                "coordinated_participant_rounds": coordinated_participant_rounds[agent_id],
                "coordination_rate": round(coordination_rate_for_agent, 6),
                "messages_sent": messages_sent_by_agent[agent_id],
                "messages_received": messages_received_by_agent[agent_id],
                "communication_share": round(communication_share, 6),
                "cost": per_agent_cost_list[agent_id],
                "oracle_cost": oracle_per_agent_cost[agent_id],
                "excess_burden": per_agent_excess_burden[agent_id],
                "fallback_cost": per_agent_fallback_cost_list[agent_id],
                "cost_efficiency": round(cost_efficiency, 6),
                "calendar_density": round(initial_calendar_densities[agent_id], 6),
                "representation_elo": representation_elo_by_agent[agent_id],
                "contribution_score": round(raw_score, 6),
                "density_adjusted_contribution_score": round(raw_score * representation_weight, 6),
            })

        model_contribution_summary: dict[str, dict] = {}
        for model in sorted(set(team_metadata["agent_models"])):
            rows = [row for row in contribution_scores if row["model"] == model]
            if not rows:
                continue
            model_contribution_summary[model] = {
                "agent_count": len(rows),
                "mean_contribution_score": round(
                    sum(float(row["contribution_score"]) for row in rows) / len(rows),
                    6,
                ),
                "mean_density_adjusted_contribution_score": round(
                    sum(float(row["density_adjusted_contribution_score"]) for row in rows) / len(rows),
                    6,
                ),
                "total_messages_sent": sum(int(row["messages_sent"]) for row in rows),
                "total_cost": sum(float(row["cost"]) for row in rows),
            }

        metrics = {
            "coordination_rate": coordination_rate,
            "slot_conflict_rate": slot_conflict_rate,
            "efficiency": efficiency,
            "fairness": fairness,
            "meetings_scheduled": coordinated_meetings,
            "total_dms_sent": total_dms_sent,
            "total_groupchat_messages_sent": total_groupchat_messages_sent,
            "total_participant_groupchat_messages_sent": total_participant_groupchat_messages_sent,
            "total_all_groupchat_messages_sent": total_all_groupchat_messages_sent,
            "total_cheap_talk_messages": total_cheap_talk_messages,
            "total_dm_chars": total_dm_chars,
            "total_groupchat_chars": total_groupchat_chars,
            "total_participant_groupchat_chars": total_participant_groupchat_chars,
            "total_all_groupchat_chars": total_all_groupchat_chars,
            "avg_dm_chars": avg_dm_chars,
            "max_dm_chars": max_dm_chars,
            "max_groupchat_chars": max_groupchat_chars,
            "max_participant_groupchat_chars": max_participant_groupchat_chars,
            "max_all_groupchat_chars": max_all_groupchat_chars,
            "dm_chars_per_meeting": dm_chars_per_meeting,
            "realized_cost": realized_cost,
            "fallback_displacement_cost": total_fallback_cost,
            "optimal_cost": optimal_cost,
            "oracle_cost_replayed": replayed_oracle_cost,
            "oracle_per_agent_cost": oracle_per_agent_cost,
            "per_agent_excess_burden": per_agent_excess_burden,
            "total_excess_burden": sum(per_agent_excess_burden),
            "nosy_agent_count": len(nosy_agent_ids),
            "communication_protocol": communication_protocol,
            "team_model_label": team_metadata["team_model_label"],
            "team_model_counts": team_metadata["team_model_counts"],
            "is_heterogeneous_team": team_metadata["is_heterogeneous_team"],
            "mean_contribution_score": (
                sum(float(row["contribution_score"]) for row in contribution_scores) / len(contribution_scores)
                if contribution_scores
                else 0.0
            ),
            "model_contribution_summary": model_contribution_summary,
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
                "oracle_per_agent_cost": oracle_per_agent_cost,
                "per_agent_excess_burden": per_agent_excess_burden,
                "per_agent_messages_sent": [
                    messages_sent_by_agent[i] for i in range(self.config.num_agents)
                ],
                "per_agent_messages_received": [
                    messages_received_by_agent[i] for i in range(self.config.num_agents)
                ],
                "initial_calendar_density_by_agent": initial_calendar_densities,
                "representation_elo_by_agent": representation_elo_by_agent,
                "contribution_scores": contribution_scores,
                "model_contribution_summary": model_contribution_summary,
                "round_outcomes": round_outcomes,
                "nosy_agent_ids": nosy_agent_ids,
                **team_metadata,
            },
            metrics=metrics,
        )


register_game("calendar", CalendarGame)
