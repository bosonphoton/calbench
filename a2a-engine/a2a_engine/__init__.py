"""a2a-engine: game-agnostic framework for agent-to-agent coordination experiments."""

from a2a_engine.agent import AgentInterface, LLMAgent
from a2a_engine.experiment import ExperimentSpec, BatchSpec, load_experiment, expand_batches
from a2a_engine.parallel import run_with_parallelism
from a2a_engine.schemas import AgentInfo, GameConfigBase, GameEvent, GameTraceBase
from a2a_engine.tracing import EventLog, write_trace, read_trace
from a2a_engine.registry import register_game, get_game, list_games
from a2a_engine.dataset import GameDataset, GameMessage, GameRecord

__all__ = [
    "AgentInterface",
    "LLMAgent",
    "ExperimentSpec",
    "BatchSpec",
    "load_experiment",
    "expand_batches",
    "run_with_parallelism",
    "AgentInfo",
    "GameConfigBase",
    "GameEvent",
    "GameTraceBase",
    "EventLog",
    "write_trace",
    "read_trace",
    "register_game",
    "get_game",
    "list_games",
    "GameDataset",
    "GameMessage",
    "GameRecord",
]
