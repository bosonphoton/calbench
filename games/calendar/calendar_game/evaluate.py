"""Summarize CalBench trace directories from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from calendar_game.dataset import CalendarGameDataset


SUMMARY_COLUMNS = [
    "n_games",
    "coordination_rate_mean",
    "meetings_scheduled_mean",
    "msgs_per_meeting_mean",
    "dm_chars_per_meeting_mean",
    "realized_cost_mean",
    "optimal_cost_mean",
    "excess_cost_mean",
    "cost_ratio_mean",
    "cost_gini_mean",
    "fairness_metric_mean",
]


def summarize_game_df(game_df: pd.DataFrame) -> pd.DataFrame:
    if game_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped = game_df.groupby(["experiment_name", "model_label"], dropna=False)
    rows = []
    for (experiment_name, model_label), group in grouped:
        rows.append({
            "experiment_name": experiment_name,
            "model_label": model_label,
            "n_games": len(group),
            "coordination_rate_mean": group["coordination_rate"].mean(),
            "meetings_scheduled_mean": group["meetings_scheduled"].mean(),
            "msgs_per_meeting_mean": group["msgs_per_meeting"].mean(),
            "dm_chars_per_meeting_mean": group["dm_chars_per_meeting"].mean(),
            "realized_cost_mean": group["realized_cost"].mean(),
            "optimal_cost_mean": group["optimal_cost"].mean(),
            "excess_cost_mean": group["excess_cost"].mean(),
            "cost_ratio_mean": group["cost_ratio"].mean(),
            "cost_gini_mean": group["cost_gini"].mean(),
            "fairness_metric_mean": group["fairness_metric"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["experiment_name", "model_label"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize CalBench JSON traces.")
    parser.add_argument("trace_dir", help="Trace directory, usually results/<experiment_name>")
    parser.add_argument("--game-csv", help="Optional path for one-row-per-game CSV output.")
    parser.add_argument("--round-csv", help="Optional path for one-row-per-round CSV output.")
    parser.add_argument("--agent-csv", help="Optional path for one-row-per-agent CSV output.")
    parser.add_argument("--message-csv", help="Optional path for one-row-per-DM CSV output.")
    parser.add_argument("--summary-csv", help="Optional path for grouped summary CSV output.")
    args = parser.parse_args(argv)

    ds = CalendarGameDataset.from_dir(args.trace_dir)
    game_df = ds.to_game_df()
    summary_df = summarize_game_df(game_df)

    if args.game_csv:
        Path(args.game_csv).parent.mkdir(parents=True, exist_ok=True)
        game_df.to_csv(args.game_csv, index=False)
    if args.round_csv:
        Path(args.round_csv).parent.mkdir(parents=True, exist_ok=True)
        ds.to_round_df().to_csv(args.round_csv, index=False)
    if args.agent_csv:
        Path(args.agent_csv).parent.mkdir(parents=True, exist_ok=True)
        ds.to_agent_df().to_csv(args.agent_csv, index=False)
    if args.message_csv:
        Path(args.message_csv).parent.mkdir(parents=True, exist_ok=True)
        ds.to_message_df().to_csv(args.message_csv, index=False)
    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_csv, index=False)

    if summary_df.empty:
        print(f"No traces found under {args.trace_dir}")
    else:
        print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
