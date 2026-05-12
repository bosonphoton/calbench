#!/usr/bin/env python3
"""Compute direct-message volume from completed calendar traces."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


PROTOCOL_KEYS = {"dsm", "imap", "sd"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    if root_candidate.exists():
        return root_candidate
    return Path.cwd() / candidate


def _trace_paths(inputs: Iterable[str], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = _resolve(raw, root=root)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        elif any(char in raw for char in "*?[]"):
            paths.extend(sorted(Path().glob(raw)))
        else:
            paths.append(path)
    return [
        path for path in paths
        if path.is_file()
        and not path.name.endswith(".metadata.json")
        and path.name != "_run_manifest.jsonl"
    ]


def _mean(values: list[float | int]) -> float | None:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.fmean(clean) if clean else None


def _median(values: list[float | int]) -> float | None:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return float(statistics.median(clean)) if clean else None


def _is_protocol_payload(parsed: object) -> bool:
    return isinstance(parsed, dict) and bool(PROTOCOL_KEYS.intersection(parsed.keys()))


def _semantic_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float, str)):
        return [str(value)]
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_semantic_tokens(item))
        return tokens
    if isinstance(value, dict):
        tokens = []
        for key, item in value.items():
            tokens.append(str(key))
            tokens.extend(_semantic_tokens(item))
        return tokens
    return [str(value)]


def _message_volume(content: object) -> tuple[int, str, str]:
    text = "" if content is None else str(content)
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return len(text), "natural_text", text
    if not _is_protocol_payload(parsed):
        return len(text), "natural_text", text

    # Baseline protocols are serialized as structured JSON, so raw character
    # counts would mostly measure transport syntax rather than communication.
    # For DSM/IMAP/SD payloads we sanitize away JSON-only syntax characters
    # `{`, `}`, `[`, `]`, `:`, `,`, and `"` plus formatting whitespace/newlines
    # introduced by serialization. We then count a normalized semantic string
    # made from object keys and primitive values separated by single spaces.
    normalized = " ".join(token for token in _semantic_tokens(parsed) if token)
    return len(normalized), "protocol_json_semantic", normalized


def _setting(trace: dict[str, Any], path: Path) -> str:
    config = trace.get("config") or {}
    text = f"{config.get('experiment_name') or path.parent.parent.name} {config.get('task_path') or ''}".lower()
    if "uniform" in text:
        return "uniform"
    if "varied" in text or "variable" in text:
        return "varied"
    return "unknown"


def _summarize_trace(path: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    rel = str(path.resolve().relative_to(root.resolve())) if path.is_relative_to(root) else str(path)
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(trace.get("events") or []):
        if event.get("type") != "dm_sent":
            continue
        data = event.get("data") or {}
        volume, mode, normalized = _message_volume(data.get("content"))
        rows.append({
            "trace_path": rel,
            "game_id": trace.get("game_id") or path.stem,
            "event_index": event_index,
            "round": data.get("round"),
            "turn": data.get("turn"),
            "from_agent": data.get("from_agent", data.get("agent_id")),
            "to_agent": data.get("to_agent"),
            "meeting_id": data.get("meeting_id"),
            "mode": mode,
            "raw_chars": len("" if data.get("content") is None else str(data.get("content"))),
            "message_volume_chars": volume,
            "normalized_preview": normalized[:160],
        })
    volumes = [int(row["message_volume_chars"]) for row in rows]
    summary = {
        "trace_path": rel,
        "game_id": trace.get("game_id") or path.stem,
        "setting": _setting(trace, path),
        "message_volume_count": len(volumes),
        "message_volume_chars_total": sum(volumes),
        "message_volume_chars_mean": _mean(volumes),
        "message_volume_chars_median": _median(volumes),
        "message_volume_chars_max": max(volumes) if volumes else None,
        "natural_text_message_count": sum(1 for row in rows if row["mode"] == "natural_text"),
        "protocol_json_semantic_message_count": sum(1 for row in rows if row["mode"] == "protocol_json_semantic"),
    }
    return summary, rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trace JSON files or directories.")
    parser.add_argument("--out-dir", default="analysis/outputs/message_volume_metric")
    args = parser.parse_args()

    root = _calendar_root()
    paths = _trace_paths(args.inputs, root=root)
    out_dir = _resolve(args.out_dir, root=root)

    summaries: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for path in paths:
        summary, rows = _summarize_trace(path, root=paths[0].parent if len(paths) == 1 else Path(args.inputs[0]).resolve())
        summaries.append(summary)
        messages.extend(rows)

    _write_csv(
        out_dir / "game_summary.csv",
        summaries,
        [
            "trace_path",
            "game_id",
            "setting",
            "message_volume_count",
            "message_volume_chars_total",
            "message_volume_chars_mean",
            "message_volume_chars_median",
            "message_volume_chars_max",
            "natural_text_message_count",
            "protocol_json_semantic_message_count",
        ],
    )
    _write_csv(
        out_dir / "message_summary.csv",
        messages,
        [
            "trace_path",
            "game_id",
            "event_index",
            "round",
            "turn",
            "from_agent",
            "to_agent",
            "meeting_id",
            "mode",
            "raw_chars",
            "message_volume_chars",
            "normalized_preview",
        ],
    )

    trace_means = [row["message_volume_chars_mean"] for row in summaries if row["message_volume_chars_mean"] is not None]
    summary = {
        "generated_at": _now(),
        "metric": "message_volume",
        "metric_direction": "lower_is_better",
        "trace_count": len(summaries),
        "message_count": len(messages),
        "message_volume_chars_mean": _mean([int(row["message_volume_chars"]) for row in messages]),
        "message_volume_chars_median": _median([int(row["message_volume_chars"]) for row in messages]),
        "trace_message_volume_chars_mean": _mean(trace_means),
        "trace_message_volume_chars_median": _median(trace_means),
        "natural_text_message_count": sum(1 for row in messages if row["mode"] == "natural_text"),
        "protocol_json_semantic_message_count": sum(1 for row in messages if row["mode"] == "protocol_json_semantic"),
        "sanitization_note": (
            "LLM/natural-text DMs count exact dm_sent.data.content characters. "
            "DSM/IMAP/SD baseline JSON DMs count normalized semantic keys and primitive values; "
            "JSON syntax characters { } [ ] : , \" and serialization whitespace/newlines are excluded."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(out_dir)
    print(f"traces={len(summaries)} messages={len(messages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
