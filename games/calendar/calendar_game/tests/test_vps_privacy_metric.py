from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_vps_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "scripts"
        / "rq5_vps_privacy_metric.py"
    )
    spec = importlib.util.spec_from_file_location("rq5_vps_privacy_metric", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vps_parser_distinguishes_blocked_errands_from_movable_errands():
    vps = _load_vps_module()
    render = "\n".join(
        [
            "Slot 0: [FREE]",
            "Slot 1: Blocked Errand #3 (cost=1)",
            "Slot 2: Errand #4 (cost=9)",
        ]
    )

    items = vps._slot_items_from_render(render, num_slots=3)

    assert items[0] is None
    assert items[1] == {"type": "blocked", "cost": 1, "blocked": True}
    assert items[2] == {"type": "errand", "cost": 9}


def test_vps_cost_weights_treat_blocked_slots_as_max_weight():
    vps = _load_vps_module()
    render = "\n".join(
        [
            "Slot 0: [FREE]",
            "Slot 1: Blocked Errand #3 (cost=1)",
            "Slot 2: Errand #4 (cost=9)",
        ]
    )

    weights = vps._slot_weights(render, num_slots=3, mode="cost", max_weight=32.0)

    assert weights == [1.0, 32.0, 4.0]


def test_vps_game_rows_keep_uniform_and_cost_modes_separate():
    vps = _load_vps_module()
    pair_rows = [
        {
            "trace_path": "/tmp/trace.json",
            "game_id": "g1",
            "target_is_participant": True,
            "observer_is_participant": True,
            "observations": 1,
            "vps_loss": 2.0,
            "weight_sum": 10.0,
            "weight_mode": "uniform",
        },
        {
            "trace_path": "/tmp/trace.json",
            "game_id": "g1",
            "target_is_participant": True,
            "observer_is_participant": True,
            "observations": 1,
            "vps_loss": 8.0,
            "weight_sum": 40.0,
            "weight_mode": "cost",
        },
    ]

    rows = vps._game_rows(pair_rows)

    by_mode = {row["weight_mode"]: row for row in rows}
    assert set(by_mode) == {"uniform", "cost"}
    assert by_mode["uniform"]["vps_loss_total"] == 2.0
    assert by_mode["cost"]["vps_loss_total"] == 8.0
    assert by_mode["cost"]["vps_loss_per_weight"] == 0.2
