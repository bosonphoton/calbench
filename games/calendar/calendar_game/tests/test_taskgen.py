from calendar_game.taskgen import build_task, derive_blocked_errand_task


def test_derive_blocked_errand_task_remains_feasible():
    source = build_task({
        "task_id": "source_5a3p5m",
        "seed": 323032,
        "total_agents": 5,
        "subset_size": 3,
        "density": 0.8,
        "pref_level": 1,
        "num_meetings": 5,
    })

    blocked = derive_blocked_errand_task(source, blocked_errands_per_agent=1)

    assert blocked["feasible"]
    assert blocked["optimal"]["cost"] is not None
    assert blocked["greedy"]["cost"] is not None
    assert blocked["params"]["blocked_errands_per_agent"] == 1
    assert blocked["blocked_errand_count"] == 5
    for calendar in blocked["calendars"]:
        assert sum(
            1
            for slot in calendar
            if isinstance(slot, dict) and slot.get("blocked")
        ) == 1
