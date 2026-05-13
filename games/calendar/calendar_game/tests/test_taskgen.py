from calendar_game.taskgen import build_task, build_tasks_from_config, derive_blocked_errand_task


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


def test_build_tasks_from_config_generates_at_least_ten_configurations():
    config = {
        "setting_name": "engine_smoke_10_configs",
        "seed_base": 700_000,
        "candidates_per_config": 3,
        "selected_per_bucket": 1,
        "configs": [
            {
                "total_agents": [2, 3],
                "subset_size": 2,
                "num_slots": 8,
                "num_meetings": 1,
                "densities": [0.25, 0.35, 0.45, 0.55, 0.65],
                "pref_level": 1,
                "meeting_cost_level": 1,
                "errand_cost_level": 1,
            }
        ],
    }

    tasks, summary = build_tasks_from_config(config)

    assert summary["total_configs"] == 10
    assert summary["total_tasks"] == 30
    assert len(tasks) == 30
    assert summary["bucket_counts"] == {"easy": 10, "medium": 10, "hard": 10}
    assert len({
        (
            task["params"]["total_agents"],
            task["params"]["subset_size"],
            task["params"]["density"],
            task["params"]["pref_level"],
            task["params"]["num_meetings"],
            task["params"]["num_slots"],
        )
        for task in tasks
    }) == 10
    assert all(task["feasible"] for task in tasks)
    assert all(task["optimal"]["cost"] is not None for task in tasks)
