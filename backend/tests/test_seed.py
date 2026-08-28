"""
Tests for db.seed.task_derived_schedules -- the function that connects
the Dependency Graph's per-task load numbers to the drift formula's
per-astronaut daily workload signal.
"""

from db.seed import task_derived_schedules, DEFAULT_TASKS, DEFAULT_CREW


def test_sums_multiple_tasks_on_same_astronaut_day():
    """A1 has exactly one task on each of days 1-6 in the default set,
    so this also implicitly checks the single-task case; construct a
    crew/day combo where summation actually matters."""
    from simulator import AstronautProfile
    schedules = task_derived_schedules(DEFAULT_CREW, 6)
    # A1: T1(day1,load3), T4(day2,load4), T7(day3,load6), T11(day4,load2), T13(day5,load5), T16(day6,load6)
    assert schedules["A1"] == [3.0, 4.0, 6.0, 2.0, 5.0, 6.0]


def test_falls_back_to_default_schedule_for_uncovered_days():
    """A crew whose astronaut_id doesn't appear in DEFAULT_TASKS at all
    should fall back to DEFAULT_TASK_SCHEDULE for every day, not zeros."""
    from simulator import AstronautProfile, DEFAULT_TASK_SCHEDULE
    custom_crew = [AstronautProfile(astronaut_id="ZZ", name="Nobody", seed=99)]
    schedules = task_derived_schedules(custom_crew, 6)
    assert schedules["ZZ"] == list(DEFAULT_TASK_SCHEDULE)


def test_partial_coverage_mixes_task_derived_and_fallback_per_day():
    """A custom crew that happens to share an astronaut_id with some
    (but not all) default tasks should get real sums on covered days
    and the fallback on uncovered ones -- not fail or zero out."""
    from simulator import AstronautProfile, DEFAULT_TASK_SCHEDULE
    custom_crew = [AstronautProfile(astronaut_id="A1", name="Chen", seed=1)]
    schedules = task_derived_schedules(custom_crew, 8)  # longer than the 6-day task set
    assert schedules["A1"][:6] == [3.0, 4.0, 6.0, 2.0, 5.0, 6.0]
    assert schedules["A1"][6] == DEFAULT_TASK_SCHEDULE[-1]  # day 7 has no task -> fallback
    assert schedules["A1"][7] == DEFAULT_TASK_SCHEDULE[-1]  # day 8 likewise


def test_every_default_crew_member_has_a_full_schedule():
    schedules = task_derived_schedules(DEFAULT_CREW, 6)
    assert set(schedules.keys()) == {"A1", "A2", "A3"}
    for aid, sched in schedules.items():
        assert len(sched) == 6
        assert all(v >= 0 for v in sched)