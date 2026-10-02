#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
#
# [tool.uv]
# # As sweep.py: never the system interpreter.
# python-preference = "only-managed"
# ///

"""How tools/zarr-bench/sweep.py ranks what carta-zarr-bench measured, asked with rows written out here.

The end-to-end test runs every stage on a fixture far too small to time, and checks that the report has
every section. What it cannot reach is the ranking itself -- which operations a group is ranked on, which
mode a slowdown is blamed on, which settings may be recommended -- and that is where both of the bugs
found on a real cube were: a7db66b and bed82ad. Here a ranking is made from rows, so every answer is one
the test chose."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "zarr-bench"))

import sweep  # noqa: E402

CORES = 8
# The baseline the defaults resolve to on CORES cores, as the CSV spells a setting.
BASELINE = ("2", str(CORES), str(1024 << 20), "0")


def config(**given: Any) -> dict[str, Any]:
    return sweep.merge(sweep.DEFAULTS, given)


def layout(name: str, current: bool = False) -> sweep.Layout:
    return sweep.Layout(name, chunk=f"l={len(name)}", current=current)


def datasets(*layouts: sweep.Layout, validate: bool = False) -> dict[str, Any]:
    """The datasets table of a sweep's state, as though each layout had been written and measured."""
    return {sweep.dataset_key(each, validate): {"identity_hash": f"id-{each.name}{'-v' if validate else ''}"}
            for each in layouts}


def row(on: sweep.Layout, mode: str, seconds: float, stage: str = "stage1", setting: tuple[str, ...] = BASELINE,
        users: int = 1, validate: bool = False, **fields: str) -> dict[str, str]:
    """One operation as carta-zarr-bench writes it, with everything the ranking does not care about
    filled in alike."""
    io, decode, cache, budget = setting
    values = {
        "label": stage, "dataset_identity_hash": f"id-{on.name}{'-v' if validate else ''}",
        "io_threads": io, "decode_threads": decode, "cache_bytes": cache, "read_budget_bytes": budget,
        "histogram_method": "", "processes": str(users), "mode": mode, "chunk_shape": "", "shape": "",
        "frame_first_s": "", "shares_chunks": "false", "cold_method": "off", "cold_ok": "true",
        "build_type": "Release", "tuning": "default", "bench_commit": "test", "status": "ok",
        "seconds": repr(seconds), "logical_bytes": "1048576", "elements": "262144", "late_frames": "",
        "late_max_s": "", "t_end_s": "", "trial": "0", "checksum": "", "process_index": "0", "op_index": "0",
        "position": "pol=0;z=0", "animation_frames": "",
    }
    values.update(fields)
    return values


class RankingOnFirstTouches(unittest.TestCase):
    """A group is ranked on its first touches -- operations that read no chunk another read before them
    -- while it has enough of them, and on every operation otherwise. a7db66b: when one layout's group
    has too few, every group it is compared with has to be ranked on every operation too, or that layout
    is ranked on cache hits its rivals were spared."""

    def test_a_short_group_ranks_the_whole_comparison_on_every_operation(self) -> None:
        shallow, deep = layout("shallow"), layout("deep")
        rows = []
        # Shallow: two first touches at a second each, eight cache hits at a tenth.
        rows += [row(shallow, "plane", 1.0, op_index=str(i)) for i in range(2)]
        rows += [row(shallow, "plane", 0.1, op_index=str(i), shares_chunks="true") for i in range(2, 10)]
        # Deep: ten first touches at half a second.
        rows += [row(deep, "plane", 0.5, op_index=str(i)) for i in range(10)]
        results = sweep.Results(rows, datasets(shallow, deep))

        baseline = sweep.setting_of(sweep.DEFAULTS["baseline"], CORES)
        short = results.get("stage1", sweep.dataset_key(shallow, False), baseline, 1, "plane")
        rival = results.get("stage1", sweep.dataset_key(deep, False), baseline, 1, "plane")
        self.assertTrue(short.short, "two first touches of ten is too few to rank on")
        self.assertTrue(short.every_operation and rival.every_operation,
                        "and so the whole comparison is ranked on every operation, not just the short group")
        self.assertAlmostEqual(short.median, 0.1, msg="shallow's median over all ten of its operations")
        self.assertEqual(len(rival.sample), 10)

    def test_enough_first_touches_rank_on_them_alone(self) -> None:
        one = layout("one")
        rows = [row(one, "plane", 1.0, op_index=str(i)) for i in range(5)]
        rows += [row(one, "plane", 0.1, op_index=str(i), shares_chunks="true") for i in range(5, 10)]
        stats = sweep.Results(rows, datasets(one)).get(
            "stage1", sweep.dataset_key(one, False), sweep.setting_of(sweep.DEFAULTS["baseline"], CORES), 1, "plane")
        self.assertFalse(stats.short or stats.every_operation)
        self.assertEqual(stats.median, 1.0, "the cache hits are left out")


class WhatASlowdownIsBlamedOn(unittest.TestCase):
    """bed82ad: a warning that a choice gave up some mode names the mode furthest from its own best,
    among the modes that count -- not the one with the largest time."""

    def test_the_worst_mode_is_measured_against_its_own_best(self) -> None:
        medians = {"plane": 2.0, "spectrum": 30.0}
        best = {"plane": 1.0, "spectrum": 20.0}
        self.assertEqual(sweep.worst_mode(medians, best, {"plane": 1.0, "spectrum": 1.0}), "plane",
                         "twice its best is worse than one and a half times, whatever the seconds")

    def test_a_mode_of_no_weight_is_never_blamed(self) -> None:
        medians = {"plane": 2.0, "spectrum": 30.0}
        best = {"plane": 1.0, "spectrum": 20.0}
        self.assertEqual(sweep.worst_mode(medians, best, {"plane": 0.0, "spectrum": 1.0}), "spectrum")

    def test_the_score_is_the_weighted_geometric_mean_and_the_worst_slowdown(self) -> None:
        score, worst = sweep.score({"plane": 2.0, "spectrum": 4.0}, {"plane": 1.0, "spectrum": 1.0},
                                   {"plane": 1.0, "spectrum": 1.0})
        self.assertAlmostEqual(score, math.sqrt(8.0))
        self.assertEqual(worst, 4.0)

    def test_a_mode_that_never_finished_cannot_be_scored(self) -> None:
        self.assertEqual(sweep.score({"plane": math.inf}, {"plane": 1.0}, {"plane": 1.0}), (math.inf, math.inf))


class WhatIsTimed(unittest.TestCase):
    def stats(self, rows: list[dict[str, str]], on: sweep.Layout, mode: str) -> sweep.Stats:
        return sweep.Results(rows, datasets(on)).get(
            "stage1", sweep.dataset_key(on, False), sweep.setting_of(sweep.DEFAULTS["baseline"], CORES), 1, mode)

    def test_an_animation_is_timed_per_frame_after_the_first(self) -> None:
        """The first frame is a cold read whatever the layout, and a plane's to rank. A cube of five
        channels plays five frames, however many were asked for."""
        one = layout("one")
        stats = self.stats([row(one, "animation", 1.2, frame_first_s="0.4", logical_bytes="500",
                                animation_frames="32", position="pol=0;chan=0:5", late_frames="1",
                                late_max_s="0.05")], one, "animation")
        self.assertAlmostEqual(stats.median, 0.2, msg="0.8 seconds over the four frames after the first, not 1.2 over all five")
        self.assertEqual(stats.logical_bytes, 100)
        self.assertEqual((stats.paced_frames, stats.late_frames, stats.longest_stall), (4, 1, 0.05))

    def test_a_timeout_is_infinitely_slow(self) -> None:
        one = layout("one")
        rows = [row(one, "plane", 0.1, op_index="0"),
                row(one, "plane", 0.0, op_index="1", status="timeout"),
                row(one, "plane", 0.0, op_index="2", status="timeout")]
        stats = self.stats(rows, one, "plane")
        self.assertEqual(stats.timeouts, 2)
        self.assertTrue(math.isinf(stats.median), "two of three never finished, so the median did not either")

    def test_an_error_is_counted_and_not_timed(self) -> None:
        one = layout("one")
        stats = self.stats([row(one, "plane", 0.1), row(one, "plane", 0.0, status="error", op_index="1")], one, "plane")
        self.assertEqual((stats.errors, len(stats.ops)), (1, 1))


class ReadingAlike(unittest.TestCase):
    """The same position of the same shape must read the same bytes whichever layout holds it."""

    def test_two_layouts_reading_one_position_differently_are_a_mismatch(self) -> None:
        first, second = layout("first"), layout("second")
        rows = [row(first, "plane", 0.1, checksum="aa"), row(second, "plane", 0.1, checksum="bb")]
        self.assertEqual(len(sweep.Results(rows, datasets(first, second)).mismatches), 1)

    def test_reading_alike_is_no_mismatch(self) -> None:
        first, second = layout("first"), layout("second")
        rows = [row(first, "plane", 0.1, checksum="aa"), row(second, "plane", 0.1, checksum="aa")]
        self.assertEqual(sweep.Results(rows, datasets(first, second)).mismatches, [])

    def test_a_row_of_a_dataset_the_sweep_did_not_write_is_not_counted(self) -> None:
        mine, stranger = layout("mine"), layout("stranger")
        results = sweep.Results([row(stranger, "plane", 0.1)], datasets(mine))
        self.assertEqual(results.groups, {})


class Recommending(unittest.TestCase):
    MODES = ["plane"]

    def analysis(self, layouts: list[sweep.Layout], rows: list[dict[str, str]], **given: Any) -> sweep.Analysis:
        settings = config(measure={"modes": self.MODES}, **given)
        written = datasets(*layouts)
        return sweep.Analysis(settings, written, sweep.Results(rows, written), layouts, CORES)

    def test_stage2_takes_the_best_of_stage1_and_always_current(self) -> None:
        best, second, current = layout("best"), layout("second"), layout("current", current=True)
        rows = [row(best, "plane", 1.0), row(second, "plane", 2.0), row(current, "plane", 3.0)]
        analysis = self.analysis([best, second, current], rows, stage2={"top": 1})
        self.assertEqual([each.name for each in analysis.stage2_layouts()], ["best", "current"],
                         "the one best layout, and the source's own however it ranked")

    def test_a_read_budget_the_backend_cannot_be_given_is_never_recommended(self) -> None:
        """The backend has no setting for the read budget, so a choice that relies on one is measured
        and reported but not recommended, however much faster it is."""
        only = layout("only")
        budget = str(64 << 20)
        library = BASELINE
        larger = BASELINE[:3] + (budget,)
        rows = [row(only, "plane", 1.0, stage="stage2"), row(only, "plane", 0.25, stage="stage2", setting=larger)]
        analysis = self.analysis([only], rows, stage2={"layouts": ["only"], "read_budget": ["0", budget]})
        choices = analysis.stage2_choices()
        self.assertEqual(choices[0].setting.read_budget, 64 << 20, "the larger budget is the faster")
        _, after = analysis.recommendations()
        self.assertEqual(after.setting.read_budget, 0, "but only the library's own may be recommended")
        self.assertEqual(after.setting.key, library)

    def test_the_recommendation_for_the_data_as_it_is_is_on_its_own_layout(self) -> None:
        better, current = layout("better"), layout("current", current=True)
        rows = [row(better, "plane", 1.0, stage="stage2"), row(current, "plane", 2.0, stage="stage2")]
        analysis = self.analysis([better, current], rows, stage2={"layouts": ["better"]})
        now, after = analysis.recommendations()
        self.assertEqual((now.layout.name, after.layout.name), ("current", "better"))


if __name__ == "__main__":
    unittest.main()
