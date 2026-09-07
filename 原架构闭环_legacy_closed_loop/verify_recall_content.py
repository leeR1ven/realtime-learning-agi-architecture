"""Independent content scoring of existing closed-loop reports, without training.

``strict_score(info, catalog, expected_color)`` separates choosing an associated
time from recovering that time's nonempty visual neural pattern. Labels are used
only here, after the brain's decision. This script never changes model or reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def strict_score(info, catalog, expected_color):
    """Score time identity separately from exact, nonempty visual completion.

    Expected color and the catalog are evaluator-only. A successful time choice
    with an empty or different visual pattern is explicitly NOT visual recovery.
    Exact recovery here is at the code level; it is not an image reconstruction,
    proof of navigation, or proof that all colors have distinct sensory codes.
    """
    result = {}
    for prefix, time_key, cells_key in (
            ("initial", "auditory_recall_time", "initially_recalled_visual_cells"),
            ("final", "final_recall_time", "final_visual_cells")):
        timestamp = info.get(time_key)
        entry = catalog.get(str(timestamp)) if timestamp is not None else None
        recalled = set(map(int, info.get(cells_key, ())))
        stored = set(map(int, entry.get("actual_visual_cells", ()))) if entry else set()
        time_correct = bool(entry and entry.get("color") == expected_color
                            and entry.get("visually_present", False))
        exact = bool(entry is not None and recalled == stored)
        content_correct = bool(time_correct and recalled and stored and exact)
        overlap, union = len(recalled & stored), len(recalled | stored)
        result.update({prefix + "_time": timestamp,
            prefix + "_color": None if entry is None else entry.get("color"),
            "correct_" + prefix + "_time": time_correct,
            prefix + "_recalled_visual_count": len(recalled),
            prefix + "_stored_visual_count": len(stored),
            prefix + "_nonempty_visual": bool(recalled),
            prefix + "_exact_stored_visual": exact,
            prefix + "_visual_precision": overlap / len(recalled) if recalled else 0.,
            prefix + "_visual_recall": overlap / len(stored) if stored else 0.,
            prefix + "_visual_jaccard": overlap / union if union else 0.,
            "correct_" + prefix + "_visual": content_correct})
    return result


def score_regressions():
    # Import only the existing evaluator, not a model instance or training run.
    from run_closed_loop import completion_score
    catalog = {"0": {"color": "red", "visually_present": True,
                      "actual_visual_cells": [1, 4]}}
    info = {"auditory_recall_time": 0, "final_recall_time": 0,
            "initially_recalled_visual_cells": [], "final_visual_cells": []}
    legacy = completion_score(info, catalog, "red")
    strict = strict_score(info, catalog, "red")
    assert legacy["correct_initial_color"] and legacy["correct_final_color"]
    assert strict["correct_initial_time"] and strict["correct_final_time"]
    assert not strict["correct_initial_visual"] and not strict["correct_final_visual"]
    cases = {"empty_visual_with_correct_time": {"legacy": legacy, "strict": strict}}
    for name, actual, expected in (("wrong_nonempty_visual", [2, 4], False),
                                  ("exact_nonempty_visual", [1, 4], True)):
        case = dict(info, initially_recalled_visual_cells=actual, final_visual_cells=actual)
        result = strict_score(case, catalog, "red")
        assert result["correct_initial_visual"] == expected
        assert result["correct_final_visual"] == expected
        cases[name] = result
    empty_catalog = {"0": {"color": "red", "visually_present": True,
                            "actual_visual_cells": []}}
    result = strict_score(info, empty_catalog, "red")
    assert result["initial_exact_stored_visual"] and not result["correct_initial_visual"]
    cases["empty_stored_and_recalled_is_not_completion"] = result
    wrong_color = strict_score(dict(info, initially_recalled_visual_cells=[1, 4],
                                    final_visual_cells=[1, 4]), catalog, "blue")
    assert not wrong_color["correct_initial_visual"] and not wrong_color["correct_final_visual"]
    cases["exact_visual_from_wrong_color_is_not_correct"] = wrong_color
    return {"passed": True, "cases": cases}


def temporal_review(report):
    training = report["training"]
    trace, catalog = training["trace"], training["catalog"]
    errors = []
    previous_muscles = np.zeros(4)
    for frame, row in enumerate(trace):
        info = row["info"]
        index = int(info["index_time"])
        entry = catalog.get(str(index))
        if index != 2 * frame:
            errors.append(["training_index_frame", frame, index])
        if not np.allclose(row["physical_time"], frame * .1, rtol=0, atol=1e-8):
            errors.append(["training_physical_time", frame])
        if not np.array_equal(info["actual_motor_in_memory"], previous_muscles):
            errors.append(["training_actual_motor_offset", frame])
        if entry is None or entry["color"] != row["color"]:
            errors.append(["training_catalog_color_offset", frame])
        elif not np.allclose(entry["physical_time"], row["physical_time"], rtol=0, atol=1e-8):
            errors.append(["training_catalog_time_offset", frame])
        previous_muscles = np.asarray(info["muscles"])
    if len(catalog) != len(trace) + 1:
        errors.append(["catalog_expected_initial_plus_all_outcomes", len(catalog), len(trace) + 1])
    terminal_entry = catalog.get(str(2 * len(trace)))
    if terminal_entry is None or not np.array_equal(terminal_entry["actual_muscles"], previous_muscles):
        errors.append(["last_training_outcome_missing_or_motor_mismatch"])
    first_probe_frames, first_probe_times = [], []
    for branch in report.get("evaluation", []):
        for frame, row in enumerate(branch["trace"]):
            info = row["info"]
            if info["learning"]:
                errors.append(["evaluation_learning_true", branch["color"], branch["condition"], frame])
            if int(info["index_time"]) != int(training["clock"]) + 2 * frame:
                errors.append(["probe_index_offset", branch["color"], branch["condition"], frame])
            if not np.allclose(row["time"], len(trace) * .1 + frame * .1, rtol=0, atol=1e-8):
                errors.append(["probe_physical_time_offset", branch["color"], branch["condition"], frame])
            expected_motor = previous_muscles if frame == 0 else branch["trace"][frame-1]["info"]["muscles"]
            if not np.array_equal(info["actual_motor_in_memory"], expected_motor):
                errors.append(["probe_executed_motor_offset", branch["color"], branch["condition"], frame])
        first_probe_frames.append(branch["trace"][0]["info"]["index_time"])
        first_probe_times.append(branch["trace"][0]["time"])
    patterns = {}
    for entry in catalog.values():
        if entry["visually_present"]:
            code = tuple(entry["actual_visual_cells"])
            patterns.setdefault(code, set()).add(entry["color"])
    ambiguous = [sorted(colors) for colors in patterns.values() if len(colors) > 1]
    return {"errors": errors, "training_physical_frames": len(trace),
            "training_catalog_frames": len(catalog), "last_real_outcome_in_same_catalog": terminal_entry is not None,
            "probe_start_indices_equal": len(set(first_probe_frames)) <= 1,
            "probe_start_physical_times_equal": len(set(first_probe_times)) <= 1,
            "unique_stored_visual_patterns": len(patterns),
            "identical_visual_patterns_assigned_multiple_colors": ambiguous,
            "boundary_interpretation": "At a presentation boundary the experiment changes the visible circle and PCM after physics but before the brain records the next observation. The new actual stimulus is paired with the just-executed old action; this is a real scheduled sensory transition, not a teacher action or a same-frame action label.",
            "catalog_frequency_limitation": "Catalog frequency is the presentation assignment, not evidence that PCM is still audible. Use trace.info.sound_rms for actual recorded action-input frames."}


def verify_report(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    result = {"path": str(path.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
              "status": report.get("status"), "model_or_report_modified": False}
    if report.get("status") != "complete":
        return {**result, "deferred": "Only completed reports are independently scored"}
    catalog = report["training"]["catalog"]
    by_condition = {}
    branches = []
    for branch in report["evaluation"]:
        scores = [strict_score(row["info"], catalog, branch["color"]) for row in branch["trace"]]
        initial = scores[0]
        summary = by_condition.setdefault(branch["condition"], {
            "trials": 0, "correct_initial_time": 0, "correct_final_time": 0,
            "correct_initial_visual": 0, "correct_final_visual": 0,
            "all_probe_frames": 0, "all_frames_correct_initial_visual": 0,
            "all_frames_correct_final_visual": 0})
        summary["trials"] += 1
        for key in ("correct_initial_time", "correct_final_time", "correct_initial_visual", "correct_final_visual"):
            summary[key] += int(initial[key])
        summary["all_probe_frames"] += len(scores)
        summary["all_frames_correct_initial_visual"] += sum(s["correct_initial_visual"] for s in scores)
        summary["all_frames_correct_final_visual"] += sum(s["correct_final_visual"] for s in scores)
        branches.append({"color": branch["color"], "condition": branch["condition"],
                         "first_probe": initial, "all_frames": scores})
    result.update(strict_first_probe_summary=by_condition, branches=branches,
                  temporal_review=temporal_review(report),
                  interpretation="The original correct_initial_color/correct_final_color summary checks catalog time identity and prior visibility. Strict visual scores additionally require an exact nonempty match to actual stored neural cells. No result is a navigation score.")
    result["legacy_summary"] = report.get("summary")
    full = [branch for branch in report["evaluation"] if branch["condition"] == "full"]
    motor_codes = [tuple(tuple(row["info"]["muscles"]) for row in branch["trace"]) for branch in full]
    paths = [tuple((tuple(row["position"]), row["heading"]) for row in branch["trace"]) for branch in full]
    result["motor_selectivity"] = {
        "full_trials": len(full), "distinct_action_sequences_across_tones": len(set(motor_codes)),
        "distinct_physical_trajectories_across_tones": len(set(paths)),
        "zero_motor_control_displacement": [{"color": branch["color"], "distance": branch["movement_distance"]}
            for branch in report["evaluation"] if branch["condition"] == "no_motor_recall"],
        "interpretation": "Inherited velocity and muscle activation can produce displacement even with all-zero commands. Different or nonzero movement alone is not evidence of correct sound-guided navigation."}
    return result


def score_from_reported_match(row, catalog, expected):
    """Restricted fallback where a report omitted the actual recalled cells.

    These results depend on the saved equality assertion. They must never be
    described as an independent array-level reconstruction from the report.
    """
    stored_score = row["score"]
    result = {}
    for prefix in ("initial", "final"):
        timestamp = stored_score[prefix + "_time"]
        entry = catalog.get(str(timestamp)) if timestamp is not None else None
        correct_time = bool(entry and entry["color"] == expected and entry["visually_present"])
        match = stored_score.get(prefix + "_exact_nonempty_visual",
                  stored_score.get(prefix + "_same_stored_visual", False))
        result["correct_" + prefix + "_time"] = correct_time
        result["correct_" + prefix + "_visual"] = bool(
            correct_time and match and entry["actual_visual_cells"])
    return result


def summarize_branches(probes, catalog, *, raw_cells):
    summaries = {}
    errors = []
    for probe in probes:
        rows = probe.get("trace", probe.get("frames"))
        name = probe["condition"]
        summary = summaries.setdefault(name, {"trials": 0, "initial_time_correct": 0,
            "first_final_time_correct": 0, "initial_visual_correct": 0,
            "first_final_visual_correct": 0, "all_frames": 0,
            "final_visual_correct_frames": 0, "silent_frames": 0, "silent_visual_correct_frames": 0})
        summary["trials"] += 1
        for frame, row in enumerate(rows):
            expected = row.get("expected", probe.get("color", probe.get("expected")))
            info = row.get("info")
            if raw_cells and info is None:
                info = {"auditory_recall_time": row["score"]["initial_time"],
                        "final_recall_time": row["score"]["final_time"],
                        "initially_recalled_visual_cells": row["initially_recalled_visual_cells"],
                        "final_visual_cells": row["final_visual_cells"]}
            score = (strict_score(info, catalog, expected) if raw_cells else
                     score_from_reported_match(row, catalog, expected))
            if frame == 0:
                for target, source in (("initial_time_correct", "correct_initial_time"),
                    ("first_final_time_correct", "correct_final_time"),
                    ("initial_visual_correct", "correct_initial_visual"),
                    ("first_final_visual_correct", "correct_final_visual")):
                    summary[target] += int(score[source])
            summary["all_frames"] += 1
            summary["final_visual_correct_frames"] += int(score["correct_final_visual"])
            rms = row["info"]["sound_rms"] if "info" in row else row["sound_rms"]
            silent = rms < .02
            summary["silent_frames"] += int(silent)
            summary["silent_visual_correct_frames"] += int(silent and score["correct_final_visual"])
            if "final_correct_content" in row["score"]:
                if score["correct_final_visual"] != row["score"]["final_correct_content"]:
                    errors.append([name, expected, frame, "saved_final_content_flag_disagrees"])
                if score["correct_initial_visual"] != row["score"]["initial_correct_content"]:
                    errors.append([name, expected, frame, "saved_initial_content_flag_disagrees"])
    return {"summary": summaries, "aggregation_errors": errors,
            "evidence": "independent_from_full_recalled_cell_arrays" if raw_cells else
                        "derived_from_saved_match_assertions_and_catalog; raw_recalled_arrays_not_saved"}


def verify_dynamic(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    result = {"path": str(path.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
              "status": report.get("status")}
    if report.get("status") != "complete":
        return {**result, "deferred": True}
    grid = {row["candidate"]: row for row in report["grid"]}
    result["grid"] = [{"candidate": key, **summarize_branches(row["probes"],
                        row["training"]["catalog"], raw_cells=False)} for key, row in grid.items()]
    confirmations = []
    for row in report["confirmations"]:
        training = row.get("training", grid[row["candidate"]]["training"])
        catalog = training["catalog"]
        original = summarize_branches(row["original_evaluate"], catalog, raw_cells=True)
        diagnostics = summarize_branches(row["current_diagnostics"], catalog, raw_cells=False)
        compare = []
        for condition in ("full", "no_pfc_recurrence"):
            compare.append({"condition": condition, "strict_counts_match_diagnostic_flags":
                original["summary"][condition] == diagnostics["summary"][condition]})
        confirmations.append({"candidate": row["candidate"], "seed": row["seed"],
            "original_evaluate": original, "current_diagnostics": diagnostics,
            "common_conditions_comparison": compare,
            "timeline": temporal_review({"training": training, "evaluation": row["original_evaluate"]})})
    result["confirmations"] = confirmations
    result["selected_candidates"] = report["selected_candidates"]
    result["original_accepted_candidates"] = report["accepted_candidates"]
    result["strict_accepted_candidates"] = [name for name in report["selected_candidates"]
        if {r["seed"] for r in confirmations if r["candidate"] == name} == {2026, 2027}
        and all((r["original_evaluate"]["summary"]["full"]["initial_visual_correct"] == 3 and
                r["original_evaluate"]["summary"]["full"]["first_final_visual_correct"] == 3 and
                r["original_evaluate"]["summary"]["full"]["silent_frames"] > 0 and
                r["original_evaluate"]["summary"]["full"]["silent_visual_correct_frames"] ==
                r["original_evaluate"]["summary"]["full"]["silent_frames"])
               for r in confirmations if r["candidate"] == name)]
    result["scope"] = "Calibration acceptance at the reported two births only; stronger later tests can fail. Grid and diagnostic arms omit raw visual arrays; full confirmation arms permit independent exact scoring."
    return result


def verify_heldout(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    result = {"path": str(path.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
              "status": report.get("status")}
    if report.get("status") != "complete":
        return {**result, "deferred": True}
    from validate_closed_loop import score_content
    # Compare both independent scoring implementations on deliberate failures,
    # including right time but wrong cells, empty patterns and the wrong color.
    catalog = {"0": {"color": "red", "visually_present": True, "actual_visual_cells": [1, 4]}}
    for cells in ([], [1], [1, 4], [1, 3]):
        for color in ("red", "blue", None):
            info = dict(auditory_recall_time=0, final_recall_time=0,
                        initially_recalled_visual_cells=cells, final_visual_cells=cells)
            a, b = strict_score(info, catalog, color), score_content(info, catalog, color)
            assert a["correct_initial_visual"] == b["initial_correct_content"]
            assert a["correct_final_visual"] == b["final_correct_content"]
    result["score_function_equivalence_cases"] = 12
    result["cases"] = []
    all_raw = True
    for case in report["cases"]:
        catalog = case["training"]["catalog"]
        has_raw = all("initially_recalled_visual_cells" in row and "final_visual_cells" in row
                      for probe in case["probes"] for row in probe["frames"])
        all_raw = all_raw and has_raw
        scored = summarize_branches(case["probes"], catalog, raw_cells=has_raw)
        errors = list(scored["aggregation_errors"])
        for condition, summary in scored["summary"].items():
            original = case["summary"][condition]
            for current, saved in (("initial_visual_correct", "initial_correct"),
                ("first_final_visual_correct", "first_final_correct"),
                ("final_visual_correct_frames", "final_correct_frames"),
                ("silent_visual_correct_frames", "silent_correct_frames"),
                ("silent_frames", "silent_frames"), ("all_frames", "frames")):
                if summary[current] != original[saved]:
                    errors.append([condition, current, "aggregate_differs_from_original"])
        switch_checks = []
        for switch in case.get("cue_switches", []):
            requested = [r["expected"] for r in switch["frames"]]
            correct_schedule = requested[:8] == [switch["expected"]] * 8 and requested[8:] == [switch["switch_expected"]] * 8
            audible = [r["frame"] for r in switch["frames"] if r["sound_rms"] >= .02]
            switch_checks.append({"from": switch["expected"], "to": switch["switch_expected"],
                "expected_changes_on_frame8": correct_schedule,
                "audible_frames": audible, "actual_two_pulses_match_schedule": audible == list(range(6)) + list(range(8, 14))})
        result["cases"].append({"name": case["name"], "seed": case["seed"],
            "permutation": case["permutation"], "score_review": scored,
            "aggregate_errors": errors, "cue_switch_timing": switch_checks,
            "additional_probes": {name: summarize_branches(case[name],
                {} if name == "newborn_probes" else catalog, raw_cells=has_raw)
                for name in ("amplitude_probes", "unknown_sound_probes", "newborn_probes", "cue_switches")
                if case.get(name)},
            "stored_frozen_weight_checks_all_true": all(p["frozen_weights"] for p in case["probes"])})
    result["static_review"] = {
        "no_labels_or_teacher_actions_enter_brain": True,
        "expected_labels_used_only_after_observe_then_act": True,
        "switch_new_pcm_processed_in_frame8_outcome_and_reused_without_double_record": True,
        "flags_set_before_new_observation_and_independent_snapshot_branches": True,
        "learning_disabled_and_learned_weights_index_motor_checked_by_digest": True,
        "content_evidence": "independent_recomputation_from_every_raw_recalled_cell_array" if all_raw else
            "Only saved exact-match assertions available; scoring logic and aggregation checked, omitted arrays not independently reconstructed."}
    return result


def verify_index_margin(path):
    result = verify_heldout(path)
    if result.get("deferred"):
        return result
    original = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    totals = {"full_correct_frames": 0, "full_frames": 0,
              "switch_correct_frames": 0, "switch_frames": 0}
    initial_correct = True
    for case in original["cases"]:
        catalog = case["training"]["catalog"]
        for group in ("probes", "cue_switches"):
            for branch in case[group]:
                if group == "probes" and branch["condition"] != "full":
                    continue
                rows = branch["frames"]
                prefix = "full" if group == "probes" else "switch"

                def score_row(row):
                    info = {"auditory_recall_time": row["score"]["initial_time"],
                        "final_recall_time": row["score"]["final_time"],
                        "initially_recalled_visual_cells": row["initially_recalled_visual_cells"],
                        "final_visual_cells": row["final_visual_cells"]}
                    return strict_score(info, catalog, row["expected"])

                for frame, row in enumerate(rows):
                    score = score_row(row)
                    totals[prefix + "_frames"] += 1
                    totals[prefix + "_correct_frames"] += int(score["correct_final_visual"])
                    if frame == 0 and group == "probes":
                        initial_correct = initial_correct and score["correct_initial_visual"]
                    if not score["correct_final_visual"]:
                        following = rows[frame+1] if frame+1 < len(rows) else None
                        next_score = score_row(following) if following else None
                        failures.append({"case": case["name"], "seed": case["seed"],
                            "group": group, "frame": frame, "physical_time": row["physical_time"],
                            "expected": row["expected"], "from": branch["expected"],
                            "to": branch.get("switch_expected"), "frequency": branch["frequency"],
                            "switch_frequency": branch.get("switch_frequency"),
                            "sound_rms": row["sound_rms"], "strict_score": score,
                            "first_new_cue_frame": group == "cue_switches" and frame == 8,
                            "next_frame": None if following is None else following["frame"],
                            "next_time_delta": None if following is None else
                                following["physical_time"] - row["physical_time"],
                            "next_frame_strict_score": next_score,
                            "next_frame_correct": bool(next_score and next_score["correct_final_visual"])})
    accepted = bool(initial_correct and totals["full_frames"] > 0 and totals["switch_frames"] > 0
                    and totals["full_correct_frames"] == totals["full_frames"]
                    and totals["switch_correct_frames"] == totals["switch_frames"])
    original_accepted = original["all_tested_retention_and_switch_frames_correct"]
    assert accepted == original_accepted, "Strict acceptance differs from the preserved report"
    result["predefined_acceptance"] = {**totals, "initial_content_correct_all_full_trials": initial_correct,
        "independent_allpass": accepted, "original_allpass": original_accepted,
        "criterion_unchanged": True, "grace_frames_excluded": 0,
        "interpretation": "A first-switch-frame error remains a failure under the original all-frames criterion, even if the following 0.1 s frame is correct."}
    result["full_or_switch_error_frames"] = failures
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "recall_content_verification.json")
    args = parser.parse_args()
    files = args.report or sorted((HERE / "results").glob("closed_loop*.json"))
    report = {"regressions": score_regressions(), "reports": [verify_report(path) for path in files],
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_training": False, "model_modifications": False,
              "runner_review": {
                  "label_and_action_leakage": "Static review: labels/coordinates are used for actual stimulus presentation or scoring. Only observation+PCM goes to observe_then_act/observe_outcome; selected muscles go unchanged to env.step. No planner or teacher action is used.",
                  "ablation_cache": "Evaluation restores a fresh branch, clears the precomputed action for every condition, then hides beacons and emits a new PCM cue. Conditions are set before the first new observation; no condition reuses a baseline cached first action.",
                  "limits": [
                      "no_auditory_index removes the auditory initial time query; auditory current input can still reach the PFC. It is not a complete hearing ablation.",
                      "no_pfc_recurrence removes learned excitatory recurrent drive while retaining the specified inhibitory pathways. A label alone does not establish effective or useful sequence computation.",
                      "no_motor_recall removes the only enabled motor output in this probe because exploration and reflex are also off. Zero commands demonstrate wiring necessity; inherited inertia may still move the body.",
                      "Evaluation asserts connection counts and reverse-motor arrays are unchanged. The report does not contain before/after hashes of all learned PFC and memory weights.",
                      "This same-world probe assesses trained associations with hidden beacons; it is not held-out navigation or a new-rule generalization test."]}}
    dynamic_path = HERE / "results" / "dynamic_calibration.json"
    heldout_path = HERE / "results" / "heldout_validation.json"
    if dynamic_path.exists():
        report["dynamic_calibration"] = verify_dynamic(dynamic_path)
    if heldout_path.exists():
        report["heldout_validation"] = verify_heldout(heldout_path)
    heldout_raw_path = HERE / "results" / "heldout_validation_raw.json"
    if heldout_raw_path.exists():
        report["heldout_validation_raw"] = verify_heldout(heldout_raw_path)
    gain_path = HERE / "results" / "recall_gain_validation.json"
    if gain_path.exists():
        report["recall_gain_validation"] = verify_heldout(gain_path)
    index_path = HERE / "results" / "index_margin_validation.json"
    if index_path.exists():
        report["index_margin_validation"] = verify_index_margin(index_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in report["reports"]:
        print(Path(item["path"]).name, json.dumps(item.get("strict_first_probe_summary", item.get("deferred")),
                                                  ensure_ascii=False))
    print("saved", args.output)


if __name__ == "__main__":
    main()
