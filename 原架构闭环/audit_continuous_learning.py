"""Independent, read-only continuity / sensory evidence / recall assessment.

Never modify a source report or a life checkpoint. Baseline memory contents are
explicitly named saved references, not retrospectively invented raw sensations.
``initial_catalog(session)`` can initialize future experiment-side catalogs.
Nothing returned by that function is a legitimate input to a brain.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def initial_catalog(session):
    """Return evaluator-only references for an existing complete life archive.

    Colors come only from the old UI's actual ``visible_rays`` record. Visual,
    auditory and motor cells come from that starting checkpoint's original
    time->feature tables at its current threshold. They establish what was
    present in the saved memory, NOT the stronger assertion that no information
    had already been lost before that archive was saved.

    ``frequency`` remains None: an old external-presentation annotation is not
    proof that PCM was still sounding at this frame. ``presentation_frequency``
    preserves that weaker annotation for explicit experiment-side diagnostics.
    Do not silently count it as a newly witnessed audiovisual co-exposure.
    """
    catalog = {}
    for stamp, row in session.catalog.items():
        timestamp = int(stamp)
        visible = row.get("visible_rays", {})
        colors = sorted(color for color, rays in visible.items() if len(rays) > 0)
        recalled = session.brain.runtime.recall_at_time(timestamp, session.brain.config.memory_threshold)
        presentation = row.get("external_presentation") or {}
        catalog[str(timestamp)] = dict(
            colors=colors, frequency=None,
            visual=np.flatnonzero(recalled["visual"]).tolist(),
            audio=np.flatnonzero(recalled["audio"]).tolist(),
            motor=np.flatnonzero(recalled["motor"]).tolist(),
            physical_time=row.get("physical_time"),
            reference_source="baseline_saved_reference",
            color_source="baseline_UI_actual_visible_rays",
            presentation_frequency=presentation.get("frequency"),
            frequency_verified_from_actual_PCM=False,
        )
    return catalog


def tree_digest(value):
    """Hash complete state, including transient arrays, flags and RNG state."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, np.ndarray):
            digest.update(b"array" + str(item.dtype).encode() + str(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: str(k)):
                add(key); add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + str(len(item)).encode())
            for part in item:
                add(part)
        elif isinstance(item, np.generic):
            add(item.item())
        else:
            digest.update(type(item).__name__.encode() + repr(item).encode())
    add(value)
    return digest.hexdigest()


def _load_runner():
    # Imported original modules sometimes add other viewer directories to the
    # search path. The runner itself restores this directory before its viewer.
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("audited_continuous_runner", HERE / "continuous_learning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corrected_scores(report, catalog):
    stages = []
    for check in report["checks"]:
        totals = {phase: Counter() for phase in ("initial", "final")}
        first = {phase: Counter() for phase in ("initial", "final")}
        corrected = []
        for cue in check["cues"]:
            if not cue["previously_experienced"]:
                continue
            for frame in cue["trace"]:
                for phase in ("initial", "final"):
                    old = frame["score"][phase]
                    stored = catalog.get(str(old["time"]))
                    actual = set(map(int, old["cells"]))
                    expected = set() if stored is None else set(stored["visual"])
                    exact = bool(actual and expected and actual == expected)
                    correct = bool(exact and cue["color"] in stored["colors"])
                    category = ("correct" if correct else "no_recall" if old["time"] is None else
                                "unmapped" if stored is None else "content_mismatch" if not exact else "wrong_color")
                    totals[phase][category] += 1
                    if frame["frame"] == 0:
                        first[phase][category] += 1
                    if correct != old["correct"]:
                        corrected.append(dict(frequency=cue["frequency"], color=cue["color"],
                            frame=frame["frame"], phase=phase, time=old["time"], old_correct=old["correct"],
                            independently_correct=correct, cells=len(actual),
                            reference_source=None if stored is None else stored.get("reference_source", "actual_continuous_record")))
        stages.append(dict(stage=check["stage"], known_cue_count=check["known_cue_count"],
            reported_first_initial=check["known_cue_initial_correct"],
            reported_first_final=check["known_cue_first_final_correct"],
            corrected_first={key: dict(value) for key, value in first.items()},
            all_probe_frames={key: dict(value) for key, value in totals.items()},
            changed_scores=corrected))
    return stages


def audit(path, *, output=None, check_probe_immutability=True):
    report_bytes = Path(path).read_bytes()
    report = json.loads(report_bytes)
    runner = _load_runner()
    base_path = Path(report["base_life"])
    assert _sha(base_path) == report["base_sha256"], "base life changed since this run"
    session = runner.LifeSession.load(base_path)
    before = tree_digest(session.snapshot())
    old = initial_catalog(session)
    assert tree_digest(session.snapshot()) == before, "initial_catalog mutated the saved life"
    combined = {**old, **report.get("catalog", {})}
    original_clock = report["starts"]["clock"]
    rows = report["records"]
    frames = report.get("current_steps", len(rows))
    catalog = report.get("catalog", {})
    stamps = sorted(map(int, catalog))
    assert stamps == list(range(original_clock, original_clock + 2 * (frames + 1), 2))
    assert all(r["clock"] == original_clock + 2 * (r["step"] + 1) for r in rows)
    physical_start = float(session.environment.world.time)
    assert all(abs(r["time"] - (physical_start + r["step"] * .1)) < 1e-7 for r in rows)
    for stamp, entry in catalog.items():
        assert abs(entry["physical_time"] - (physical_start + (int(stamp) - original_clock) * .05)) < 1e-7
    checks = corrected_scores(report, combined)
    result = dict(status="complete", source_report=str(path),
        source_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        audited_runner_sha256=report["source_sha256"]["continuous_learning.py"],
        base_life_sha256=report["base_sha256"], source_report_status=report["status"],
        continuous_steps=frames, initial_clock=original_clock, final_clock=rows[-1]["clock"],
        chronological_catalog_and_physical_time_exact=True,
        starting_saved_references=len(old), continuous_actual_catalog_entries=len(catalog),
        initial_catalog_read_only=True, corrected_checks=checks,
        reference_limit="Old nonempty cells are baseline saved memory references, not reconstructed raw sensations.",
        findings=[
            {"severity": "scoring_bug", "lines": [177, 91],
             "detail": "Empty starting catalog labels genuine old-time recall incorrect; independent report preserves and corrects those scores."},
            {"severity": "scope", "lines": [177, 221],
             "detail": "Original direct anchors begin after continuation start, so direct_anchor scores do not by themselves test earlier life memories."},
            {"severity": "scope", "lines": [243],
             "detail": "LifeSession.save preserves its UI catalog, while new strict references live separately in this report; resume audits need both."},
            {"severity": "boundary", "lines": [244],
             "detail": "steps+1 clock assertion assumes first present changes prepared packet; true in this run, not for every possible base life."},
        ])

    if check_probe_immutability:
        # A small frozen copy-only probe; no learning and no writes to archives.
        # Three direct old anchors cover actual stored colors. Cue diagnostics
        # include all six tones but still run only two physical frames per clone.
        anchors = {}
        for stamp, reference in old.items():
            for color, frequency in runner.PAIRS[:3]:
                if color in reference["colors"] and reference["visual"]:
                    anchors.setdefault(str(frequency), int(stamp))
        immutable_start = tree_digest(session.snapshot())
        runner.assess(session, old, anchors, "independent_readonly_probe", probe_steps=2)
        immutable_end = tree_digest(session.snapshot())
        assert immutable_start == immutable_end, "assess mutated main brain/world/session"
        result["assess_complete_main_life_state_unchanged"] = dict(
            before_sha256=immutable_start, after_sha256=immutable_end,
            frozen_clone_physics_frames=12, no_training=True,
            direct_anchors=len(anchors), includes_three_endogenous_clone_microsteps=True)

    corpus_name = report.get("actual_sensor_corpus")
    if corpus_name:
        with np.load(corpus_name, allow_pickle=False) as corpus:
            n = len(corpus["time"])
            assert n == frames
            assert corpus["rgb"].shape == (n, 31, 3)
            assert corpus["pcm"].shape == (n, 800)
            assert corpus["body"].shape == (n, 8)
            assert corpus["muscles"].shape == (n, 4)
            assert np.array_equal(corpus["index_time"], original_clock + 2 * np.arange(n))
            assert np.max(np.abs(corpus["time"] - (physical_start + .1 * np.arange(n)))) < 1e-7
            coexposures = Counter()
            failures = []
            color_rgb = {"red": (1., .04, .04), "green": (.04, 1., .04), "blue": (.04, .04, 1.)}
            fft_frequency = np.fft.rfftfreq(800, 1 / 8000)
            for i in range(n):
                entry = catalog[str(int(corpus["index_time"][i]))]
                visible = sorted(color for color, rgb in color_rgb.items()
                    if np.any(np.all(np.isclose(corpus["rgb"][i], rgb, rtol=0, atol=1e-7), axis=1)))
                rms = float(np.sqrt(np.mean(corpus["pcm"][i] ** 2)))
                audible = rms > .02
                if visible != sorted(entry["colors"]) or audible != (entry["frequency"] is not None):
                    failures.append(i)
                if audible:
                    peak = float(fft_frequency[np.abs(np.fft.rfft(corpus["pcm"][i])).argmax()])
                    if abs(peak - entry["frequency"]) > 1e-8:
                        failures.append(i)
                    if visible:
                        coexposures[f"{peak:g}:{','.join(visible)}"] += 1
            assert not failures, failures
            result["actual_raw_corpus_alignment"] = dict(
                corpus_sha256=_sha(corpus_name), frames=n, failures=failures,
                rgb_color_rays_and_PCM_spectrum_agree_with_catalog=True,
                coexposure_frames=dict(coexposures),
                records_include_true_final_outcome_but_corpus_contains_preaction_frames_only=True)
    if output is None:
        output = Path(path).with_name(Path(path).stem + "_independent_audit.json")
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-probe-immutability", action="store_true")
    args = parser.parse_args()
    result, output = audit(args.report, output=args.output,
                           check_probe_immutability=not args.skip_probe_immutability)
    print(json.dumps({"output": str(output), "status": result["status"],
                      "source_report_sha256": result["source_report_sha256"],
                      "corrected_first": [{"stage": c["stage"], **c["corrected_first"]}
                                          for c in result["corrected_checks"]]}, ensure_ascii=False))
