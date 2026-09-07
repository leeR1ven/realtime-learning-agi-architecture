"""Read-only report/parameter audit; continuation occurs only in disposable copies.

Visual cue scores are same-event-content diagnostics. They do not establish
correct or incorrect action/route behavior, and no AGI conclusion is drawn.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_continuous_learning import (initial_catalog, tree_digest, corrected_scores,
                                      _load_runner, _sha)

HERE = Path(__file__).resolve().parent


def _table_parameters(brain):
    fields = ("强化量", "衰减率", "消失下限", "警戒线", "衰减间隔", "维护次数", "上次衰减维护")
    tables = {"pfc_excitatory": brain.runtime.pfc.联想, "pfc_disinhibitory": brain.runtime.pfc.去抑}
    for name, memory in brain.runtime.memories.items():
        tables[name + "_feature_to_time"] = memory.特征到时间
        tables[name + "_time_to_feature"] = memory.时间到特征
    return {name: {field: getattr(table, field) for field in fields} for name, table in tables.items()}


def _verify_parameters(brain, parameters):
    config = asdict(brain.config)
    assert all(config[name] == value for name, value in parameters.items())
    tables = _table_parameters(brain)
    for name, table in tables.items():
        if name.startswith("pfc_"):
            assert table["警戒线"] == parameters["pfc_decay_warning"]
            assert table["衰减间隔"] == parameters["pfc_decay_interval"]
            assert table["衰减率"] == parameters["pfc_decay_rate"]
            assert table["强化量"] == parameters["pfc_hebb_increment"]
        else:
            assert table["衰减率"] == parameters["memory_decay_rate"]
    return tables


def _retention(brain, references):
    counts = {name: {"correct": 0, "total": 0, "empty_references": 0, "minimum_expected_weight": 1e300}
              for name in ("visual", "audio", "motor")}
    failures = []
    for stamp, reference in references.items():
        recalled = brain.runtime.recall_at_time(int(stamp), brain.config.memory_threshold)
        for name, row in counts.items():
            expected = set(reference[name])
            actual = set(np.flatnonzero(recalled[name]))
            if not expected:
                row["empty_references"] += 1
                continue
            row["total"] += 1
            row["correct"] += int(actual == expected)
            row["minimum_expected_weight"] = min(row["minimum_expected_weight"],
                min(float(recalled[name + "_current"][cell]) for cell in expected))
            if actual != expected:
                failures.append({"time": int(stamp), "modality": name,
                                 "missing": sorted(expected - actual), "extra": sorted(actual - expected)})
    return counts, failures


def _sensor_alignment(report, baseline_session):
    start_clock = report["starts"]["clock"]
    start_time = float(baseline_session.environment.world.time)
    n = report["current_steps"]
    catalog = report["catalog"]
    new_stamps = sorted(int(stamp) for stamp in catalog if int(stamp) >= start_clock)
    assert new_stamps == list(range(start_clock, start_clock + 2 * (n + 1), 2))
    for row in report["records"]:
        assert row["clock"] == start_clock + 2 * (row["step"] + 1)
        assert abs(row["time"] - (start_time + .1 * row["step"])) < 1e-7
    colors = {"red": (1., .04, .04), "green": (.04, 1., .04), "blue": (.04, .04, 1.)}
    coexposures = {}
    encoding_samples = {}
    adapter = type(baseline_session.brain.adapter).from_snapshot(baseline_session.brain.adapter.snapshot())
    auditory_forward = copy.deepcopy(baseline_session.brain.forward)
    zero_visual = np.zeros(adapter.widths['visual'], bool)
    zero_motor = np.zeros(adapter.widths['motor'], bool)
    fit_cache = {}
    maximum_tone_residual = 0.
    with np.load(report["actual_sensor_corpus"], allow_pickle=False) as archive_arrays:
        # NpzFile indexing decompresses the entire member on every access.
        # Read each immutable raw array once before the per-frame audit.
        corpus = {name: archive_arrays[name] for name in archive_arrays.files}
        assert corpus["rgb"].shape == (n, 31, 3)
        assert corpus["distance"].shape == (n, 31)
        assert corpus["body"].shape == (n, 8)
        assert corpus["pcm"].shape == (n, 800)
        assert corpus["muscles"].shape == (n, 4)
        assert np.array_equal(corpus["index_time"], start_clock + 2 * np.arange(n))
        assert np.max(np.abs(corpus["time"] - start_time - .1 * np.arange(n))) < 1e-7
        frequencies = np.fft.rfftfreq(800, 1 / 8000)
        for i in range(n):
            entry = catalog[str(int(corpus["index_time"][i]))]
            visible = sorted(name for name, rgb in colors.items() if
                np.any(np.all(np.isclose(corpus["rgb"][i], rgb, atol=1e-7, rtol=0), axis=1)))
            assert visible == sorted(entry["colors"]), i
            rms = float(np.sqrt(np.mean(corpus["pcm"][i] ** 2)))
            assert (rms > .02) == (entry["frequency"] is not None), i
            assert abs(entry["physical_time"] - corpus["time"][i]) < 1e-7
            if rms > .02:
                peak = float(frequencies[np.abs(np.fft.rfft(corpus["pcm"][i])).argmax()])
                actual_frequency = float(entry["frequency"])
                assert abs(peak - actual_frequency) <= 5. + 1e-8, i
                # Non-bin-centred tones (e.g. 307 Hz) cannot be checked by
                # equality to a 10 Hz FFT bin. Fit its actual sampled sine and
                # cosine components; phase is free, frequency is not.
                if actual_frequency not in fit_cache:
                    phase = 2 * np.pi * actual_frequency * np.arange(800) / 8000
                    basis = np.c_[np.sin(phase), np.cos(phase)]
                    fit_cache[actual_frequency] = (basis, np.linalg.pinv(basis))
                basis, inverse = fit_cache[actual_frequency]
                coefficients = inverse @ corpus['pcm'][i]
                residual = float(np.sqrt(np.mean((basis @ coefficients - corpus['pcm'][i]) ** 2)))
                maximum_tone_residual = max(maximum_tone_residual, residual)
                assert residual < 2e-8 and abs(float(np.linalg.norm(coefficients)) - .45) < 2e-8, i
                if visible:
                    key = f"{actual_frequency:g}:{','.join(visible)}"
                    coexposures[key] = coexposures.get(key, 0) + 1
                samples = encoding_samples.setdefault(str(actual_frequency), [])
                if len(samples) < 8:
                    body = corpus['body'][i]
                    packet = dict(observation=dict(ray_distances=corpus['distance'][i],
                        ray_colors=corpus['rgb'][i], velocity=body[:2], angular_velocity=body[2],
                        pain=body[3], touch=body[4:]), waveform=corpus['pcm'][i])
                    previous_muscles = (baseline_session.brain.last_executed if i == 0 else corpus['muscles'][i-1])
                    encoded = adapter.encode(packet, previous_muscles)
                    audio = encoded['audio']
                    projected = auditory_forward.前向传播(zero_visual, audio, zero_motor)
                    diagnostic = encoded['diagnostics']['audio']
                    samples.append(dict(frame=i, audible_pcm_rms=rms,
                        audio_cells=np.flatnonzero(audio).tolist(),
                        auditory_forward_cells=np.flatnonzero(projected).tolist(),
                        raw_original_audio_count=diagnostic.get('original_output_active'),
                        positive_deviation_count=len(diagnostic.get('positive_deviation_ids', [])),
                        negative_deviation_count=len(diagnostic.get('negative_deviation_ids', []))))
        muscles = corpus['muscles']
        linear_speed = np.linalg.norm(corpus['body'][:, :2], axis=1)
        left = muscles[:, 0] - muscles[:, 1]
        right = muscles[:, 2] - muscles[:, 3]
        motion = dict(frames=n, source_counts=dict(Counter(row['source'] for row in report['records'])),
            nonzero_muscle_frames=int(np.sum(np.max(np.abs(muscles), axis=1) > 1e-8)),
            nonzero_net_force_command_frames=int(np.sum(np.abs(left) + np.abs(right) > 1e-8)),
            maximum_muscle_activation=float(muscles.max()), mean_muscle_activation=float(muscles.mean()),
            rms_muscle_activation=float(np.sqrt(np.mean(muscles ** 2))),
            linear_speed_above_001_frames=int(np.sum(linear_speed > .01)),
            maximum_linear_speed=float(linear_speed.max()), mean_linear_speed=float(linear_speed.mean()),
            angular_speed_above_001_frames=int(np.sum(np.abs(corpus['body'][:, 2]) > .01)),
            maximum_angular_speed=float(np.max(np.abs(corpus['body'][:, 2]))),
            touch_frames=int(np.sum(np.max(corpus['body'][:, 4:], axis=1) > 0)),
            pain_frames=int(np.sum(corpus['body'][:, 3] > 0)))
        # Last true outcome is not in the pre-action raw corpus. Its actual
        # fields are preserved in the full final life checkpoint, checked below.
    collisions = []
    frequencies_seen = sorted(encoding_samples, key=float)
    for i, first in enumerate(frequencies_seen):
        patterns_a = {tuple(row['audio_cells']) for row in encoding_samples[first]}
        for second in frequencies_seen[i+1:]:
            patterns_b = {tuple(row['audio_cells']) for row in encoding_samples[second]}
            common = patterns_a & patterns_b
            if common:
                collisions.append(dict(frequencies=[float(first), float(second)],
                    shared_audio_patterns=len(common),
                    all_sampled_patterns_identical=patterns_a == patterns_b and len(patterns_a) == 1,
                    shared_pattern_cell_counts=[len(pattern) for pattern in common]))
    return dict(frames=n, actual_color_and_PCM_spectrum_all_match=True,
                one_timeline=True, start_clock=start_clock,
                final_clock=report["checks"][-1]["clock"],
                new_observations=n + 1, coexposure_frames=coexposures,
                maximum_PCM_sinusoid_fit_rms=maximum_tone_residual,
                actual_motor_and_body_statistics=motion,
                audio_encoder_first_eight_audible_samples=encoding_samples,
                observed_audio_code_overlaps=collisions,
                encoder_scope='Observed ordinary adapter/forward codes from raw PCM; first eight audible samples per frequency, not a behavior or all-waveform equivalence test.',
                corpus_sha256=_sha(report["actual_sensor_corpus"]))


def audit_one(report_path, parameters_path, *, continuation_steps=10, previous_report=None):
    runner = _load_runner()
    from checkpoint import load_tree
    report_path, parameters_path = Path(report_path), Path(parameters_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert all(report['brain_config'][name] == value for name, value in parameters.items())
    assert not report['parameters'] or report['parameters'] == parameters
    assert _sha(report["base_life"]) == report["base_sha256"]
    archive = Path(report["complete_life_checkpoint"])
    input_hashes = {str(path): _sha(path) for path in
                    (report_path, parameters_path, Path(report["base_life"]), archive)}
    if report.get("complete_life_sha256"):
        assert _sha(archive) == report["complete_life_sha256"]
    base = runner.LifeSession.load(report["base_life"])
    references = initial_catalog(base)
    previous_path = previous_report or report.get('previous_report')
    previous_identity = None
    if previous_path:
        previous_path = Path(previous_path)
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        assert previous['status'] == 'complete'
        assert previous['checks'][-1]['clock'] == report['starts']['clock']
        assert previous['starts']['control_steps'] + previous['current_steps'] == report['starts']['control_steps']
        assert Path(previous['complete_life_checkpoint']).resolve() == Path(report['base_life']).resolve()
        if previous.get('complete_life_sha256'):
            assert previous['complete_life_sha256'] == report['base_sha256']
        # Prior raw references take precedence over reading the starting memory
        # again, so missing earlier contents cannot be silently removed.
        references.update(copy.deepcopy(previous['catalog']))
        input_hashes[str(previous_path)] = _sha(previous_path)
        previous_identity = dict(path=str(previous_path), sha256=_sha(previous_path),
                                 strict_reference_count=len(previous['catalog']))
    a = runner.LifeSession.load(archive)
    tables_before = _verify_parameters(a.brain, parameters)
    current_config = asdict(a.brain.config)
    assert all(current_config[name] == value for name, value in report['brain_config'].items())
    added_defaults = {name: value for name, value in current_config.items() if name not in report['brain_config']}
    allowed_inactive_defaults = {'pfc_weight_ceiling': None, 'pfc_weight_depression': 'multiplicative'}
    assert all(name in allowed_inactive_defaults and value == allowed_inactive_defaults[name]
               for name, value in added_defaults.items()), added_defaults
    assert a.brain.runtime.parameters.get('pfc_plasticity') is None
    assert type(a.brain.runtime.pfc.联想).__name__ == '连接表'
    assert int(a.brain.runtime.clock.当前时间) == report["checks"][-1]["clock"]
    assert a.brain.runtime.clock.时间激活.sum() == 1
    assert a.control_steps == report["starts"]["control_steps"] + report["current_steps"]
    assert a.brain.runtime.frames_recorded == report["starts"]["sensory_frames"] + report["current_steps"] + 1
    assert int(a.brain.runtime.clock.当前时间) == 2 * a.brain.runtime.frames_recorded
    for stamp, reference in references.items():
        stored = report["catalog"][stamp]
        assert all(stored[key] == reference[key] for key in ("visual", "audio", "motor", "colors"))
        assert stored.get("reference_source") == reference.get("reference_source")
    direct, failures = _retention(a.brain, references)
    assert not failures, failures
    reported = report["checks"][-1]["old_saved_content_retention"]
    assert all({k: v[k] for k in ("correct", "total")} == reported[name] for name, v in direct.items())
    raw = load_tree(archive)
    recovered = runner.OriginalBrain.from_snapshot(raw["brain"])
    assert tree_digest(recovered.snapshot()) == tree_digest(a.brain.snapshot())
    final_entry = report["catalog"][str(a.info["index_time"]) ]
    assert a.brain.prepared is not None and a.brain.pending is None
    for name in ("visual", "audio", "motor"):
        assert final_entry[name] == np.flatnonzero(a.brain.runtime.memories[name].激活).tolist()
    assert abs(final_entry["physical_time"] - a.environment.world.time) < 1e-7
    sensory = _sensor_alignment(report, base)
    motion = sensory['actual_motor_and_body_statistics']
    start_position, final_position = base.environment.world.position, a.environment.world.position
    motion.update(start_position=start_position.tolist(), final_position=final_position.tolist(),
        physical_displacement=float(np.linalg.norm(final_position-start_position)),
        integrated_physical_path_length=float(a.environment._path_length-base.environment._path_length),
        physical_contact_steps=int(a.environment._contact_steps-base.environment._contact_steps),
        contact_normal_impulse=float(a.environment._contact_impulse-base.environment._contact_impulse),
        elapsed_physical_seconds=float(a.environment.world.time-base.environment.world.time),
        source_limit='pfc_time_motor_recall includes return-line fallback; source name does not establish learned target or route behavior.')
    assert abs(motion['elapsed_physical_seconds'] - .1 * report['current_steps']) < 1e-7
    assert asdict(a.environment.world.config) == asdict(base.environment.world.config)
    assert np.array_equal(a.environment._initial_pose, base.environment._initial_pose)
    assert tree_digest(a.environment.world.walls) == tree_digest(base.environment.world.walls)
    scores = corrected_scores(report, report["catalog"])
    assert all(not row["changed_scores"] for row in scores)
    continuation = []
    with tempfile.TemporaryDirectory(prefix="original_continuity_audit_") as temp:
        temporary_life = Path(temp) / "roundtrip.npz"
        a.save(temporary_life)
        b = runner.LifeSession.load(temporary_life)
        assert tree_digest(a.snapshot()) == tree_digest(b.snapshot())
        _verify_parameters(b.brain, parameters)
        for frame in range(continuation_steps):
            # Actual agent-selected four-muscle forces in disposable bodies;
            # learning remains enabled to exercise restored forgetting counters.
            a.tick(); b.tick()
            digest_a, digest_b = tree_digest(a.snapshot()), tree_digest(b.snapshot())
            assert digest_a == digest_b, frame
            continuation.append(dict(frame=frame + 1, complete_state_sha256=digest_a,
                clock=int(a.brain.runtime.clock.当前时间), physical_time=float(a.environment.world.time),
                muscles=a.executed.tolist()))
        tables_after = _verify_parameters(a.brain, parameters)
        assert tree_digest(a.brain.snapshot()) == tree_digest(b.brain.snapshot())
    assert input_hashes == {path: _sha(path) for path in input_hashes}
    result = dict(status="complete", source_report=str(report_path),
        unchanged_input_sha256=input_hashes, parameters=parameters,
        reported_parameter_intervention=report['parameters'],
        inherited_custom_parameters=not bool(report['parameters']),
        inactive_config_defaults_added_on_load=added_defaults,
        optional_weight_dependent_plasticity_enabled=False,
        runtime_now_sha256=_sha(HERE / "original_runtime.py"),
        runtime_reported_sha256=report["source_sha256"]["original_runtime.py"],
        runtime_note="Current runtime may use an exact derived index cache; serialized canonical memory and equations are unchanged. Separate index-equivalence audit is external.",
        checkpoint_hash_present_in_original_report=bool(report.get("complete_life_sha256")),
        checkpoint_identity_evidence="Current SHA captured; configuration, exact terminal memory activations, clock, and control counts cross-checked with report.",
        saved_reference_retention=direct, retention_failures=failures,
        previous_report_identity=previous_identity,
        sensory_alignment=sensory,
        same_event_visual_content_diagnostics=scores,
        behavior_limit="These visual-content diagnostics do not grade action/route correctness; learned shortcuts can change intermediate recalled content. No navigation, logic-collapse, or AGI inference is made.",
        original_brain_from_snapshot_equal=True, full_life_roundtrip_equal=True,
        transient_prepared_action_retained=True,
        custom_parameters_and_maintenance_before=tables_before,
        custom_parameters_and_maintenance_after=tables_after,
        continuation_learning_enabled=True, continuation_frames=continuation,
        formal_life_or_report_written=False)
    output = report_path.with_name(report_path.stem + "_restore_audit.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("--previous-report", type=Path)
    args = parser.parse_args()
    result, output = audit_one(args.report, args.parameters, previous_report=args.previous_report)
    print(json.dumps({"status": result["status"], "output": str(output),
        "old_retention": result["saved_reference_retention"],
        "restored_real_learning_frames_identical": len(result["continuation_frames"])}, ensure_ascii=False))
