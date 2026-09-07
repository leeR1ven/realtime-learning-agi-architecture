"""Predeclared finite calibration of ORIGINAL sensory/PFC parameters.

No learned cue labels, action classifier or navigation reward. Each candidate
uses all three fixed birth seeds. This file owns its disposable adapters only.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from sensory_adapter import SensoryAdapter, ROOT, _LocalNumpy

sys.path.insert(0, str(ROOT / "色标导航"))
from environment import BeaconNavigationEnv


SEEDS = (2026, 2027, 2028)
SENSORY_THRESHOLDS = (.52, .54, .55, .56, .58)
SPREADS = (0, 1, 2, 3, 5, 10)
HIDDEN_THRESHOLDS = (.4, .5, .55)
OUTPUT_THRESHOLDS = (.6, .8, 1., 1.2)
FREQUENCIES = (0, 220, 440, 660, 990, 1320, 1750)
MOTOR_PATTERNS = np.array([[0, 0, 0, 0], [.6, 0, .6, 0], [0, .6, 0, .6],
                           [0, .3, .3, 0], [.3, 0, 0, .3]], np.float64)


def pfc_class():
    path = ROOT / "前额叶区_prefrontal.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ("扩散", "前额叶神经网络")]
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec")


PFC_COMPILED = pfc_class()


def make_pfc(width, seed, spread, hidden, output):
    namespace = {"np": _LocalNumpy(seed + 10001)}
    exec(PFC_COMPILED, namespace)
    return namespace["前额叶神经网络"](width, 隐藏层阈值=hidden,
                                   输出层阈值=output, 扩散宽度=spread)


def tone(hz):
    return .45 * np.sin(2 * np.pi * hz * np.arange(800) / 8000)


def code_stats(codes):
    values = np.asarray(codes, bool)
    counts = values.sum(axis=1)
    intersection = values.astype(np.int32) @ values.astype(np.int32).T
    distances = counts[:, None] + counts[None, :] - 2 * intersection
    offdiag = ~np.eye(len(values), dtype=bool)
    fractions = intersection / np.maximum(counts[:, None], 1)
    return dict(active=counts.astype(int).tolist(), all_nonempty=bool(np.all(counts > 0)),
        all_distinct=len({row.tobytes() for row in values}) == len(values),
        minimum_hamming=int(distances[offdiag].min()),
        max_directed_overlap=float(fractions[offdiag].max()),
        pairwise_hamming=distances.astype(int).tolist())


def stimuli():
    env = BeaconNavigationEnv(scenario="barrier", seed=57301)
    base = env.reset(start="left_middle", target="red")
    base["waveform"] = np.zeros(800)
    views = []
    for start in ("left_middle", "left_bottom", "right_middle", "bottom_middle"):
        packet = env.reset(start=start, target="red")
        packet["waveform"] = np.zeros(800)
        views.append((start, copy.deepcopy(packet)))
    # Each calibration target below really executes in the same physical env.
    # There is no desired action label; the controller's own RNG emits forces.
    actions = np.random.default_rng(177301).random((48, 4))
    actual_packets = []
    for muscles in actions:
        actual_packets.append(copy.deepcopy(env.step(muscles, .1)))
    return base, views, actions, actual_packets


def motor_metrics(adapter, actions):
    # 24 executed samples are learned; later 24 executed samples are held out.
    encoded = []
    raw = []
    for muscles in actions:
        signal = adapter._motor_namespace["发力转信号"](muscles)
        code = adapter.networks["motor"].前向传播(signal).astype(bool, copy=True)
        encoded.append(code)
        raw.append(adapter.pools["motor"].激活.copy())
    encoded, raw = np.asarray(encoded), np.asarray(raw)
    newborn = np.array([adapter.decode_motor(code) for code in encoded])
    shuffled_adapter = SensoryAdapter.from_snapshot(adapter.snapshot())
    permutation = np.random.default_rng(177302).permutation(24)
    for i in range(24):
        adapter.motor_reverse.学习一帧(raw[i], encoded[i])
        shuffled_adapter.motor_reverse.学习一帧(raw[i], encoded[permutation[i]])
    trained = np.array([adapter.decode_motor(code) for code in encoded])
    shuffled = np.array([shuffled_adapter.decode_motor(code) for code in encoded])
    def scores(output, section):
        expected = actions[section]
        actual = output[section]
        net_expected = np.c_[expected[:, 0] - expected[:, 1], expected[:, 2] - expected[:, 3]]
        net_actual = np.c_[actual[:, 0] - actual[:, 1], actual[:, 2] - actual[:, 3]]
        twist_expected = net_expected[:, 1] - net_expected[:, 0]
        twist_actual = net_actual[:, 1] - net_actual[:, 0]
        notable = np.abs(twist_expected) >= .2
        return dict(mae=float(np.abs(actual - expected).mean()),
            net_force_mae=float(np.abs(net_actual - net_expected).mean()),
            torque_mae=float(np.abs(twist_actual - twist_expected).mean()),
            turning_sign_accuracy=float(np.mean(np.sign(twist_actual[notable]) == np.sign(twist_expected[notable]))),
            prediction_std=float(actual.std(axis=0).mean()))
    seen, heldout = slice(0, 24), slice(24, 48)
    result = dict(original_reverse_edges=adapter.motor_reverse.条数(),
        actual_execution_count=48, learned_count=24, heldout_count=24,
        active_codes=encoded.sum(axis=1).astype(int).tolist(),
        seen={name: scores(output, seen) for name, output in
              (("newborn", newborn), ("paired", trained), ("shuffled_pairs", shuffled))},
        heldout={name: scores(output, heldout) for name, output in
                 (("newborn", newborn), ("paired", trained), ("shuffled_pairs", shuffled))})
    result["useful_on_heldout"] = (result["heldout"]["paired"]["mae"] < .20 and
        result["heldout"]["paired"]["mae"] < min(result["heldout"][name]["mae"] for name in ("newborn", "shuffled_pairs")) - .02)
    return result


def diffuse_segments(rows, widths, spread):
    result = []
    start = 0
    for width in widths:
        segment = rows[:, start:start + width]
        expanded = segment.copy()
        for offset in range(1, spread + 1):
            expanded |= np.roll(segment, offset, axis=1)
            expanded |= np.roll(segment, -offset, axis=1)
        result.append(expanded)
        start += width
    return np.concatenate(result, axis=1)


def run():
    begin = time.perf_counter()
    base, views, actions, physical_packets = stimuli()
    source_files = [ROOT / name for name in ("视觉前处理_visual_preprocess.py", "听觉前处理_auditory_preprocess.py", "运动输出区_motor_output.py", "前额叶区_prefrontal.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    rows = []
    sensory = []
    verification_count = 0
    for threshold in SENSORY_THRESHOLDS:
        for seed in SEEDS:
            adapter = SensoryAdapter(seed=seed, threshold=threshold)
            widths = list(adapter.widths.values())
            audio_codes, full_codes = [], []
            for frequency in FREQUENCIES:
                packet = copy.deepcopy(base)
                packet["waveform"] = tone(frequency)
                code = adapter.encode(packet, np.zeros(4))
                audio_codes.append(code["audio"])
                full_codes.append(np.concatenate([code[k] for k in ("visual", "audio", "motor")]))
            visual_codes = [adapter.encode(packet, np.zeros(4))["visual"] for _, packet in views]
            motor_codes = []
            for muscles in MOTOR_PATTERNS:
                motor_codes.append(adapter.encode(base, muscles)["motor"])
            metric = dict(seed=seed, sensory_threshold=threshold, audio=code_stats(audio_codes),
                visual=code_stats(visual_codes), motor_patterns=code_stats(motor_codes),
                motor_reconstruction=motor_metrics(adapter, actions))
            sensory.append(metric)
            cue_rows = np.concatenate((np.zeros((len(FREQUENCIES), widths[0]), bool),
                                       np.array(audio_codes), np.zeros((len(FREQUENCIES), widths[2]), bool)), axis=1)
            all_rows = np.concatenate((cue_rows, full_codes), axis=0)
            # All candidates share the SAME ORIGINAL BIRTH connection values.
            reference = make_pfc(sum(widths), seed, 0, .55, 1.2)
            for spread in SPREADS:
                expanded = diffuse_segments(all_rows, widths, spread)
                first_current = (reference.权重[0][None, :, :] * expanded[:, reference.来源[0]]).sum(axis=2)
                for hidden in HIDDEN_THRESHOLDS:
                    first = first_current >= hidden
                    second_current = (reference.权重[1][None, :, :] * first[:, reference.来源[1]]).sum(axis=2)
                    for output in OUTPUT_THRESHOLDS:
                        result = second_current >= output
                        cues, full = result[:len(FREQUENCIES)], result[len(FREQUENCIES):]
                        pfc_stats = code_stats(cues)
                        row = dict(seed=seed, sensory_threshold=threshold, spread=spread,
                            hidden_threshold=hidden, output_threshold=output,
                            audio_all_distinct=metric["audio"]["all_distinct"],
                            visual_all_nonempty=metric["visual"]["all_nonempty"],
                            visual_all_distinct=metric["visual"]["all_distinct"],
                            motor_patterns_all_distinct=metric["motor_patterns"]["all_distinct"],
                            motor_useful_on_heldout=metric["motor_reconstruction"]["useful_on_heldout"],
                            partial_pfc=pfc_stats, full_pfc_active=full.sum(axis=1).astype(int).tolist())
                        row["sensory_pfc_acceptable"] = (row["audio_all_distinct"] and
                            row["visual_all_nonempty"] and row["visual_all_distinct"] and
                            row["motor_patterns_all_distinct"] and pfc_stats["all_distinct"] and
                            pfc_stats["all_nonempty"] and int(full.sum(axis=1).max()) <= 300)
                        rows.append(row)
                        # Every requested parameter combination is independently
                        # checked against the original class for one real row.
                        reference.扩散宽度 = spread
                        reference.阈值[0][:] = hidden
                        reference.阈值[1][:] = output
                        source = all_rows[0]
                        exact = reference.前向传播(source[:widths[0]], source[widths[0]:widths[0]+widths[1]], source[-widths[2]:])
                        assert np.array_equal(exact, result[0])
                        verification_count += 1
            print(json.dumps(dict(progress_threshold=threshold, seed=seed,
                audio_distinct=metric["audio"]["all_distinct"],
                motor_distinct=metric["motor_patterns"]["all_distinct"],
                motor_heldout_mae=metric["motor_reconstruction"]["heldout"]["paired"]["mae"])), flush=True)
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in ("sensory_threshold", "spread", "hidden_threshold", "output_threshold"))
        grouped.setdefault(key, []).append(row)
    candidates = []
    for key, group in grouped.items():
        if all(row["sensory_pfc_acceptable"] for row in group):
            candidates.append(dict(config=dict(zip(("sensory_threshold", "pfc_spread", "pfc_hidden_threshold", "pfc_output_threshold"), key)),
                worst_partial_overlap=max(row["partial_pfc"]["max_directed_overlap"] for row in group),
                minimum_partial_hamming=min(row["partial_pfc"]["minimum_hamming"] for row in group),
                maximum_full_activity=max(max(row["full_pfc_active"]) for row in group),
                all_motor_useful=all(row["motor_useful_on_heldout"] for row in group)))
    candidates.sort(key=lambda row: (row["worst_partial_overlap"], -row["minimum_partial_hamming"], row["maximum_full_activity"]))
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in source_files)
    return dict(protocol=dict(seeds=SEEDS, sensory_thresholds=SENSORY_THRESHOLDS, spreads=SPREADS,
        hidden_thresholds=HIDDEN_THRESHOLDS, output_thresholds=OUTPUT_THRESHOLDS,
        frequencies_hz=FREQUENCIES, motor_patterns=MOTOR_PATTERNS.tolist(),
        pfc_birth_seed="adapter seed + 10001, matches current brain.py", maximum_full_pfc_activity=300,
        motor_acceptance="heldout MAE < .20 and at least .02 better than both same-birth newborn and shuffled pairings",
        calibration_action_rng_seed=177301, pairing_shuffle_seed=177302,
        note="Parameter selection uses all three predefined seeds. These are calibration data, not held-out task validation."),
        source_sha256=hashes, original_files_unchanged=True,
        elapsed_seconds=time.perf_counter()-begin,
        original_class_equivalence_checks=verification_count,
        sensory_rows=sensory, pfc_rows=rows, acceptable_sensory_pfc_candidates=candidates,
        full_joint_candidates=[row for row in candidates if row["all_motor_useful"]],
        physical_action_history=actions.tolist(),
        actual_physical_elapsed_seconds=48*.1,
        limitations=["Exact code distinction does not guarantee hippocampal partial-cue disambiguation; directed overlaps are reported separately.",
            "Motor calibration has separate seen/heldout actual actions and shuffled-pair controls; it is not navigation or supervised route demonstration."])


def run_contrast(motor_config=None):
    """Second registered step: authorized fixed contrast, no frequency labels."""
    begin = time.perf_counter()
    motor_config = dict(motor_config or {"motor_threshold": .52})
    base, views, actions, _ = stimuli()
    rows, frontend_rows = [], []
    for visual_threshold in (.56, .58):
        for audio_threshold in (.52, .54, .55):
            for seed in SEEDS:
                config = dict(seed=seed, threshold=.6, visual_threshold=visual_threshold,
                    audio_threshold=audio_threshold, contrast_to_silence=True, **motor_config)
                adapter = SensoryAdapter(**config)
                widths = list(adapter.widths.values())
                audio_codes, full_codes = [], []
                for frequency in FREQUENCIES:
                    packet = copy.deepcopy(base)
                    packet["waveform"] = tone(frequency)
                    code = adapter.encode(packet, np.zeros(4))
                    assert np.array_equal(code["audio"], code["raw_outputs"]["audio"] ^ adapter.audio_silence_baseline)
                    audio_codes.append(code["audio"])
                    full_codes.append(np.concatenate([code[k] for k in ("visual", "audio", "motor")]))
                visual_codes = [adapter.encode(packet, np.zeros(4))["visual"] for _, packet in views]
                front = dict(seed=seed, visual_threshold=visual_threshold, audio_threshold=audio_threshold,
                    audio=code_stats(audio_codes), visual=code_stats(visual_codes),
                    silent_output_is_empty=not bool(audio_codes[0].any()),
                    silence_baseline_sha256=hashlib.sha256(adapter.audio_silence_baseline.tobytes()).hexdigest())
                frontend_rows.append(front)
                cue_rows = np.concatenate((np.zeros((len(FREQUENCIES), widths[0]), bool),
                    audio_codes, np.zeros((len(FREQUENCIES), widths[2]), bool)), axis=1)
                combined = np.concatenate((cue_rows, full_codes), axis=0)
                pfc = make_pfc(sum(widths), seed, 0, .55, 1.2)
                for spread in SPREADS:
                    expanded = diffuse_segments(combined, widths, spread)
                    first_current = (pfc.权重[0][None] * expanded[:, pfc.来源[0]]).sum(axis=2)
                    for hidden in HIDDEN_THRESHOLDS:
                        first = first_current >= hidden
                        second_current = (pfc.权重[1][None] * first[:, pfc.来源[1]]).sum(axis=2)
                        for output in OUTPUT_THRESHOLDS:
                            result = second_current >= output
                            cues, full = result[:len(FREQUENCIES)], result[len(FREQUENCIES):]
                            stats = code_stats(cues)
                            acceptable = (front["audio"]["all_distinct"] and stats["all_distinct"] and
                                all(n > 0 for n in stats["active"][1:]) and stats["active"][0] == 0 and
                                stats["max_directed_overlap"] < .9 and front["visual"]["all_distinct"] and
                                front["visual"]["all_nonempty"] and int(full.sum(axis=1).max()) <= 300)
                            rows.append(dict(seed=seed, visual_threshold=visual_threshold,
                                audio_threshold=audio_threshold, spread=spread, hidden_threshold=hidden,
                                output_threshold=output, partial_pfc=stats,
                                full_pfc_active=full.sum(axis=1).astype(int).tolist(), acceptable=acceptable))
                print(json.dumps(dict(contrast=True, seed=seed, visual_threshold=visual_threshold,
                    audio_threshold=audio_threshold, audio_activity=front["audio"]["active"])), flush=True)
    grouped = {}
    fields = ("visual_threshold", "audio_threshold", "spread", "hidden_threshold", "output_threshold")
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in fields), []).append(row)
    candidates = []
    for key, group in grouped.items():
        if all(row["acceptable"] for row in group):
            candidates.append(dict(config=dict(zip(fields, key)),
                worst_overlap=max(row["partial_pfc"]["max_directed_overlap"] for row in group),
                minimum_hamming=min(row["partial_pfc"]["minimum_hamming"] for row in group),
                max_activity=max(max(row["full_pfc_active"]) for row in group)))
    candidates.sort(key=lambda row: (row["worst_overlap"], -row["minimum_hamming"], row["max_activity"]))
    result = dict(phase="Explicitly authorized fixed audio XOR-to-silence output layer",
        protocol=dict(seeds=SEEDS, frequencies_hz=FREQUENCIES, audio_thresholds=[.52,.54,.55],
            visual_thresholds=[.56,.58], spreads=SPREADS, hidden_thresholds=HIDDEN_THRESHOLDS,
            output_thresholds=OUTPUT_THRESHOLDS, silence_must_be_zero=True,
            known_cue_directed_overlap_limit=.9, full_pfc_activity_limit=300),
        frontend_rows=frontend_rows, pfc_rows=rows, candidates=candidates,
        motor_config=motor_config,
        elapsed_seconds=time.perf_counter()-begin)
    if candidates:
        best = candidates[0]["config"]
        # Fixed relative offsets probe unseen nearby pitches; distant sounds are
        # not assigned labels. Report every directed containment, including 1.
        probes = (198, 237, 411, 472, 634, 683, 963, 1023, 1297, 1347, 1733, 1783,
                  307, 777, 2047, 3333)
        unknown_rows = []
        for seed in SEEDS:
            adapter = SensoryAdapter(seed=seed, threshold=.6, visual_threshold=best["visual_threshold"],
                audio_threshold=best["audio_threshold"], contrast_to_silence=True, **motor_config)
            pfc = make_pfc(sum(adapter.widths.values()), seed, best["spread"], best["hidden_threshold"], best["output_threshold"])
            audio, thoughts = [], []
            for frequency in FREQUENCIES[1:] + probes:
                packet = copy.deepcopy(base)
                packet["waveform"] = tone(frequency)
                code = adapter.encode(packet, np.zeros(4))["audio"]
                audio.append(code)
                thoughts.append(pfc.前向传播(np.zeros(adapter.widths["visual"], bool), code,
                                  np.zeros(adapter.widths["motor"], bool)).astype(bool))
            def directed(codes):
                values = np.array(codes, np.int32)
                return ((values[6:] @ values[:6].T) / np.maximum(values[6:].sum(axis=1)[:,None], 1)).tolist()
            unknown_rows.append(dict(seed=seed, audio_directed_unknown_to_known=directed(audio),
                pfc_directed_unknown_to_known=directed(thoughts),
                audio_unknown_active=[int(code.sum()) for code in audio[6:]],
                pfc_unknown_active=[int(code.sum()) for code in thoughts[6:]]))
        result["unknown_pitch_probe"] = dict(known_hz=FREQUENCIES[1:], query_hz=probes, rows=unknown_rows,
            limitation="Nearby pitches may be the same bucket or a subset. This is a diagnostic, not unknown-sound rejection or language.")
        result["provisional_config"] = best
    return result


def run_joint():
    path = Path(__file__).resolve().parent / "results" / "calibration_motor-wide.json"
    motor = json.loads(path.read_text(encoding="utf-8"))["candidates"][0]["config"]
    result = run_contrast(motor)
    result["phase"] = "Joint recheck after motor width change; original birth graph regenerated at exact resulting width"
    return result


def validate_recommendation():
    """One fixed final candidate; new physical action RNG, no retuning here."""
    directory = Path(__file__).resolve().parent / "results"
    joint = json.loads((directory / "calibration_joint.json").read_text(encoding="utf-8"))
    selected = joint["candidates"][0]["config"]
    adapter_config = dict(threshold=.6, visual_threshold=selected["visual_threshold"],
        audio_threshold=selected["audio_threshold"], contrast_to_silence=True, **joint["motor_config"])
    pfc_config = dict(spread=selected["spread"], hidden=selected["hidden_threshold"], output=selected["output_threshold"])
    recommendation = dict(adapter_kwargs=adapter_config,
        pfc_kwargs=dict(pfc_spread=pfc_config["spread"], pfc_hidden_threshold=pfc_config["hidden"],
                        pfc_output_threshold=pfc_config["output"]),
        widths=SensoryAdapter(**adapter_config).widths, seed_protocol=list(SEEDS),
        new_structure="One fixed per-cell audio XOR-to-silence layer after the unchanged original auditory network; positive and negative deviations retain original cell identity.",
        original_parameter_changes="Per-modality thresholds; original PFC spread/output threshold; 40 thermometer pairs per muscle and the original reverse-Hebb window distance 20; original one-shot update unchanged.",
        calibration_files=["calibration.json", "calibration_contrast.json", "calibration_motor.json", "calibration_motor-wide.json", "calibration_joint.json"],
        validation_action_rng_seed=177401)
    # Fix the candidate before seeing the independent-action results.
    (directory / "recommended_config.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    actions = np.random.default_rng(177401).random((48, 4))
    physical_env = BeaconNavigationEnv(scenario="barrier", seed=177402)
    physical_env.reset(start="left_middle", target="red")
    for muscles in actions:
        last_packet = physical_env.step(muscles, .1)
    rows = []
    for seed in SEEDS:
        adapter = SensoryAdapter(seed=seed, **adapter_config)
        pfc = make_pfc(sum(adapter.widths.values()), seed, **pfc_config)
        audio, full, original_rows = [], [], []
        for frequency in FREQUENCIES:
            packet = copy.deepcopy(last_packet)
            packet["waveform"] = tone(frequency)
            encoded = adapter.encode(packet, np.zeros(4))
            audio.append(pfc.前向传播(np.zeros(adapter.widths["visual"], bool), encoded["audio"],
                        np.zeros(adapter.widths["motor"], bool)).astype(bool))
            full.append(pfc.前向传播(encoded["visual"], encoded["audio"], encoded["motor"]).astype(bool))
            original_rows.append(dict(frequency_hz=frequency, raw_audio_active=int(encoded["raw_outputs"]["audio"].sum()),
                contrast_active=int(encoded["audio"].sum()),
                positive_deviations=len(encoded["diagnostics"]["audio"]["positive_deviation_ids"]),
                negative_deviations=len(encoded["diagnostics"]["audio"]["negative_deviation_ids"])))
        visual_rows = []
        for scenario in ("barrier", "offset", "crossbar"):
            env = BeaconNavigationEnv(scenario=scenario, seed=177403)
            codes = []
            for start in ("left_middle", "left_bottom", "right_middle", "bottom_middle"):
                packet = env.reset(start=start, target="red")
                code = adapter.encode(packet, np.zeros(4))
                codes.append(code["visual"])
            visual_rows.append(dict(scenario=scenario, visual=code_stats(codes)))
        pattern_codes = [adapter.encode(last_packet, muscles)["motor"] for muscles in MOTOR_PATTERNS]
        motor_result = motor_metrics(adapter, actions)
        with tempfile.TemporaryDirectory(prefix="calibrated-original-adapter-") as temporary:
            path = Path(temporary) / "adapter.npz"
            adapter.save(path)
            restored = SensoryAdapter.load(path)
            for snapshot in (adapter.snapshot(), restored.snapshot()):
                assert "fixed_audio_silence_baseline" in snapshot["arrays"]
            for i in range(6):
                packet = copy.deepcopy(last_packet)
                packet["waveform"] = tone(FREQUENCIES[i + 1])
                first, second = adapter.encode(packet, actions[i]), restored.encode(packet, actions[i])
                for key in ("visual", "audio", "motor"):
                    assert np.array_equal(first[key], second[key])
                adapter.learn_motor_from_encoded(first)
                restored.learn_motor_from_encoded(second)
                assert np.array_equal(adapter.decode_motor(first["motor"]), restored.decode_motor(second["motor"]))
                sa, sb = adapter.snapshot(), restored.snapshot()
                assert sa["metadata"] == sb["metadata"]
                assert all(np.array_equal(sa["arrays"][key], sb["arrays"][key]) for key in sa["arrays"])
        rows.append(dict(seed=seed, known_audio_partial_pfc=code_stats(audio),
            full_pfc_active=[int(code.sum()) for code in full], audio_contrast_trace=original_rows,
            visual_scenarios=visual_rows, basic_motor_pattern_codes=code_stats(pattern_codes),
            independent_motor_validation=motor_result,
            safe_snapshot_and_six_online_steps_exact=True))
    return dict(phase="Frozen recommendation, independent actual motor action sequence and exact original-PFC calls",
        recommended_config=recommendation, rows=rows,
        independent_action_rng_seed=177401, independently_executed_actions=actions.tolist(),
        retuned_after_validation=False,
        limitations=["Nearby pitches can fully overlap a known cue. No natural-language or arbitrary unknown-sound rejection claim.",
            "Force reconstruction is approximate and checked after 24 examples; long-life saturation remains unvalidated.",
            "The 300-cell bound was calibrated on predefined stationary views. New views and simultaneously firing recalled content can exceed it."])


def run_motor(level_options=(10,)):
    begin = time.perf_counter()
    base, _, actions, _ = stimuli()
    contrast_path = Path(__file__).resolve().parent / "results" / "calibration_contrast.json"
    contrast = json.loads(contrast_path.read_text(encoding="utf-8"))
    chosen = contrast["candidates"][0]["config"]
    rows = []
    thresholds = (.48, .50, .52, .54, .55)
    windows = (None, 10, 20)
    for level_count in level_options:
      for threshold in thresholds:
        for window in windows:
            for once in (True, False):
                for seed in SEEDS:
                    adapter = SensoryAdapter(seed=seed, threshold=.6,
                        visual_threshold=chosen["visual_threshold"], audio_threshold=chosen["audio_threshold"],
                        motor_threshold=threshold, motor_reverse_window=window, motor_once=once,
                        contrast_to_silence=True, motor_levels=level_count)
                    patterns = [adapter.encode(base, muscles)["motor"] for muscles in MOTOR_PATTERNS]
                    metric = motor_metrics(adapter, actions)
                    # A useful actuator must also preserve turn direction, not
                    # merely emit the mean activation of its training samples.
                    acceptable = (metric["useful_on_heldout"] and
                        metric["heldout"]["paired"]["torque_mae"] < .30 and
                        metric["heldout"]["paired"]["turning_sign_accuracy"] >= .70 and
                        code_stats(patterns)["all_distinct"])
                    rows.append(dict(seed=seed, motor_threshold=threshold, motor_reverse_window=window,
                        motor_once=once, motor_levels=level_count, pattern_codes=code_stats(patterns),
                        reconstruction=metric, acceptable=acceptable))
    grouped = {}
    for row in rows:
        key = (row["motor_threshold"], row["motor_reverse_window"], row["motor_once"], row["motor_levels"])
        grouped.setdefault(key, []).append(row)
    candidates = []
    for key, group in grouped.items():
        if all(row["acceptable"] for row in group):
            candidates.append(dict(config=dict(zip(("motor_threshold", "motor_reverse_window", "motor_once", "motor_levels"), key)),
                worst_mae=max(row["reconstruction"]["heldout"]["paired"]["mae"] for row in group),
                worst_torque_mae=max(row["reconstruction"]["heldout"]["paired"]["torque_mae"] for row in group),
                min_turn_accuracy=min(row["reconstruction"]["heldout"]["paired"]["turning_sign_accuracy"] for row in group)))
    candidates.sort(key=lambda row: (row["worst_torque_mae"], row["worst_mae"]))
    return dict(phase="Original reverse-Hebb existing window/one-shot/count parameters; no new learning rule",
        protocol=dict(seeds=SEEDS, motor_thresholds=thresholds, original_reverse_windows=windows,
            once_options=[True, False], motor_levels=list(level_options), actual_training_count=24, actual_heldout_count=24,
            acceptance="heldout force MAE < .20 and .02 below newborn and shuffled; torque MAE < .30; sign accuracy >= .70 for |true torque|>=.20"),
        sensory_pfc_config=chosen, rows=rows, candidates=candidates, elapsed_seconds=time.perf_counter()-begin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("original", "contrast", "motor", "motor-wide", "joint", "validate"), default="original")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "calibration.json")
    args = parser.parse_args()
    result = {"original": run, "contrast": run_contrast, "motor": run_motor,
              "motor-wide": lambda: run_motor((20, 40, 64)), "joint": run_joint,
              "validate": validate_recommendation}[args.phase]()
    if args.phase != "original" and args.output.name == "calibration.json":
        args.output = args.output.with_name("calibration_" + args.phase + ".json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(elapsed=result.get("elapsed_seconds"),
        candidates=len(result.get("acceptable_sensory_pfc_candidates", result.get("candidates", []))),
        best=result.get("acceptable_sensory_pfc_candidates", result.get("candidates", []))[:5]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
