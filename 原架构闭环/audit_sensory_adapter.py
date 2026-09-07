"""Bounded adapter checks and the prerequested .50/.55/.60 threshold table.

No navigation training, reward, teacher actions, labels fed to a brain or
modification of original files. Actual random motor execution is used only to
exercise the original reverse-Hebbian motor interface in a disposable adapter.
"""
from __future__ import annotations

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


FREQUENCIES_TO_TEST = (0, 220, 440, 660, 990, 1320)


def tone(frequency):
    return .45 * np.sin(2 * np.pi * frequency * np.arange(800) / 8000)


def original_pfc(total_width):
    path = ROOT / "前额叶区_prefrontal.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ("扩散", "前额叶神经网络")]
    namespace = {"np": _LocalNumpy(260906)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace["前额叶神经网络"](total_width, 层数=3, 连接半径=10,
             权重范围=.1, 隐藏层阈值=.55, 输出层阈值=1.2, 扩散宽度=10)


def same_snapshot(a, b):
    left, right = a.snapshot(), b.snapshot()
    return left["metadata"] == right["metadata"] and left["arrays"].keys() == right["arrays"].keys() and all(
        np.array_equal(left["arrays"][name], right["arrays"][name]) for name in left["arrays"])


def assert_same_encoding(a, b):
    for name in ("visual", "audio", "motor"):
        assert np.array_equal(a[name], b[name])
    for group in ("input_arrays", "scalars"):
        assert a[group].keys() == b[group].keys()
        assert all(np.array_equal(a[group][name], b[group][name]) for name in a[group])


def threshold_table(packet):
    rows = []
    for threshold in (.50, .55, .60):
        adapter = SensoryAdapter(threshold=threshold)
        pfc = original_pfc(sum(adapter.widths.values()))
        codes, cue_codes, conditions = [], [], []
        for frequency in FREQUENCIES_TO_TEST:
            sample = copy.deepcopy(packet)
            sample["waveform"] = tone(frequency)
            encoded = adapter.encode(sample, np.zeros(4))
            cue = pfc.前向传播(np.zeros(adapter.widths["visual"], bool), encoded["audio"],
                             np.zeros(adapter.widths["motor"], bool)).astype(bool)
            full = pfc.前向传播(encoded["visual"], encoded["audio"], encoded["motor"]).astype(bool)
            codes.append(encoded["audio"].copy())
            cue_codes.append(cue.copy())
            conditions.append(dict(frequency_hz=frequency, visual_active=int(encoded["visual"].sum()),
                audio_active=int(encoded["audio"].sum()), motor_active=int(encoded["motor"].sum()),
                partial_audio_pfc_active=int(cue.sum()), full_pfc_active=int(full.sum()),
                audio_nonzero_scalar_bands=np.flatnonzero(encoded["scalars"]["audio"] > .05).tolist(),
                layer_diagnostics=encoded["diagnostics"]))
        rows.append(dict(threshold=threshold, conditions=conditions,
            audio_hamming_matrix=[[int(np.count_nonzero(a != b)) for b in codes] for a in codes],
            partial_pfc_hamming_matrix=[[int(np.count_nonzero(a != b)) for b in cue_codes] for a in cue_codes],
            audio_all_six_distinct=len({a.tobytes() for a in codes}) == 6,
            partial_pfc_all_six_distinct=len({a.tobytes() for a in cue_codes}) == 6))
    return rows


def main():
    started = time.perf_counter()
    original_names = ("视觉前处理_visual_preprocess.py", "听觉前处理_auditory_preprocess.py", "运动输出区_motor_output.py", "前额叶区_prefrontal.py")
    hashes_before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in original_names}
    checks = []
    global_rng = np.random.get_state()
    adapter = SensoryAdapter()
    after_rng = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(global_rng, after_rng))
    assert adapter.widths == dict(visual=2640, audio=2560, motor=80)
    checks.append("original classes instantiated at explicit widths without changing global RNG")
    env = BeaconNavigationEnv(scenario="barrier", seed=47011)
    packet = env.reset(start="left_middle", target="red")
    packet["waveform"] = tone(583)
    encoded = adapter.encode(packet, [0., .1, .6, 1.])
    assert all(np.all(array.reshape(-1, 2).sum(axis=1) == 1) for array in encoded["input_arrays"].values())
    assert np.array_equal(encoded["input_arrays"]["motor"], adapter.thermometer(np.array([0., .1, .6, 1.])))
    for name, network in adapter.networks.items():
        # Independently apply every stored original source/weight/threshold row.
        activity = encoded["input_arrays"][name].astype(bool)
        for sources, weights, threshold in zip(network.来源, network.权重, network.阈值):
            activity = (weights * activity[sources]).sum(axis=1) >= threshold
        output = encoded[name] if name in ("audio", "motor") else encoded["visual"][slice(*adapter.visual_segments[name])]
        assert np.array_equal(activity, output)
    assert adapter.motor_reverse.条数() == 0
    checks.append("all five segments exactly reproduce original fixed network equations; encode does not learn")
    poisoned = copy.deepcopy(packet)
    poisoned.update(goal="blue", target="blue", route="top", reward=1e6,
                    private_metrics={"correct_action": [1, 0, 0, 1]})
    poisoned["observation"].update(position=[9, 9], heading=3.14, target="blue", beacon_id=2)
    assert_same_encoding(encoded, adapter.encode(poisoned, [0., .1, .6, 1.]))
    checks.append("all extra target/route/reward/coordinate fields are ignored")
    depth_codes = []
    for distance in (6., 9., 12.):
        sample = copy.deepcopy(packet)
        sample["observation"]["ray_distances"] = np.full(31, distance)
        depth_codes.append(adapter.encode(sample, np.zeros(4))["input_arrays"]["depth"])
    assert len({code.tobytes() for code in depth_codes}) == 3
    checks.append("fixed linear 16 m depth scale distinguishes 6/9/12 m paired receptor codes")
    for bad in (np.zeros(799), np.full(800, np.nan), np.ones(800) * 2):
        before = adapter.snapshot()
        invalid = copy.deepcopy(packet)
        invalid["waveform"] = bad
        try:
            adapter.encode(invalid, np.zeros(4))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid PCM accepted")
        assert same_snapshot(adapter, SensoryAdapter.from_snapshot(before))
    checks.append("invalid PCM rejected before neural state mutation")
    # Exercise actual autonomous motor execution, not a labeled demonstration.
    rng = np.random.default_rng(17003)
    motor_rows = []
    for frame in range(24):
        muscles = rng.random(4)
        packet = env.step(muscles, .1)
        before_edges = adapter.motor_reverse.条数()
        code = adapter.learn_executed(muscles)
        decoded = adapter.decode_motor(code)
        motor_rows.append(dict(frame=frame, actual_muscles=muscles.tolist(),
            motor_code_active=int(code.sum()), decoded_muscles=decoded.tolist(),
            new_reverse_edges=adapter.motor_reverse.条数() - before_edges))
        assert np.isfinite(decoded).all() and np.all((decoded >= 0) & (decoded <= 1))
    assert adapter.motor_learning_steps == 24
    assert adapter.motor_reverse.条数() == np.count_nonzero(adapter.motor_reverse.矩阵)
    checks.append("24 actual autonomous physical muscle executions feed original reverse-Hebbian learning")
    with tempfile.TemporaryDirectory(prefix="original-adapter-audit-") as directory:
        destination = Path(directory) / "adapter.npz"
        adapter.save(destination)
        with np.load(destination, allow_pickle=False) as archive:
            assert all(archive[name].dtype != object for name in archive.files)
        restored = SensoryAdapter.load(destination)
        assert same_snapshot(adapter, restored)
        for _ in range(12):
            muscles = rng.random(4)
            packet = env.step(muscles, .1)
            left = adapter.encode(packet, muscles)
            right = restored.encode(packet, muscles)
            assert_same_encoding(left, right)
            adapter.learn_motor_from_encoded(left)
            restored.learn_motor_from_encoded(right)
            assert np.array_equal(adapter.decode_motor(left["motor"]), restored.decode_motor(right["motor"]))
            assert same_snapshot(adapter, restored)
    checks.append("pickle-free atomic save/load preserves every array and 12 further online updates exactly")
    invalid_snapshot = adapter.snapshot()
    invalid_snapshot["arrays"]["fixed_audio_weights_0"][0, 0] += .001
    try:
        SensoryAdapter.from_snapshot(invalid_snapshot)
    except ValueError:
        pass
    else:
        raise AssertionError("changed birth accepted")
    checks.append("changed fixed birth connection is rejected on restoration")
    table_env = BeaconNavigationEnv(scenario="barrier", seed=47011)
    table = threshold_table(table_env.reset(start="left_middle", target="red"))
    real_view_rows = []
    real_view_codes = []
    fresh = SensoryAdapter()
    for start in ("left_middle", "left_bottom", "right_middle", "bottom_middle"):
        seen = fresh.encode(table_env.reset(start=start, target="red"), np.zeros(4))
        real_view_codes.append(seen["visual"].copy())
        real_view_rows.append(dict(start=start, visual_active=int(seen["visual"].sum()),
                                   segment_active=seen["diagnostics"]))
    hashes_after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in original_names}
    assert hashes_before == hashes_after
    result = dict(scope="Original sensory and motor classes with explicit physical adapters, not a navigation result",
        checks_passed=checks, widths=adapter.widths, source_sha256=hashes_before,
        original_files_unchanged=True, elapsed_seconds=time.perf_counter() - started,
        threshold_table=table, real_view_rows=real_view_rows,
        real_view_hamming_matrix=[[int(np.count_nonzero(a != b)) for b in real_view_codes] for a in real_view_codes],
        actual_motor_execution_rows=motor_rows,
        motor_mean_absolute_reconstruction_error=float(np.mean([np.abs(np.array(row["actual_muscles"])
            - np.array(row["decoded_muscles"])).mean() for row in motor_rows])),
        limitations=["The original all-pairs one-shot reverse table is a fuzzy motor reconstruction; saturation and errors are reported, not repaired by a classifier.",
            "Different sensory codes are not proof of image understanding, target retention, language, planning or navigation.",
            "The predeclared small threshold table changes only allowed threshold parameters; the adapter default remains the original global .60.",
            "Bodies and depth are explicitly separate segments inside visual; no 1x31 image is resampled to the original 24x16 layout."])
    destination = Path(__file__).resolve().parent / "results" / "sensory_adapter_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(checks=len(checks), elapsed=result["elapsed_seconds"],
        threshold_summary=[dict(threshold=row["threshold"],
            audio_distinct=row["audio_all_six_distinct"], pfc_distinct=row["partial_pfc_all_six_distinct"],
            activity=[{key: r[key] for key in ("frequency_hz", "audio_active", "partial_audio_pfc_active", "full_pfc_active")}
                      for r in row["conditions"]]) for row in table],
        motor_mean_absolute_error=result["motor_mean_absolute_reconstruction_error"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
