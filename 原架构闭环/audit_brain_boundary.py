"""Independent short physical/time/checkpoint audit; does not edit the brain.

Uses default original brain parameters, eight initial physical frames and frozen
plasticity. Two temporary numerical checkpoint roundtrips each continue for ten
real physics frames. This audits interfaces and exact continuation, not learning
quality or navigation success.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from brain import OriginalBrain, packet_digest
from experience_environment import ExperienceEnvironment

HERE = Path(__file__).resolve().parent


def equal_tree(left, right, path="root"):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict), (path, type(left).__name__, type(right).__name__)
        assert left.keys() == right.keys(), path
        for key in left:
            equal_tree(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, (tuple, list)), path
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, path + "/" + str(index))
    else:
        assert left == right, (path, left, right)


def digest_tree(value):
    h = hashlib.sha256()

    def update(item):
        if isinstance(item, np.ndarray):
            h.update(b"array" + str(item.dtype).encode() + str(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                h.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (tuple, list)):
            for element in item:
                update(element)
        elif isinstance(item, np.generic):
            update(item.item())
        else:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    update(value)
    return h.hexdigest()


def validate_invalid_outcomes(brain, packet, muscles):
    checks = []
    mutations = {
        "nan_rgb": lambda p: p["observation"]["ray_colors"].__setitem__((0, 0), np.nan),
        "wrong_rgb_shape": lambda p: p["observation"].__setitem__("ray_colors", np.zeros((3, 3))),
        "nan_pcm": lambda p: p["waveform"].__setitem__(0, np.nan),
        "wrong_pcm_length": lambda p: p.__setitem__("waveform", np.zeros(79)),
        "negative_distance": lambda p: p["observation"]["ray_distances"].__setitem__(0, -1.),
        "invalid_touch": lambda p: p["observation"]["touch"].__setitem__(0, 2.),
        "private_position": lambda p: p["observation"].__setitem__("position", np.zeros(2)),
        "private_target": lambda p: p.__setitem__("target", "red"),
    }
    for label, modify in mutations.items():
        bad = copy.deepcopy(packet)
        modify(bad)
        before = brain.snapshot()
        try:
            brain.observe_outcome(bad, muscles)
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise AssertionError("Invalid outcome accepted: " + label)
        equal_tree(before, brain.snapshot(), label)
        checks.append(label)
    for label, applied, terminal in (
            ("different_applied_muscles", 1. - muscles, False),
            ("non_boolean_terminal", muscles, 1)):
        before = brain.snapshot()
        try:
            brain.observe_outcome(packet, applied, terminal=terminal)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action/terminal accepted: " + label)
        equal_tree(before, brain.snapshot(), label)
        checks.append(label)
    return checks


def checkpoint_continuation(brain, env, state_kind, folder):
    if state_kind == "pending":
        brain.observe_then_act(env.sensor_packet(), learning=False)
        assert brain.pending is not None and brain.prepared is None
    else:
        assert brain.pending is None and brain.prepared is not None
    checkpoint = folder / (state_kind + ".npz")
    brain.save(checkpoint)
    restored = OriginalBrain.load(checkpoint)
    other_env = copy.deepcopy(env)
    equal_tree(brain.snapshot(), restored.snapshot(), state_kind + "/loaded")
    initial_frame = brain.runtime.frames_recorded
    initial_clock = int(brain.runtime.clock.当前时间)
    trajectory = []
    for frame in range(10):
        if brain.pending is not None:
            muscles = brain.pending["muscles"].copy()
            other_muscles = restored.pending["muscles"].copy()
        else:
            muscles, info = brain.observe_then_act(env.sensor_packet(), learning=False)
            other_muscles, other_info = restored.observe_then_act(other_env.sensor_packet(), learning=False)
            equal_tree(info, other_info, state_kind + "/act_info")
        assert np.array_equal(muscles, other_muscles)
        packet = env.step(muscles)
        other_packet = other_env.step(other_muscles)
        equal_tree(packet, other_packet, state_kind + "/packet")
        info = brain.observe_outcome(packet, muscles)
        other_info = restored.observe_outcome(other_packet, other_muscles)
        equal_tree(info, other_info, state_kind + "/outcome_info")
        equal_tree(brain.snapshot(), restored.snapshot(), state_kind + "/brain")
        equal_tree(env.get_view_state(), other_env.get_view_state(), state_kind + "/physics")
        trajectory.append({"frame": frame, "muscles": muscles.tolist(),
                           "position": env.world.position.tolist()})
    return {"kind": state_kind, "continued_physical_frames": 10,
            "added_recorded_frames": brain.runtime.frames_recorded - initial_frame,
            "added_time_steps": int(brain.runtime.clock.当前时间) - initial_clock,
            "full_snapshot_and_physics_equal_after_every_frame": True,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "final_snapshot_sha256": digest_tree(brain.snapshot()), "trajectory": trajectory}


def clone_isolation(brain, packet):
    """Two restored branches must not share writable containers or arrays."""
    source = brain.snapshot()
    source_before = copy.deepcopy(source)
    branch_a = OriginalBrain.from_snapshot(source)
    branch_b = OriginalBrain.from_snapshot(source)
    other_before = branch_b.snapshot()
    branch_a.flags["exploration"] = not branch_a.flags["exploration"]
    branch_a.prepared["muscles"][0] += .125
    branch_a.prepared["info"]["source"] = "audit_only_prepared_change"
    branch_a.last_info["source"] = "audit_only_last_info_change"
    equal_tree(source_before, source, "prepared_clone/source_unchanged")
    equal_tree(other_before, branch_b.snapshot(), "prepared_clone/other_unchanged")
    # Consume the genuine prepared observation to obtain a real proposed action,
    # still unexecuted. This only changes the second branch's cache ownership.
    recorded_before = branch_b.runtime.frames_recorded
    branch_b.observe_then_act(packet, learning=False)
    assert branch_b.runtime.frames_recorded == recorded_before
    assert branch_b.pending is not None and branch_b.prepared is None
    source = branch_b.snapshot()
    source_before = copy.deepcopy(source)
    pending_a = OriginalBrain.from_snapshot(source)
    pending_b = OriginalBrain.from_snapshot(source)
    other_before = pending_b.snapshot()
    pending_a.flags["reflex"] = not pending_a.flags["reflex"]
    pending_a.pending["muscles"][2] = .9
    pending_a.pending["learning"] = True
    pending_a.last_info["source"] = "audit_only_pending_change"
    equal_tree(source_before, source, "pending_clone/source_unchanged")
    equal_tree(other_before, pending_b.snapshot(), "pending_clone/other_unchanged")
    return {"prepared_flags_muscles_info_independent": True,
            "pending_flags_muscles_learning_info_independent": True,
            "source_snapshot_unmodified": True,
            "scope": "Ownership-only mutation of a genuine pending proposal; no mutated action executed or learned."}


def run_audit():
    started = time.perf_counter()
    source_files = ("brain.py", "original_runtime.py", "sensory_adapter.py", "experience_environment.py")
    hashes_before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_files}
    env = ExperienceEnvironment(seed=317)
    packet = env.setup_pairing("red", 777.)
    brain = OriginalBrain()
    original_record = brain.runtime.record
    last_index_input = {}

    def checked_record(*args, **kwargs):
        assert kwargs.get("index_pfc_bool") is not None
        last_index_input["value"] = kwargs["index_pfc_bool"].copy()
        return original_record(*args, **kwargs)

    brain.runtime.record = checked_record
    baseline_connections = copy.deepcopy(brain.runtime.connection_counts())
    frames = []
    invalid_checks = []
    for frame in range(8):
        before = brain.runtime.frames_recorded
        muscles, info = brain.observe_then_act(packet, learning=False)
        expected = 1 if frame == 0 else 0
        assert brain.runtime.frames_recorded - before == expected
        assert brain.pending is not None and brain.prepared is None
        post = env.step(muscles)
        if frame == 0:
            invalid_checks = validate_invalid_outcomes(brain, post, muscles)
        before_outcome = brain.runtime.frames_recorded
        outcome = brain.observe_outcome(post, muscles)
        assert np.array_equal(np.flatnonzero(last_index_input["value"]),
                              outcome["external_pfc_cells"])
        assert brain.runtime.frames_recorded == before_outcome + 1
        assert brain.pending is None and brain.prepared is not None
        assert np.array_equal(outcome["actual_motor_in_memory"], muscles)
        frames.append({"physical_frame": frame, "act_added_memory_frames": expected,
                       "recorded_frames": brain.runtime.frames_recorded,
                       "shared_clock": int(brain.runtime.clock.当前时间),
                       "actual_muscles": muscles.tolist()})
        packet = post
    assert brain.runtime.frames_recorded == 9
    assert brain.runtime.internal_steps_recorded == 18
    assert int(brain.runtime.clock.时间激活.sum()) == 1
    assert brain.adapter.motor_learning_steps == 0
    equal_tree(baseline_connections, brain.runtime.connection_counts())
    brain.runtime.record = original_record
    clone_checks = clone_isolation(brain, env.sensor_packet())
    # Rejected act input must not consume an already prepared real observation.
    invalid_act_checks = []
    for label in ("nan_rgb", "short_pcm", "private_position", "private_target"):
        bad = copy.deepcopy(env.sensor_packet())
        if label == "nan_rgb":
            bad["observation"]["ray_colors"][0, 0] = np.nan
        elif label == "short_pcm":
            bad["waveform"] = np.zeros(7)
        elif label == "private_position":
            bad["observation"]["position"] = np.zeros(2)
        else:
            bad["target"] = "red"
        before = brain.snapshot()
        try:
            brain.observe_then_act(bad, learning=False)
        except (ValueError, TypeError, KeyError):
            pass
        else:
            raise AssertionError("Invalid act input accepted: " + label)
        equal_tree(before, brain.snapshot(), "invalid_act/" + label)
        invalid_act_checks.append(label)
    with tempfile.TemporaryDirectory(prefix="brain_boundary_", dir=HERE / "results") as directory:
        folder = Path(directory)
        prepared = checkpoint_continuation(brain, env, "prepared", folder)
        pending = checkpoint_continuation(brain, env, "pending", folder)
    # Terminal adds the actual final sensory state once on the same lifetime
    # clock, carrying the preceding real execution. No pending future action.
    muscles, _ = brain.observe_then_act(env.sensor_packet(), learning=False)
    packet = env.step(muscles)
    terminal_before = brain.runtime.frames_recorded
    terminal_rng = copy.deepcopy(brain.rng.bit_generator.state)
    terminal_exploration_left = brain.exploration_left
    terminal_exploration_muscles = brain.exploration_muscles.copy()
    terminal_info = brain.observe_outcome(packet, muscles, terminal=True)
    assert brain.runtime.frames_recorded == terminal_before + 1
    assert brain.pending is None and brain.prepared is None
    assert np.array_equal(terminal_info["actual_motor_in_memory"], muscles)
    equal_tree(terminal_rng, brain.rng.bit_generator.state, "terminal_rng")
    assert terminal_exploration_left == brain.exploration_left
    assert np.array_equal(terminal_exploration_muscles, brain.exploration_muscles)
    assert terminal_info["source"] == "terminal_observation"
    expected_motor = brain.adapter.encode(packet, muscles)["motor"]
    assert np.array_equal(brain.runtime.motor_memory.激活, expected_motor)
    assert brain.adapter.motor_learning_steps == 0
    # Current sensor semantics ignore private fields. Diagnose whether cache
    # identity accidentally depends on them, without treating them as legal input.
    legal = env.sensor_packet()
    augmented = copy.deepcopy(legal)
    augmented["observation"]["position"] = np.array([999., -999.])
    augmented["observation"]["target_index"] = 2
    private_fields_accepted = True
    try:
        a = brain.adapter._scalars(legal, muscles)
        b = brain.adapter._scalars(augmented, muscles)
        equal_tree(a, b, "private_sensor_exclusion")
    except (ValueError, KeyError, TypeError):
        private_fields_accepted = False
    try:
        extra_key_changes_digest = packet_digest(legal) != packet_digest(augmented)
    except (ValueError, KeyError, TypeError):
        extra_key_changes_digest = False
    issues = []
    if private_fields_accepted and extra_key_changes_digest:
        issues.append({"kind": "private_metadata_affects_prepared_cache",
            "detail": "Adapter ignores extra observation keys, but packet_digest includes them; adding a private field can force a second processing/time step. Official environment packets contain no such fields."})
    hashes_after = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_files}
    assert hashes_before == hashes_after, "Sources changed during audit; rerun against a frozen version"
    return {"core_checks_passed": True, "source_sha256": hashes_after,
            "default_brain_config": brain.snapshot()["config"],
            "initial_physical_frames": 8, "initial_single_timeline_frames": 9,
            "initial_shared_clock_steps": 18, "timeline_active_cells": 1,
            "frames": frames, "prepared_avoids_duplicate_recording": True,
            "index_receives_actual_same_frame_external_pfc": True,
            "invalid_outcome_rejections_with_entire_snapshot_unchanged": invalid_checks,
            "invalid_act_rejections_with_entire_snapshot_unchanged": invalid_act_checks,
            "learning": False, "motor_learning_steps": 0,
            "checkpoint_continuations": [prepared, pending],
            "two_branches_from_one_snapshot_are_isolated": clone_checks,
            "terminal_real_outcome_recorded_once": True,
            "terminal_memory_uses_actual_execution": True,
            "terminal_has_no_pending_or_prepared_action": True,
            "terminal_does_not_consume_exploration_rng_or_sequence": True,
            "adapter_alone_ignores_extra_keys": private_fields_accepted,
            "brain_entry_rejects_private_fields_before_any_state_change": True,
            "extra_private_keys_change_packet_digest": extra_key_changes_digest,
            "issues": issues, "elapsed_seconds": time.perf_counter() - started,
            "operational_limits": [
                "Changing flags after an observation is already prepared does not retroactively change its cached next action; set ablations before processing that observation.",
                "Frozen evaluation advances the single timeline and current modality activity but does not permanently learn new memory weights."],
            "scope": "Short frozen-plasticity interface audit, not learned recall quality or navigation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "brain_boundary_audit.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run_audit()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("core_checks_passed", "issues", "elapsed_seconds")},
                     ensure_ascii=False, indent=2))
