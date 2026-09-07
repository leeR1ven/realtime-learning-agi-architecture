"""Independent bounded integrity checks for the new route-switch experiment.

These checks test interfaces, deterministic cloning, retained auditory state and
checkpoint continuation. They do not train a navigator or establish successful
route selection. Physical success belongs to the fixed, paired runner trials.
No teacher/planner is imported and no artificial reward is supplied here.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from environment import SENSOR_KEYS
from route_switch_environment import RouteSwitchEnv, ROUTE_NAMES
from route_switch_controller import RouteSwitchController


HERE = Path(__file__).resolve().parent
EVALUATION_SEEDS = (831, 931, 1031)


def digest(value):
    """Canonical value hash, including random generators and nested objects."""
    h = hashlib.sha256()

    def add(item):
        h.update(type(item).__qualname__.encode() + b":")
        if isinstance(item, np.ndarray):
            h.update(item.dtype.str.encode())
            h.update(repr(item.shape).encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            add(item.item())
        elif isinstance(item, np.random.Generator):
            add(item.bit_generator.state)
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__qualname__, repr(x))):
                add(key)
                add(item[key])
        elif isinstance(item, (set, frozenset)):
            for child in sorted(digest(child) for child in item):
                h.update(child.encode())
        elif isinstance(item, (list, tuple, deque)):
            if isinstance(item, deque):
                add(item.maxlen)
            for child in item:
                add(child)
        elif hasattr(item, "__dict__"):
            add(vars(item))
        elif item is None or isinstance(item, (str, bytes, int, float, bool)):
            h.update(repr(item).encode())
        else:
            raise TypeError(f"Unsupported state type: {type(item)!r}")
        h.update(b";")

    add(value)
    return h.hexdigest()


def check_packet(packet):
    assert set(packet) == {"observation", "waveform"}
    assert set(packet["observation"]) == SENSOR_KEYS
    assert packet["waveform"].ndim == 1
    assert all(np.isfinite(np.asarray(v)).all() for v in packet["observation"].values())
    assert np.isfinite(packet["waveform"]).all()


def policy_step(env, brain, *, learning=False):
    packet = env.sensor_packet()
    check_packet(packet)
    muscles, info = brain.observe_then_act(packet, learning=learning)
    assert info["teaching"] is False
    assert np.array_equal(muscles, np.asarray(info["predicted_muscles"]))
    env.step(muscles, .1)
    return muscles, info


def interface_checks():
    env = RouteSwitchEnv(seed=831)
    packet = env.reset(target="red", route="bottom", start="left_middle")
    check_packet(packet)
    pristine = RouteSwitchController(seed=2026, mixed_neurons=512, max_events=1024)
    a, b = copy.deepcopy(pristine), copy.deepcopy(pristine)
    # The real interface omits all these values. An additional negative test
    # confirms neither packet extras nor observation extras steer the policy.
    poisoned = copy.deepcopy(packet)
    poisoned.update(target="blue", route="top", position=[99., 99.],
                    goal_position=[-100., 0.], trial_id=999,
                    teacher_muscles=[1., 1., 1., 1.])
    poisoned["observation"].update(target="green", route="top", heading=99.)
    ma, ia = a.observe_then_act(packet, learning=False)
    mb, ib = b.observe_then_act(poisoned, learning=False)
    assert np.array_equal(ma, mb) and digest(a) == digest(b)
    assert ia["goal_active"] == [0] and ia["rule_active"] == [1]
    assert np.count_nonzero(a.event_ids >= 0) == 0

    # Present a real second sound while preserving the same body and goal.
    before_rng = digest(env.rng)
    for _ in range(6):
        env.step(np.zeros(4), .1)
    a.observe_then_act(env.sensor_packet(), learning=False)
    before_world = digest(env.world)
    env.switch_route("top")
    assert digest(env.world) == before_world and digest(env.rng) == before_rng
    _, switched = a.observe_then_act(env.sensor_packet(), learning=False)
    assert switched["goal_active"] == [0] and switched["rule_active"] == [2]
    silent = copy.deepcopy(env.sensor_packet())
    silent["waveform"][:] = 0.
    for _ in range(200):
        _, info = a.observe_then_act(silent, learning=False)
        assert info["goal_active"] == [0] and info["rule_active"] == [2]
    goal = a.context.copy()
    a.intervene_rule_state([1., 0., 0.])
    assert np.array_equal(a.context, goal)
    _, info = a.observe_then_act(silent, learning=False)
    assert info["rule_active"] == [0] and info["goal_active"] == [0]
    return dict(packet_exact_sensor_whitelist=True,
                omitted_private_answers_cannot_steer_policy=True,
                route_sound_keeps_red_goal_population=True,
                silent_rule_population_retention_frames=200,
                rule_only_intervention_preserves_goal=True,
                artificial_reward_calls=0, teacher_calls=0,
                limitation="State retention is a unit check, not evidence of behavioral route-memory dependence.")


def prefix_clone_checks():
    rows = []
    for seed in EVALUATION_SEEDS:
        for route in ROUTE_NAMES:
            env = RouteSwitchEnv(seed=seed)
            env.reset(target="red", route=route, start="left_middle")
            brain = RouteSwitchController(seed=2026, mixed_neurons=512, max_events=1024)
            brain.rng = np.random.default_rng(seed)
            trace = []
            for _ in range(100):
                if env.private_metrics()["terminated"]:
                    break
                muscles, _ = policy_step(env, brain, learning=True)
                trace.append(digest((env.world, muscles)))
            metrics = env.private_metrics()
            invalid = []
            if len(trace) != 100 or metrics["target_reached"]:
                invalid.append("first_red_arrival_before_fixed_split")
            if any(p["direction"] == "left_to_right" for p in metrics["passages"]):
                invalid.append("completed_forward_passage_before_fixed_split")
            if any(p["direction"] == "left_to_right" for p in metrics["midline_crossings"]):
                invalid.append("forward_midline_already_crossed")
            state_hash = digest((env, brain))
            clone_env, clone_brain = copy.deepcopy(env), copy.deepcopy(brain)
            assert digest((clone_env, clone_brain)) == state_hash
            assert not np.shares_memory(clone_brain.event_muscles, brain.event_muscles)
            assert not np.shares_memory(clone_env.world.position, env.world.position)
            clone_rng = digest((env.rng, brain.rng))
            if not metrics["terminated"]:
                opposite = ROUTE_NAMES[1 - ROUTE_NAMES.index(route)]
                env.switch_route(route)
                clone_env.switch_route(opposite)
                assert digest(env.world) == digest(clone_env.world)
                assert digest(brain) == digest(clone_brain)
                assert digest((env.rng, brain.rng)) == clone_rng
                pa, pb = env.sensor_packet(), clone_env.sensor_packet()
                assert digest(pa["observation"]) == digest(pb["observation"])
                assert not np.array_equal(pa["waveform"], pb["waveform"])
                assert env.private_metrics()["target"] == clone_env.private_metrics()["target"] == "red"
                # An untouched duplicate under the same cue has exactly the
                # same subsequent stochastic actions and physical trajectory.
                replay_env, replay_brain = copy.deepcopy(env), copy.deepcopy(brain)
                continuation_steps = 0
                for _ in range(25):
                    if env.private_metrics()["terminated"]:
                        break
                    x, ix = policy_step(env, brain)
                    y, iy = policy_step(replay_env, replay_brain)
                    assert np.array_equal(x, y)
                    assert digest(env.world) == digest(replay_env.world)
                    assert digest(ix) == digest(iy)
                    continuation_steps += 1
            else:
                continuation_steps = 0
            rows.append(dict(seed=seed, initial_route=route, prefix_frames=len(trace),
                prefix_trajectory_sha256=digest(trace), cloned_complete_state_sha256=state_hash,
                exact_clone=True, same_cue_continuation_steps=continuation_steps,
                invalid_reasons=invalid))
    return dict(model="newborn 512-neuron audit fixture; not the trained model",
                split_frames=100, dt=.1, seeds=list(EVALUATION_SEEDS), rows=rows)


def checkpoint_check():
    env = RouteSwitchEnv(seed=19)
    env.reset(route="bottom")
    brain = RouteSwitchController(seed=29, mixed_neurons=512, max_events=256)
    for _ in range(15):
        policy_step(env, brain, learning=True)
    env.switch_route("top")
    for _ in range(10):
        policy_step(env, brain, learning=True)
    with tempfile.TemporaryDirectory(prefix="route-integrity-") as folder:
        path = Path(folder) / "audit_only.npz"
        brain.save(path)
        resumed = RouteSwitchController.load(path)
        clone_env = copy.deepcopy(env)
        assert digest(brain.rng) == digest(resumed.rng)
        for _ in range(25):
            a, ia = policy_step(env, brain, learning=True)
            b, ib = policy_step(clone_env, resumed, learning=True)
            assert np.array_equal(a, b) and digest(ia) == digest(ib)
            assert digest(env.world) == digest(clone_env.world)
        for key in ("event_ids", "event_previous", "event_next", "event_context",
                    "event_strength", "event_muscles", "rule_context", "context"):
            assert np.array_equal(getattr(brain, key), getattr(resumed, key))
        assert np.count_nonzero(brain.event_kind == 2) == 0
    return dict(exact_online_continuation_frames=25, sparse_time_links_equal=True,
                teacher_events=0, test_archive_temporary_and_removed=True)


def checkpoint_inventory(path):
    """Only observe an existing trained archive; do not update or resave it."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes().decode("utf-8"))
        occupied = data["event_ids"] >= 0
        confirmed = occupied & (data["event_strength"] > 0)
        teacher_count = int(np.count_nonzero(occupied & (data["event_kind"] == 2)))
        signatures = {str(int(signature)): int(np.count_nonzero(confirmed & (data["event_context"] == signature)))
                      for signature in np.unique(data["event_context"][confirmed])}
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    birth_hash=metadata["birth_hash"], stored_events=int(occupied.sum()),
                    confirmed_events=int(confirmed.sum()), teacher_events=teacher_count,
                    confirmed_context_signature_counts=signatures,
                    no_teacher_events=teacher_count == 0,
                    limitation="Audio-context counts alone do not prove both physical passages were learned; the episode trajectories and reward logs must also agree.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "route_switch_integrity.json")
    args = parser.parse_args()
    start = time.perf_counter()
    result = dict(schema_version=1,
                  scope="Independent integrity units, not navigation or AGI acceptance",
                  interface=interface_checks(), prefix_cloning=prefix_clone_checks(),
                  checkpoint_continuation=checkpoint_check())
    if args.model:
        result["trained_checkpoint_inventory"] = checkpoint_inventory(args.model)
    result["integrity_checks_passed"] = result.get("trained_checkpoint_inventory", {}).get("no_teacher_events", True)
    result["source_sha256"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("audit_route_switch.py", "route_switch_controller.py", "route_switch_environment.py")}
    result["wall_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), wall_seconds=result["wall_seconds"],
                         integrity_checks_passed=result["integrity_checks_passed"]), ensure_ascii=False))
    if not result["integrity_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
