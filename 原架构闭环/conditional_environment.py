"""Experiment-side delayed conditional choice in a real two-branch arena.

Only ``sensor_packet()`` and the FIRST result of ``step`` may enter the brain.
The two context layouts swap ordinary red/blue circles. Two arbitrary tones
precede the layout by default. Both passages stay physically open in every
condition; a private XOR truth table scores the first actual passage crossing.
This is a small delayed conditional-choice test, not a navigation policy or a
claim that a model has acquired abstract logic. No teacher action is generated.

API::
    env = ConditionalEnvironment()
    packet = env.reset(cue=0, context=1)
    packet, reward, terminated, private_info = env.step(muscles, dt=.1)

``reward`` is external experimental feedback. It is NOT a sensor field and is
not automatically connected to any brain. ``rule_flip`` is a run-level reversal
of the reward mapping: it must not be silently randomized within an otherwise
indistinguishable learning task. ``balanced_conditions`` balances all cue/context
combinations independently at every supplied start, using one fixed rule_flip.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experience_environment import (ExperienceEnvironment, BeaconNavigationEnv, Scene,
                                    SAMPLE_RATE, SENSOR_KEYS)


DEFAULT_START = (1.35, 3.0, 0.0)
STARTS = (DEFAULT_START, (1.35, 2.8, 0.0), (1.35, 3.2, 0.0))
ARENA_SCENE = Scene(
    "conditional_fork", 6.0, 6.0, ((2.8, 2.55, 4.2, 3.45),),
    ((4.8, 4.4), (5.4, 3.0), (4.8, 1.6)),
    {"middle": DEFAULT_START, "lower_offset": STARTS[1], "upper_offset": STARTS[2]},
)
GATE_X = 3.5  # Inside the solid divider: no middle bypass can be a response.
COLORS = {"red": (1.0, .04, .04), "blue": (.04, .04, 1.0)}
POSITIONS = {"upper": (4.8, 4.4), "lower": (4.8, 1.6)}


def _bit(value, name):
    if isinstance(value, (bool, np.bool_)) or value not in (0, 1):
        raise ValueError(f"{name} must be integer 0 or 1")
    return int(value)


def balanced_conditions(repeats=1, *, seed=0, starts=STARTS, rule_flip=0):
    """Evaluator-only randomized full crossing, with no answer-dependent starts."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rule_flip = _bit(rule_flip, "rule_flip")
    starts = [tuple(map(float, p)) for p in starts]
    if not starts or any(len(p) != 3 or not np.isfinite(p).all() for p in starts):
        raise ValueError("starts must contain finite (x,y,heading) poses")
    rows = [{"cue": cue, "context": context, "rule_flip": rule_flip,
             "start": pose, "repeat": repeat}
            for repeat in range(repeats) for pose in starts
            for cue in (0, 1) for context in (0, 1)]
    order = np.random.default_rng(seed).permutation(len(rows))
    return [rows[int(i)] for i in order]


class _PhysicalArena(ExperienceEnvironment):
    """Keep the audited physics loop; observe its existing <=10 ms samples."""

    def __init__(self, **kwargs):
        self._choice_owner = None
        super().__init__(scenario=ARENA_SCENE, **kwargs)

    def sensor_packet(self, dt=None):
        packet = super().sensor_packet(dt=dt)
        if self._choice_owner is not None:
            packet["waveform"] = self._choice_owner._waveform(len(packet["waveform"]))
        else:
            packet["waveform"][:] = 0
        return packet

    def _score_arrivals(self):
        super()._score_arrivals()
        if self._choice_owner is not None:
            self._choice_owner._physics_sample()

    def step(self, muscles, dt=.1):
        # The wrapper logs its own scheduled PCM; the generic Experience logger
        # assumes its own immediate-tone scheduler, which is unused here.
        return BeaconNavigationEnv.step(self, muscles, dt=dt)


class ConditionalEnvironment:
    """Real four-muscle choice, with all task labels confined to the evaluator.

    Default: cue [0,.6) seconds, context RGB from 1.0 seconds, 12 s timeout.
    cue_delay/context_delay allow simultaneous-stimulus diagnostic controls.
    The body is always free to act. A response before both cue onset and context
    onset terminates as premature (reward zero); no hidden gate restrains it.
    Objects are ordinary non-solid colored circles; only the divider is solid.
    """

    def __init__(self, *, seed=20260906, ray_count=31, frequencies=(220., 440.),
                 cue_delay=0., cue_seconds=.6, context_delay=1., amplitude=.45,
                 max_seconds=12.):
        self.seed = int(seed)
        self.ray_count = ray_count
        self.frequencies = tuple(map(float, frequencies))
        if (len(self.frequencies) != 2 or len(set(self.frequencies)) != 2 or
                not all(np.isfinite(f) and 0 < f < SAMPLE_RATE / 2 for f in self.frequencies)):
            raise ValueError("two distinct finite frequencies below Nyquist are required")
        self.cue_delay, self.cue_seconds, self.context_delay, self.amplitude, self.max_seconds = (
            map(float, (cue_delay, cue_seconds, context_delay, amplitude, max_seconds)))
        if not all(np.isfinite(v) for v in (self.cue_delay, self.cue_seconds,
                                           self.context_delay, self.amplitude, self.max_seconds)):
            raise ValueError("timing and amplitude must be finite")
        if (min(self.cue_delay, self.context_delay) < 0 or self.cue_seconds <= 0 or
                not 0 < self.amplitude <= 1 or self.max_seconds <= 0 or
                max(self.context_delay, self.cue_delay + self.cue_seconds) >= self.max_seconds):
            raise ValueError("invalid cue/context timing, amplitude or timeout")
        self.reset()

    @property
    def world(self):
        """Rich physical state for evaluators/viewers; never pass it to a brain."""
        return self._arena.world

    def reset(self, cue=0, context=0, *, start=DEFAULT_START, rule_flip=0):
        cue, context, rule_flip = (_bit(cue, "cue"), _bit(context, "context"),
                                    _bit(rule_flip, "rule_flip"))
        if isinstance(start, str):
            if start not in ARENA_SCENE.starts:
                raise ValueError("unknown start")
            start = ARENA_SCENE.starts[start]
        pose = np.asarray(start, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or not pose[0] < 2.8 - .18:
            raise ValueError("start must be a finite pose entirely left of the fork")
        # Same physical RNG and geometry for every cue/context/rule condition.
        arena = _PhysicalArena(seed=self.seed, ray_count=self.ray_count, beacon_radius=.4)
        arena.reset(start=pose, target="red")
        arena.hide_beacons()
        self._arena = arena
        self._cue, self._context, self._rule_flip = cue, context, rule_flip
        self._expected_choice = ("upper", "lower")[cue ^ context ^ rule_flip]
        self._terminated = False
        self._response = None
        self._total_reward = 0.
        self._reward_delivered = False
        self._context_presented_at = None
        self._samples = []
        self._previous_position = self.world.position.copy()
        self._previous_time = float(self.world.time)
        self._arena._choice_owner = self
        if self.context_delay == 0:
            self._present_context()
        return self.sensor_packet()

    def _waveform(self, count):
        # Deterministic mono PCM; no context, response or correct-choice input.
        times = self.world.time + np.arange(count) / SAMPLE_RATE
        active = ((times >= self.cue_delay - 1e-10) &
                  (times < self.cue_delay + self.cue_seconds - 1e-10))
        waveform = np.zeros(count, dtype=float)
        waveform[active] = self.amplitude * np.sin(
            2 * math.pi * self.frequencies[self._cue] * (times[active] - self.cue_delay))
        return waveform

    def _present_context(self):
        # Only the independently assigned visual condition changes the colors.
        upper_name, lower_name = (("red", "blue") if self._context == 0 else ("blue", "red"))
        self.world.set_geometry(objects=[
            {"name": name, "position": POSITIONS[side], "radius": .4, "color": COLORS[name]}
            for side, name in (("upper", upper_name), ("lower", lower_name))])
        self._arena._inside_visible = self._arena._visible_inside()
        self._context_presented_at = float(self.world.time)

    def _physics_sample(self):
        t, p = float(self.world.time), self.world.position.copy()
        if self._context_presented_at is None and t >= self.context_delay - 1e-10:
            self._present_context()
        previous = self._previous_position
        if self._response is None and previous[0] < GATE_X <= p[0]:
            fraction = float((GATE_X - previous[0]) / (p[0] - previous[0]))
            at = previous + fraction * (p - previous)
            crossing_time = self._previous_time + fraction * (t - self._previous_time)
            radius = self.world.config.radius
            choice = ("upper" if at[1] >= 3.45 + radius - 1e-8 else
                      "lower" if at[1] <= 2.55 - radius + 1e-8 else "invalid_middle")
            premature = crossing_time < max(self.context_delay, self.cue_delay) - 1e-10
            timely = crossing_time <= self.max_seconds + 1e-10
            correct = choice == self._expected_choice and not premature and timely
            self._response = {"choice": choice, "time": crossing_time, "position": at.tolist(),
                              "premature": premature, "within_budget": timely, "correct": correct}
            self._total_reward = float(correct)
            self._terminated = True
        if t >= self.max_seconds - 1e-10:
            self._terminated = True
        self._previous_position, self._previous_time = p, t

    def sensor_packet(self, dt=None):
        """Only this unlabelled packet is a legitimate brain input."""
        return self._arena.sensor_packet(dt=dt)

    def step(self, muscles, dt=.1):
        """Execute actual muscles; return packet and separate terminal feedback.

        Terminal calls are rejected, preventing repeated reward or a second
        choice after an incorrect first response. The remainder of the current
        <=1 s control interval still follows the same applied muscle forces.
        Use dt=.1 for the standard protocol; crossings are measured every .01 s.
        """
        if self._terminated:
            raise RuntimeError("trial is terminal; reset before applying another action")
        dt = float(dt)
        if not np.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("dt must be finite and in (0,1]")
        action = np.asarray(muscles, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any((action < 0) | (action > 1)):
            raise ValueError("muscles must contain four finite activations in [0,1]")
        before = self.sensor_packet(dt=dt)
        visible = self._arena.visible_beacon_rays()
        start_time = float(self.world.time)
        remaining = self.max_seconds - start_time
        actual_dt = min(dt, remaining)
        packet = self._arena.step(action, dt=actual_dt)
        self._samples.append({"time": start_time, "dt": actual_dt,
                              "sound_rms": float(np.sqrt(np.mean(before["waveform"] ** 2))),
                              "visible_rays": visible, "applied_muscles": action.tolist(),
                              "position_after": self.world.position.tolist()})
        reward = self._total_reward if self._terminated and not self._reward_delivered else 0.
        if self._terminated:
            self._reward_delivered = True
        return packet, reward, self._terminated, self.private_metrics()

    def private_metrics(self):
        """Scoring and audit information; never merge any of it into packet."""
        return {"task": "delayed_conditional_choice", "cue": self._cue, "context": self._context,
                "rule_flip": self._rule_flip, "expected_choice": self._expected_choice,
                "frequency_hz": self.frequencies[self._cue], "cue_delay": self.cue_delay,
                "cue_seconds": self.cue_seconds, "context_delay": self.context_delay,
                "context_presented_at": self._context_presented_at,
                "max_seconds": self.max_seconds, "elapsed_seconds": float(self.world.time),
                "initial_pose": self._arena._initial_pose.tolist(),
                "final_position": self.world.position.tolist(), "gate_x": GATE_X,
                "response": copy.deepcopy(self._response), "terminated": self._terminated,
                "timeout": self._terminated and self._response is None,
                "total_reward": self._total_reward, "path_length": self._arena._path_length,
                "contact_steps": self._arena._contact_steps, "teacher_actions": 0,
                "control_steps": self._arena._steps, "samples": copy.deepcopy(self._samples)}

    def get_view_state(self):
        state = self.world.get_state()
        state.update({"private_metrics": self.private_metrics(),
                      "choice_line": {"x": GATE_X, "upper_min_y": 3.45 + self.world.config.radius,
                                      "lower_max_y": 2.55 - self.world.config.radius}})
        return state


def _equal_packets(a, b):
    return (set(a) == set(b) == {"observation", "waveform"} and
            set(a["observation"]) == set(b["observation"]) == SENSOR_KEYS and
            all(np.array_equal(a["observation"][key], b["observation"][key]) for key in SENSOR_KEYS) and
            np.array_equal(a["waveform"], b["waveform"]))


def self_check():
    """No brain or training. Fixed action sequences are physics test fixtures."""
    results = {"scope": "environment only; scripted fixture actions are not agent success"}
    env = ConditionalEnvironment(seed=71)
    initial = env.sensor_packet()
    assert _equal_packets(initial, env.sensor_packet())
    assert initial["observation"]["ray_colors"].shape == (31, 3)
    assert initial["observation"]["touch"].shape == (4,)
    timeline = []
    for frame in range(13):
        packet = env.sensor_packet()
        rays = env._arena.visible_beacon_rays()
        timeline.append({"frame": frame, "time": float(env.world.time),
                         "audio_rms": float(np.sqrt(np.mean(packet["waveform"] ** 2))),
                         "red_rays": rays["red"], "blue_rays": rays["blue"]})
        assert bool(np.any(packet["waveform"])) == (frame < 6)
        assert bool(rays["red"]) == (frame >= 10)
        assert bool(rays["blue"]) == (frame >= 10)
        env.step(np.zeros(4))
    results["default_actual_audio_visual_timing"] = timeline
    results["white_list_pure_read_and_actual_two_markers_visible"] = True

    # Same cue: before reveal, contexts are physically and sensorially identical.
    a, b = ConditionalEnvironment(), ConditionalEnvironment()
    a.reset(cue=1, context=0)
    b.reset(cue=1, context=1)
    for _ in range(9):
        assert _equal_packets(a.sensor_packet(), b.sensor_packet())
        a.step(np.zeros(4)); b.step(np.zeros(4))
    a.step(np.zeros(4)); b.step(np.zeros(4))
    assert np.array_equal(a.sensor_packet()["observation"]["ray_distances"],
                          b.sensor_packet()["observation"]["ray_distances"])
    assert not np.array_equal(a.sensor_packet()["observation"]["ray_colors"],
                              b.sensor_packet()["observation"]["ray_colors"])
    results["context_only_changes_actual_object_RGB_at_fixed_reveal"] = True

    # Counterbalance: each cue alone and each context alone has equal answers.
    schedule = balanced_conditions(2, seed=39)
    counts = {}
    for row in schedule:
        side = ("upper", "lower")[row["cue"] ^ row["context"] ^ row["rule_flip"]]
        for variable in ("cue", "context"):
            key = f"{row['start']}:{variable}={row[variable]}"
            counts.setdefault(key, {"upper": 0, "lower": 0})[side] += 1
    assert all(c["upper"] == c["lower"] == 2 for c in counts.values())
    results["counterbalance_counts"] = counts

    # Actual physics: fixed left/right spin followed by forward forces. These
    # are identical fixtures for all visual/audio conditions, not generated
    # from an answer or exposed to a brain as demonstrations.
    choices = []
    for side in ("upper", "lower"):
        spin = np.array([0., .25, .25, 0.]) if side == "upper" else np.array([.25, 0., 0., .25])
        actions = [np.zeros(4)] * 10 + [spin] * 6 + [np.zeros(4)] * 7 + [np.array([.6, 0., .6, 0.])] * 80
        for cue in (0, 1):
            for context in (0, 1):
                pair = [ConditionalEnvironment(seed=15) for _ in range(2)]
                for flip, e in enumerate(pair):
                    e.reset(cue=cue, context=context, rule_flip=flip)
                reward_pair = []
                for action in actions:
                    outputs = [e.step(action) for e in pair]
                    assert _equal_packets(outputs[0][0], outputs[1][0])
                    assert np.array_equal(pair[0].world.position, pair[1].world.position)
                    assert outputs[0][2] == outputs[1][2]
                    if outputs[0][2]:
                        reward_pair = [o[1] for o in outputs]
                        break
                response = pair[0].private_metrics()["response"]
                assert response is not None and response["choice"] == side, response
                assert sorted(reward_pair) == [0., 1.]
                choices.append({"cue": cue, "context": context, "fixture": side,
                                "response": response, "rule_flip_rewards": reward_pair,
                                "contact_steps": pair[0]._arena._contact_steps})
                try:
                    pair[0].step(np.zeros(4))
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("terminal trial accepted another action")
    results["physical_both_choices_all_conditions_rule_flip_only_reward"] = choices

    # Straight driving must hit the solid divider, never count middle crossing.
    straight = ConditionalEnvironment(max_seconds=3.)
    while not straight._terminated:
        _, reward, _, _ = straight.step([1., 0., 1., 0.])
        assert reward == 0.
    assert straight._response is None and straight._arena._contact_steps > 0
    assert straight.world.position[0] <= 2.8 - straight.world.config.radius + 1e-8
    results["straight_collision_no_middle_shortcut"] = {
        "contacts": straight._arena._contact_steps, "position": straight.world.position.tolist()}

    # Simultaneous and delayed-cue controls preserve the same ordinary interface.
    overlap = ConditionalEnvironment(context_delay=0., cue_delay=0.)
    assert np.any(overlap.sensor_packet()["waveform"])
    assert all(overlap._arena.visible_beacon_rays()[n] for n in ("red", "blue"))
    delayed = ConditionalEnvironment(cue_delay=.4, context_delay=0.)
    assert not np.any(delayed.sensor_packet()["waveform"])
    for _ in range(4):
        delayed.step(np.zeros(4))
    wave = delayed.sensor_packet()["waveform"]
    freqs = np.fft.rfftfreq(len(wave), 1 / SAMPLE_RATE)
    assert abs(freqs[np.abs(np.fft.rfft(wave)).argmax()] - 220.) < 1e-8
    results["optional_simultaneous_and_delayed_cue_controls"] = True
    results["no_brain_run_or_training"] = True
    results["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "conditional_environment_audit.json")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("use --self-check; this module contains no model or policy runner")
    report = self_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output),
                      "source_sha256": report["source_sha256"]}, ensure_ascii=False))
