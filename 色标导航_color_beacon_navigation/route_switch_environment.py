"""Private route scoring around the frozen navigation world's central barrier.

Only ``sensor_packet()`` reaches a controller: the original six observation
fields plus mono PCM. Destination red stays unchanged. Route words, geometry,
trajectory, success and request times are evaluator/UI data only. There is no
teacher, planner, navigation policy, automatic reward or automatic switch here.

An experiment must stop at the FIRST entry into red, including a wrong-route
entry. A valid passage traverses the ENTIRE barrier width with body clearance;
touching or crossing its midline and returning to the same side does not count.
After a changed instruction, the qualifying midline crossing must happen after
that instruction. Repeating the same cue does not invalidate an earlier passage.

For paired intervention trials, deepcopy BOTH environment and controller before
the fixed switch time (DEFAULT_SWITCH_TIME = 10 s), then emit a different cue.
Ten seconds is a schedule, not a promise the freely exploring body is still on
the approach side: private request metadata explicitly records that limitation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from environment import BeaconNavigationEnv, SAMPLE_RATE, SENSOR_KEYS


ROUTE_NAMES = ("bottom", "top")
ROUTE_FREQUENCIES = (990.0, 1320.0)
DEFAULT_SWITCH_TIME = 10.0


class RouteSwitchEnv(BeaconNavigationEnv):
    """Same four-muscle physics, with a private route requirement.

    ``reset(target='red', route='bottom', start='left_middle')`` starts a trial.
    Route may be None to withhold a route instruction until a later cue.
    ``switch_route(route)`` / ``emit_route_cue(route)`` changes only PCM and
    evaluator bookkeeping. Route frequencies may be swapped across training
    runs; they have no built-in direction meaning in the controller.
    """

    def __init__(self, *, seed=20260906, ray_count=31, route_tone_map=None,
                 route_tone_permutation=None, route_cue_seconds=.6,
                 route_cue_amplitude=.45, **kwargs):
        self._route_ready = False
        self._route_tone_map = dict(zip(ROUTE_NAMES, ROUTE_FREQUENCIES))
        if route_tone_map is not None and route_tone_permutation is not None:
            raise ValueError("Specify route_tone_map OR route_tone_permutation")
        if route_tone_map is not None:
            self.set_route_tone_map(route_tone_map)
        if route_tone_permutation is not None:
            self.set_route_tone_permutation(route_tone_permutation)
        if not np.isfinite(route_cue_seconds) or route_cue_seconds <= 0:
            raise ValueError("route_cue_seconds must be positive")
        if not np.isfinite(route_cue_amplitude) or not 0 < route_cue_amplitude <= 1:
            raise ValueError("route_cue_amplitude must be in (0,1]")
        kwargs.setdefault("cue_amplitude", .45)
        if route_cue_amplitude + kwargs["cue_amplitude"] > 1 + 1e-12:
            raise ValueError("Combined target and route amplitude must be <=1")
        if kwargs.pop("scenario", "barrier") != "barrier":
            raise ValueError("This evaluator is defined only for barrier")
        self.route_cue_seconds = float(route_cue_seconds)
        self.route_cue_amplitude = float(route_cue_amplitude)
        super().__init__(scenario="barrier", seed=seed, ray_count=ray_count, **kwargs)
        if set(self._route_tone_map.values()) & set(self._tone_map.values()):
            raise ValueError("Route tones must differ from all target tones")

    def set_route_tone_map(self, mapping):
        if not isinstance(mapping, Mapping):
            mapping = tuple(mapping)
            if len(mapping) != 2:
                raise ValueError("route_tone_map sequence must have exactly two frequencies")
        values = dict(mapping) if isinstance(mapping, Mapping) else dict(zip(ROUTE_NAMES, mapping))
        if set(values) != set(ROUTE_NAMES):
            raise ValueError("route_tone_map must contain bottom and top")
        values = {name: float(values[name]) for name in ROUTE_NAMES}
        if (len(set(values.values())) != 2 or not all(
                np.isfinite(f) and 50 <= f < SAMPLE_RATE / 2 for f in values.values())):
            raise ValueError("Two distinct finite frequencies below Nyquist are required")
        if hasattr(self, "_tone_map") and set(values.values()) & set(self._tone_map.values()):
            raise ValueError("Route tones must differ from all target tones")
        self._route_tone_map = values

    def set_route_tone_permutation(self, permutation):
        permutation = tuple(permutation)
        if sorted(permutation) != [0, 1]:
            raise ValueError("route_tone_permutation must be (0,1) or (1,0)")
        self.set_route_tone_map({name: ROUTE_FREQUENCIES[i]
                                for name, i in zip(ROUTE_NAMES, permutation)})

    def reset(self, scenario=None, start="left_middle", target="red", *, route="bottom",
              route_tone_map=None, route_tone_permutation=None, **kwargs):
        if scenario not in (None, "barrier"):
            raise ValueError("Route scoring requires the unchanged barrier scene")
        if target not in ("red", 0):
            raise ValueError("This task keeps the destination red")
        if route is not None and route not in ROUTE_NAMES:
            raise ValueError("route must be bottom, top or None")
        if route_tone_map is not None and route_tone_permutation is not None:
            raise ValueError("Specify route_tone_map OR route_tone_permutation")
        if route_tone_map is not None:
            self.set_route_tone_map(route_tone_map)
        if route_tone_permutation is not None:
            self.set_route_tone_permutation(route_tone_permutation)
        self._route_ready = False
        super().reset(scenario="barrier", start=start, target="red", **kwargs)
        if set(self._route_tone_map.values()) & set(self._tone_map.values()):
            raise ValueError("Route tones must differ from all target tones")
        x0, y0, x1, y1 = self.scene.walls[0]
        r = self.world.config.radius
        self._gate = {"mid_x": (x0+x1)/2, "left_x": x0-r, "right_x": x1+r,
                      "bottom_max_y": y0-r, "top_min_y": y1+r}
        self._requested_route = None
        self._requirement_since = None
        self._route_cue_start = -math.inf
        # Fixed phase avoids an extra RNG draw changing the base environment.
        self._route_phase = 0.0
        self._route_requests = []
        self._passages = []
        self._midline_crossings = []
        self._arrival_route_score = None
        self._terminal = False
        self._route_previous_position = self.world.position.copy()
        self._route_previous_time = float(self.world.time)
        self._origin_side = self._clear_side(self.world.position)
        self._pending_crossing = None
        self._route_trajectory = []
        self._route_ready = True
        self._append_trajectory()
        if route is not None:
            self._request_route(route)
        # Starting inside red is a terminal failed task: no passage was taken.
        if self._first_target_time is not None:
            self._score_first_target()
        return self.sensor_packet()

    def _clear_side(self, position):
        if position[0] <= self._gate["left_x"] + 1e-10:
            return "left"
        if position[0] >= self._gate["right_x"] - 1e-10:
            return "right"
        return None

    def _append_trajectory(self):
        self._route_trajectory.append({"time": float(self.world.time),
            "position": self.world.position.tolist(), "heading": float(self.world.heading),
            "request_index": len(self._route_requests)-1})

    def _request_route(self, route):
        changed = route != self._requested_route
        if changed:
            self._requirement_since = float(self.world.time)
        self._requested_route = route
        self._route_cue_start = float(self.world.time)
        before = [p for p in self._passages if p["direction"] == "left_to_right"]
        self._route_requests.append({"request_index": len(self._route_requests),
            "route": route, "frequency_hz": self._route_tone_map[route],
            "time": float(self.world.time), "changed": changed,
            "position": self.world.position.tolist(), "heading": float(self.world.heading),
            "trajectory_index": len(self._route_trajectory)-1,
            "completed_passages_before": len(self._passages),
            "already_passed_left_to_right": bool(before),
            "last_passage_before": copy.deepcopy(self._passages[-1]) if self._passages else None,
            "still_on_left_approach": self._clear_side(self.world.position) == "left",
            "inside_barrier_width": self._clear_side(self.world.position) is None,
            "requirement_since": self._requirement_since})

    def switch_route(self, route):
        """Environment/UI-only. No change to world, muscles, target or RNG."""
        if route not in ROUTE_NAMES:
            raise ValueError("route must be bottom or top")
        if self._terminal:
            raise RuntimeError("Trial ended at first red arrival; reset before a new instruction")
        self._request_route(route)
        return self.sensor_packet()

    emit_route_cue = switch_route

    def _audit_passages(self):
        p0, p1 = self._route_previous_position, self.world.position
        t0, t1 = self._route_previous_time, float(self.world.time)
        mid = self._gate["mid_x"]
        direction = ("left_to_right" if p0[0] < mid <= p1[0] else
                     "right_to_left" if p0[0] > mid >= p1[0] else None)
        if direction is not None:
            fraction = float((mid-p0[0])/(p1[0]-p0[0]))
            y = float(p0[1]+fraction*(p1[1]-p0[1]))
            route = ("bottom" if y <= self._gate["bottom_max_y"]+1e-7 else
                     "top" if y >= self._gate["top_min_y"]-1e-7 else None)
            crossing = {"route": route, "direction": direction,
                "time": t0+fraction*(t1-t0), "t": t0+fraction*(t1-t0), "position": [mid, y],
                "request_index": len(self._route_requests)-1}
            self._midline_crossings.append(crossing)
            self._pending_crossing = crossing
        side = self._clear_side(p1)
        if side is not None:
            if self._origin_side is not None and side != self._origin_side:
                expected = "left_to_right" if side == "right" else "right_to_left"
                pending = self._pending_crossing
                if pending is not None and pending["direction"] == expected:
                    self._passages.append({**pending, "completed_time": t1,
                        "completed_position": p1.tolist(), "origin_side": self._origin_side})
            self._origin_side = side
            self._pending_crossing = None
        self._route_previous_position = p1.copy()
        self._route_previous_time = t1

    def _score_first_target(self):
        if self._terminal:
            return
        passage = self._passages[-1] if self._passages else None
        used = passage["route"] if passage is not None and passage["direction"] == "left_to_right" else None
        post_instruction = bool(passage is not None and self._requirement_since is not None
                                and passage["time"] >= self._requirement_since-1e-9)
        correct = self._requested_route is not None and used == self._requested_route and post_instruction
        self._arrival_route_score = {"time": float(self._first_target_time),
            "requested_route": self._requested_route, "used_passage": used,
            "post_instruction_passage": post_instruction, "correct_arrival": bool(correct),
            "passage": copy.deepcopy(passage), "request_index": len(self._route_requests)-1,
            "position": self.world.position.tolist()}
        self._terminal = True

    def _score_arrivals(self):
        if self._route_ready:
            self._audit_passages()
        super()._score_arrivals()
        if self._route_ready and self._first_target_time is not None:
            self._score_first_target()

    def step(self, muscles, dt=.1):
        if self._terminal:
            raise RuntimeError("Trial ended at first red arrival; call reset")
        packet = super().step(muscles, dt)
        self._append_trajectory()
        return packet

    def sensor_packet(self, dt=None):
        packet = super().sensor_packet(dt=dt)
        if not self._route_ready or self._requested_route is None:
            return packet
        times = self.world.time + np.arange(len(packet["waveform"])) / SAMPLE_RATE
        elapsed = times-self._route_cue_start
        active = (elapsed >= -1e-10) & (elapsed < self.route_cue_seconds-1e-10)
        route_wave = self.route_cue_amplitude * np.sin(
            2*math.pi*self._route_tone_map[self._requested_route]*elapsed + self._route_phase)
        route_wave[~active] = 0
        packet["waveform"] += route_wave
        return packet

    def private_metrics(self, *, include_trajectory=False):
        """Private evaluator data. Reward is +1 ONLY if correct_arrival is true."""
        metrics = super().private_metrics()
        arrival = self._arrival_route_score
        first_forward = next((p for p in self._passages if p["direction"] == "left_to_right"), None)
        metrics.update({"requested_route": self._requested_route,
            "route_tone_map": dict(self._route_tone_map), "route_gate_geometry": dict(self._gate),
            "route_requests": copy.deepcopy(self._route_requests),
            "passages": copy.deepcopy(self._passages),
            "crossings": copy.deepcopy(self._passages),
            "chosen_route": first_forward["route"] if first_forward else None,
            "midline_crossings": copy.deepcopy(self._midline_crossings),
            "arrival_route_score": copy.deepcopy(arrival),
            "terminated": self._terminal,
            "used_passage": arrival["used_passage"] if arrival else None,
            "correct_arrival": bool(arrival and arrival["correct_arrival"]),
            "success": bool(arrival and arrival["correct_arrival"]),
            "trajectory_samples": len(self._route_trajectory)})
        if include_trajectory:
            metrics["route_trajectory"] = self.private_trajectory()
        return metrics

    def private_trajectory(self):
        """Control-frame trajectory and request indices, for offline audit only."""
        return copy.deepcopy(self._route_trajectory)

    def get_view_state(self):
        view = super().get_view_state()
        view.update({"route_cue_active": self._requested_route is not None and
                     self.world.time < self._route_cue_start+self.route_cue_seconds-1e-10,
                     "requested_route": self._requested_route,
                     "route_tone_map": dict(self._route_tone_map)})
        return view


RouteSwitchEnvironment = RouteSwitchEnv


def self_check():
    """Physical passage fixtures and scorer units, never model demonstrations."""
    report = {}
    env = RouteSwitchEnv(seed=33)
    packet = env.reset(route="bottom")
    assert set(packet) == {"observation", "waveform"}
    assert set(packet["observation"]) == SENSOR_KEYS
    assert np.array_equal(packet["waveform"], env.sensor_packet()["waveform"])
    spectrum = np.abs(np.fft.rfft(packet["waveform"])) * 2 / len(packet["waveform"])
    bins = np.fft.rfftfreq(len(packet["waveform"]), 1/SAMPLE_RATE)
    amplitude = lambda f: float(spectrum[np.argmin(abs(bins-f))])
    assert abs(amplitude(220)-.45) < 1e-10 and abs(amplitude(990)-.45) < 1e-10
    assert amplitude(1320) < 1e-10
    report["simultaneous_target_and_route_pcm"] = {"220_Hz": amplitude(220), "990_Hz": amplitude(990)}
    for _ in range(100):
        env.step([0, 0, 0, 0])
    assert np.count_nonzero(env.sensor_packet()["waveform"]) == 0
    branch = copy.deepcopy(env)
    before = copy.deepcopy(env.world.get_state())
    original_rng = copy.deepcopy(env.rng.bit_generator.state)
    bottom = env.emit_route_cue("bottom")
    top = branch.emit_route_cue("top")
    assert env.world.get_state() == before == branch.world.get_state()
    assert env.rng.bit_generator.state == original_rng == branch.rng.bit_generator.state
    for key in SENSOR_KEYS:
        assert np.array_equal(bottom["observation"][key], top["observation"][key])
    assert not np.array_equal(bottom["waveform"], top["waveform"])
    assert env.private_trajectory() == branch.private_trajectory()
    rng = np.random.default_rng(73)
    for _ in range(40):
        action = rng.uniform(0, .6, 4)
        a, b = env.step(action), branch.step(action)
        for key in SENSOR_KEYS:
            assert np.array_equal(a["observation"][key], b["observation"][key])
        assert env.world.get_state() == branch.world.get_state()
    assert np.count_nonzero(env.sensor_packet()["waveform"]) == 0
    report["clone_switch_changes_only_pcm_and_private_bookkeeping"] = True
    report["cue_stops_and_exact_same_actions_same_physics"] = True
    swapped = RouteSwitchEnv(seed=33, route_tone_permutation=(1, 0))
    p = swapped.reset(route="bottom")["waveform"]
    expected = RouteSwitchEnv(seed=33).reset(route="top")["waveform"]
    assert np.array_equal(p, expected)
    report["swapped_mapping_has_identical_pcm_for_equal_frequency"] = True

    # Actual muscles traverse each gap. Fixed input is a test of the physics and
    # passage detector, not a learned navigator and not a training demonstration.
    actual = {}
    for route, y in (("bottom", .55), ("top", 6.8)):
        e = RouteSwitchEnv(seed=4)
        e.reset(route=route, start=(4., y, 0.))
        for _ in range(35):
            e.step([.45, 0, .45, 0])
        passages = e.private_metrics()["passages"]
        assert len(passages) == 1 and passages[0]["route"] == route
        assert passages[0]["direction"] == "left_to_right"
        assert passages[0]["completed_time"] > passages[0]["time"] > 0
        assert e.private_metrics()["contact_steps"] == 0
        actual[route] = passages[0]
    report["actual_muscle_passages"] = actual
    blocked = RouteSwitchEnv(seed=4)
    blocked.reset(route="bottom", start=(4., 3.4, 0.))
    for _ in range(35):
        blocked.step([.45, 0, .45, 0])
    assert not blocked.private_metrics()["passages"]
    assert blocked.world.position[0] <= blocked._gate["left_x"]+1e-7
    assert blocked.private_metrics()["contact_steps"] > 0
    report["blocked_forward_muscles_cannot_fake_passage"] = True

    # Isolated scoring units use explicit synthetic history, not a claim of
    # physical goal-reaching. They test first-entry finality and temporal rules.
    detector = RouteSwitchEnv(seed=8)
    detector.reset(start=(4., .55, 0.), route="bottom")
    for x in (5.1, 4.0):
        detector.world.position[0] = x
        detector.world.time += 1.0
        detector._audit_passages()
    assert len(detector._midline_crossings) == 2 and not detector._passages
    for x in (5.5, 4.0):
        detector.world.position[0] = x
        detector.world.time += 1.0
        detector._audit_passages()
    assert [p["direction"] for p in detector._passages] == ["left_to_right", "right_to_left"]
    report["detector_units_midline_retreat_excluded_and_reverse_direction_recorded"] = True
    no_passage = RouteSwitchEnv(seed=8)
    no_passage.reset(start=(8.5, 1.5, 0.), route="bottom")
    assert no_passage.private_metrics()["terminated"] and not no_passage.private_metrics()["correct_arrival"]
    try:
        no_passage.step([0, 0, 0, 0])
        raise AssertionError("An ended trial advanced")
    except RuntimeError:
        pass
    for requested, used, time, since, correct in (
            ("bottom", "bottom", 3., 0., True), ("top", "bottom", 3., 0., False),
            ("bottom", "bottom", 3., 4., False), ("top", "top", 5., 4., True)):
        e = RouteSwitchEnv(seed=4)
        e.reset(route=requested)
        e._requirement_since = since
        e._first_target_time = 8.
        e._passages = [{"route": used, "direction": "left_to_right", "time": time}]
        e._score_first_target()
        assert e.private_metrics()["correct_arrival"] is correct
    report["scoring_units_first_arrival_final_and_latest_instruction_required"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).parent / "results" / "route_environment_audit.json"))
    args = parser.parse_args()
    result = self_check()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
