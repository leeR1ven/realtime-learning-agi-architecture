"""Color-beacon navigation environment with a deliberately restricted sensor API.

The simulated body is the existing Newtonian planar four-muscle body. A target's
non-spatial mono tone sounds for the first 0.6 s of a trial, then stops. All
beacons remain physically identical regardless of which one is the target.
The environment has no navigation policy or demonstration route.

Model API::
    env = BeaconNavigationEnv(ray_count=31, seed=3)
    packet = env.reset(scenario="barrier", start="left_middle", target="red")
    packet = env.step(four_muscle_activations, dt=.1)

Pass ONLY packet to a model. The packet has ``observation`` and ``waveform``.
Observation keys: ray_distances, ray_colors, velocity, angular_velocity, pain,
touch. Velocity is body-frame [forward, left], NOT world x/y. ``world``,
``private_metrics`` and ``get_view_state`` are for the evaluator/viewer only.

Units: m, s, N, rad; waveforms are mono float64 PCM sampled at 8 kHz. Distances
are measured from the body centre. Ray angles span [-90,+90] degrees and are
fixed, so they are configuration rather than a per-frame privileged input.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "闭环仿真"))
from world import World, WorldConfig

SAMPLE_RATE = 8000
BASE_FREQUENCIES = (220.0, 440.0, 660.0)
SENSOR_KEYS = frozenset(("ray_distances", "ray_colors", "velocity", "angular_velocity", "pain", "touch"))
BEACON_NAMES = ("red", "green", "blue")
BEACON_COLORS = ((1.0, .04, .04), (.04, 1.0, .04), (.04, .04, 1.0))


@dataclass(frozen=True)
class Scene:
    name: str
    width: float
    height: float
    walls: tuple[tuple[float, float, float, float], ...]
    beacon_positions: tuple[tuple[float, float], ...]
    starts: Mapping[str, tuple[float, float, float]]


SCENES = {
    "barrier": Scene("barrier", 10., 8., ((4.8, 1.1, 5.2, 6.2),),
        ((8.5, 1.5), (8.5, 6.5), (1.5, 6.5)),
        {"left_middle": (1.3, 3.4, 0.), "left_bottom": (1.3, 1.4, .15),
         "right_middle": (8.6, 3.8, math.pi), "bottom_middle": (3.0, .55, math.pi/2)}),
    "offset": Scene("offset", 10., 8., ((3.2, .8, 3.5, 5.8), (6.3, 2.2, 6.6, 7.2)),
        ((8.6, 1.2), (8.6, 6.8), (1.2, 6.8)),
        {"left_middle": (1.2, 3., 0.), "left_bottom": (1.2, 1.2, .2),
         "right_middle": (8.8, 4., math.pi), "bottom_middle": (5.1, .55, math.pi/2)}),
    "crossbar": Scene("crossbar", 10., 8., ((2.4, 3.5, 7.7, 3.8), (5., 3.8, 5.3, 6.4)),
        ((8.7, 1.3), (8.7, 6.7), (1.3, 6.7)),
        {"left_middle": (1.2, 3., 0.), "left_bottom": (1.2, 1.3, .1),
         "right_middle": (8.8, 4., math.pi), "bottom_middle": (5.1, .55, math.pi/2)}),
}


class _PhysicsWorld(World):
    """Skip only redundant ray rendering during environment scoring substeps.

    World.step already performs all integration and collision handling before
    its final observe() call. During that call the navigation environment needs
    only contact feedback. No physical calculation or scoring sample is skipped.
    The flag is restored even if World.step rejects an action or geometry.
    """

    def __init__(self, *args, **kwargs):
        self._contact_observation_only = False
        super().__init__(*args, **kwargs)

    def observe(self):
        if self._contact_observation_only:
            return {"contact_impulse": float(self.contact_impulse),
                    "contact": bool(self.contact), "touch": self.touch.copy()}
        return super().observe()

    def step_physics_only(self, muscles, dt):
        previous = self._contact_observation_only
        self._contact_observation_only = True
        try:
            return super().step(muscles, dt)
        finally:
            self._contact_observation_only = previous


class BeaconNavigationEnv:
    """A trial environment. Reset randomly chooses omitted start and target.

    ``target`` is a beacon name or index, for the experiment driver only.
    ``start`` is a named start, an index into named starts, or (x,y,heading).
    ``tone_map`` maps beacon names to distinct Hz; ``tone_permutation`` remaps
    the three fixed base frequencies. Both are stable until explicitly changed.
    """

    def __init__(self, scenario="barrier", *, ray_count=31, seed=20260906,
                 tone_map=None, tone_permutation=None, cue_seconds=.6,
                 cue_amplitude=.7, beacon_radius=.3, arrival_tolerance=.08,
                 fast_internal_observe=True):
        if not isinstance(ray_count, int) or ray_count < 5 or ray_count % 2 != 1:
            raise ValueError("ray_count must be an odd integer >=5")
        if not np.isfinite(cue_seconds) or cue_seconds <= 0:
            raise ValueError("cue_seconds must be positive")
        if not np.isfinite(cue_amplitude) or not 0 < cue_amplitude <= 1:
            raise ValueError("cue_amplitude must be in (0,1]")
        if beacon_radius <= 0 or arrival_tolerance < 0:
            raise ValueError("Invalid beacon or arrival radius")
        self.rng = np.random.default_rng(seed)
        self.ray_count = ray_count
        self.ray_angles = np.linspace(-math.pi/2, math.pi/2, ray_count)
        self.cue_seconds = float(cue_seconds)
        self.cue_amplitude = float(cue_amplitude)
        self.beacon_radius = float(beacon_radius)
        self.arrival_tolerance = float(arrival_tolerance)
        self.fast_internal_observe = bool(fast_internal_observe)
        self._tone_map = dict(zip(BEACON_NAMES, BASE_FREQUENCIES))
        if tone_map is not None and tone_permutation is not None:
            raise ValueError("Specify tone_map OR tone_permutation")
        if tone_map is not None:
            self.set_tone_map(tone_map)
        if tone_permutation is not None:
            self.set_tone_permutation(tone_permutation)
        self.scene = self._scene(scenario)
        self.world: World
        self._control_dt = .1
        self.reset(scenario=scenario)

    @staticmethod
    def _scene(scenario):
        if isinstance(scenario, Scene):
            return scenario
        if scenario not in SCENES:
            raise ValueError(f"Unknown scenario {scenario!r}; choose {tuple(SCENES)}")
        return SCENES[scenario]

    def set_tone_map(self, mapping):
        if isinstance(mapping, Mapping):
            values = dict(mapping)
        else:
            sequence = tuple(mapping)
            if len(sequence) != len(BEACON_NAMES):
                raise ValueError("tone_map sequence must contain exactly three frequencies")
            values = dict(zip(BEACON_NAMES, sequence))
        if set(values) != set(BEACON_NAMES):
            raise ValueError("tone_map must contain red, green and blue")
        values = {name: float(values[name]) for name in BEACON_NAMES}
        if (len(set(values.values())) != 3 or
                not all(np.isfinite(f) and 50 <= f < SAMPLE_RATE/2 for f in values.values())):
            raise ValueError("Three distinct finite audible frequencies below Nyquist are required")
        self._tone_map = values

    def set_tone_permutation(self, permutation):
        permutation = tuple(permutation)
        if sorted(permutation) != [0, 1, 2]:
            raise ValueError("tone_permutation must be a permutation of (0,1,2)")
        self.set_tone_map({name: BASE_FREQUENCIES[i] for name, i in zip(BEACON_NAMES, permutation)})

    def reset(self, scenario=None, start=None, target=None, *, tone_map=None, tone_permutation=None):
        if scenario is not None:
            self.scene = self._scene(scenario)
        if tone_map is not None and tone_permutation is not None:
            raise ValueError("Specify tone_map OR tone_permutation")
        if tone_map is not None:
            self.set_tone_map(tone_map)
        if tone_permutation is not None:
            self.set_tone_permutation(tone_permutation)
        scene = self.scene
        if len(scene.beacon_positions) != len(BEACON_NAMES):
            raise ValueError("A scene must supply one position for each of the three beacons")
        if target is None:
            target = int(self.rng.integers(3))
        if isinstance(target, str):
            if target not in BEACON_NAMES:
                raise ValueError("Unknown target beacon")
            target = BEACON_NAMES.index(target)
        if isinstance(target, bool) or int(target) != target or not 0 <= target < 3:
            raise ValueError("target must be a beacon name or index 0..2")
        self._target_index = int(target)
        self._target_name = BEACON_NAMES[self._target_index]
        starts = tuple(scene.starts)
        if start is None:
            start = starts[int(self.rng.integers(len(starts)))]
        if isinstance(start, int):
            if not 0 <= start < len(starts):
                raise ValueError("Start index out of range")
            start = starts[start]
        if isinstance(start, str):
            if start not in scene.starts:
                raise ValueError(f"Unknown start {start!r}")
            self._start_name = start
            pose = np.asarray(scene.starts[start], dtype=float)
        else:
            self._start_name = "custom"
            pose = np.asarray(start, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("start must be a named start or finite (x,y,heading)")
        self._initial_pose = pose.copy()
        config = WorldConfig(width=scene.width, height=scene.height,
                             ray_range=float(math.hypot(scene.width, scene.height)),
                             ray_angles=tuple(self.ray_angles))
        # Initial construction has no walls, so arbitrary valid custom starts
        # can be set before testing wall overlap.
        self.world = _PhysicsWorld(config)
        self.world.reset(position=pose[:2], heading=float(pose[2]), walls=scene.walls,
            objects=[{"position": p, "radius": self.beacon_radius, "color": col, "name": name}
                     for name, p, col in zip(BEACON_NAMES, scene.beacon_positions, BEACON_COLORS)])
        self._audio_phase = float(self.rng.uniform(0, 2*math.pi))
        self._control_dt = .1
        self._touch = np.zeros(4)
        self._pain = 0.
        self._steps = 0
        self._path_length = 0.
        self._contact_steps = 0
        self._contact_impulse = 0.
        self._arrivals = []
        self._inside_beacons = set()
        self._first_target_time = None
        self._first_beacon_name = None
        self._wrong_entries = 0
        self._score_arrivals()
        return self.sensor_packet()

    def _score_arrivals(self):
        radius = self.beacon_radius + self.world.config.radius + self.arrival_tolerance
        inside = {i for i, p in enumerate(self.scene.beacon_positions)
                  if np.linalg.norm(self.world.position - p) <= radius}
        for i in sorted(inside - self._inside_beacons):
            name = BEACON_NAMES[i]
            correct = i == self._target_index
            if self._first_beacon_name is None:
                self._first_beacon_name = name
            if correct and self._first_target_time is None:
                self._first_target_time = float(self.world.time)
            if not correct:
                self._wrong_entries += 1
            self._arrivals.append({"name": name, "time": float(self.world.time), "correct": correct})
        self._inside_beacons = inside

    def step(self, muscles, dt=.1):
        if not np.isfinite(dt) or dt <= 0 or dt > 1:
            raise ValueError("dt must be finite, positive and <=1 second")
        self._control_dt = float(dt)
        self._touch[:] = 0
        impulse = 0.
        contact = False
        # Score geometry every <=10ms, independent of the model's action rate.
        # This measures path length more accurately and detects brief entries.
        count = max(1, math.ceil(dt / .01))
        subdt = dt / count
        for _ in range(count):
            old_position = self.world.position.copy()
            obs = (self.world.step_physics_only(muscles, subdt) if self.fast_internal_observe
                   else self.world.step(muscles, subdt))
            self._path_length += float(np.linalg.norm(self.world.position - old_position))
            impulse += obs["contact_impulse"]
            contact = contact or obs["contact"]
            self._touch = np.maximum(self._touch, obs["touch"])
            self._score_arrivals()
        self._pain = float(np.clip(impulse / self.world.config.pain_impulse_scale, 0, 1))
        self._contact_impulse += float(impulse)
        self._contact_steps += int(contact)
        self._steps += 1
        return self.sensor_packet(dt=dt)

    def sensor_packet(self, dt=None):
        """Pure read; repeated calls at the same state return identical PCM.

        A packet's audio covers [current simulation time, current time+dt).
        A late read never replays the beginning of the cue.
        """
        dt = self._control_dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0 or dt > 1:
            raise ValueError("Audio chunk duration must be finite and in (0,1]")
        obs = self.world.observe()
        f = np.array([math.cos(self.world.heading), math.sin(self.world.heading)])
        left = np.array([-f[1], f[0]])
        sensory = {"ray_distances": obs["ray_distances"].copy(),
                   "ray_colors": obs["ray_colors"].copy(),
                   "velocity": np.array([self.world.velocity @ f, self.world.velocity @ left]),
                   "angular_velocity": float(obs["angular_velocity"]),
                   "pain": float(self._pain), "touch": self._touch.copy()}
        sample_count = max(1, int(round(dt * SAMPLE_RATE)))
        times = self.world.time + np.arange(sample_count) / SAMPLE_RATE
        wave = self.cue_amplitude * np.sin(2*math.pi*self._tone_map[self._target_name]*times + self._audio_phase)
        wave[times >= self.cue_seconds - 1e-10] = 0.
        return {"observation": sensory, "waveform": wave}

    def private_metrics(self):
        """Evaluator-only: never append these fields to model observations."""
        target = np.asarray(self.scene.beacon_positions[self._target_index])
        return {"scenario": self.scene.name, "start": self._start_name,
                "target": self._target_name, "target_index": self._target_index,
                "target_frequency_hz": self._tone_map[self._target_name],
                "tone_map": dict(self._tone_map), "elapsed_seconds": float(self.world.time),
                "control_steps": self._steps, "target_reached": self._first_target_time is not None,
                "first_target_time": self._first_target_time, "first_beacon": self._first_beacon_name,
                "first_beacon_correct": self._first_beacon_name == self._target_name,
                "wrong_beacon_entries": self._wrong_entries, "arrivals": [dict(a) for a in self._arrivals],
                "contact_steps": self._contact_steps, "normal_contact_impulse": self._contact_impulse,
                "path_length": self._path_length, "path_sampling_max_dt": .01,
                "remaining_target_distance": float(np.linalg.norm(self.world.position-target)),
                "initial_pose": self._initial_pose.tolist(), "final_position": self.world.position.tolist()}

    def get_view_state(self):
        """Rich state for a viewer, separated from the sensor_packet API."""
        view = self.world.get_state()
        view.update({"scenario": self.scene.name, "beacon_names": list(BEACON_NAMES),
                     "tone_map": dict(self._tone_map), "cue_active": self.world.time < self.cue_seconds,
                     "private_metrics": self.private_metrics()})
        return view

    def manual_step(self, muscles, dt=.1):
        """Teacher/UI may supply the same four muscle activations as the model.

        The environment provides no route or target-dependent action. A caller
        may record returned sensory packets and demonstrated muscle values.
        """
        return self.step(muscles, dt)


NavigationEnv = BeaconNavigationEnv


def _geometry_reachability(scene, body_radius=.18, grid=.10):
    """Audit-only flood fill of free configuration space; returns no routes."""
    xs = np.arange(body_radius + .025, scene.width-body_radius, grid)
    ys = np.arange(body_radius + .025, scene.height-body_radius, grid)
    free = np.ones((len(xs), len(ys)), dtype=bool)
    margin = body_radius + .025
    for x0, y0, x1, y1 in scene.walls:
        # Axis-expanded boxes are conservative around corners, so a free grid
        # path is valid for the circle with the stated clearance.
        free &= ~((xs[:, None] >= x0-margin) & (xs[:, None] <= x1+margin)
                  & (ys[None, :] >= y0-margin) & (ys[None, :] <= y1+margin))
    def cell(p):
        return int(np.abs(xs-p[0]).argmin()), int(np.abs(ys-p[1]).argmin())
    first = cell(next(iter(scene.starts.values())))
    assert free[first]
    visited = {first}
    queue = deque([first])
    while queue:
        i, j = queue.popleft()
        for ni, nj in ((i-1,j), (i+1,j), (i,j-1), (i,j+1)):
            if 0 <= ni < len(xs) and 0 <= nj < len(ys) and free[ni,nj] and (ni,nj) not in visited:
                visited.add((ni,nj)); queue.append((ni,nj))
    points = list(scene.starts.values()) + list(scene.beacon_positions)
    return {"all_starts_and_beacons_connected": all(cell(p) in visited for p in points),
            "grid_metres": grid, "body_radius_with_clearance": margin,
            "reachable_free_cells": len(visited)}


def self_check():
    results = {}
    env = BeaconNavigationEnv(seed=77)
    initial = env.reset("barrier", "left_middle", "red")
    assert set(initial) == {"observation", "waveform"}
    assert set(initial["observation"]) == SENSOR_KEYS
    assert initial["observation"]["ray_distances"].shape == (31,)
    assert initial["observation"]["ray_colors"].shape == (31, 3)
    assert np.array_equal(initial["waveform"], env.sensor_packet()["waveform"])
    results["white_list_and_pure_read"] = True
    # Target choice can affect the cue but cannot secretly mark its image.
    red = env.reset("barrier", "left_middle", "red")
    blue = env.reset("barrier", "left_middle", "blue")
    for key in SENSOR_KEYS:
        assert np.array_equal(red["observation"][key], blue["observation"][key])
    assert not np.array_equal(red["waveform"], blue["waveform"])
    results["target_identity_not_leaked_into_visual_or_body_sensors"] = True

    env.reset("barrier", "left_middle", "red")
    assert np.linalg.norm(env.sensor_packet()["waveform"]) > 1
    for _ in range(7):
        env.step([0,0,0,0])
    assert np.count_nonzero(env.sensor_packet()["waveform"]) == 0
    results["cue_stops_after_point_six_seconds"] = True

    env.reset("barrier", "left_middle", "red")
    initial_colors = env.sensor_packet()["observation"]["ray_colors"]
    assert not np.any(np.all(np.isclose(initial_colors, BEACON_COLORS[0]), axis=1))
    target = np.asarray(env.scene.beacon_positions[0])
    direction = target - env.world.position
    distance = np.linalg.norm(direction)
    direction /= distance
    hit = World._ray_box(env.world.position, direction, np.asarray(env.scene.walls[0]))
    assert 0 < hit < distance
    env.world.heading = float(math.atan2(direction[1], direction[0]))
    centre = env.sensor_packet()["observation"]
    assert np.allclose(centre["ray_colors"][15], env.world.config.wall_color)
    assert centre["ray_distances"][15] < distance
    results["direct_line_to_red_is_physically_occluded_by_wall"] = True
    results["red_target_absent_from_all_initial_rays"] = True

    env.reset("barrier", "left_middle", "red")
    start_x = env.world.position[0]
    for _ in range(80):
        packet = env.step([.6,0,.6,0])
        assert set(packet["observation"]) == SENSOR_KEYS
        assert np.isfinite(np.r_[env.world.position, env.world.velocity, env.world.angular_velocity]).all()
    assert env.world.position[0] > start_x + 1
    assert env.world.position[0] <= 4.8-env.world.config.radius+1e-7
    assert env.private_metrics()["normal_contact_impulse"] > 0
    results["real_muscles_move_body_and_cannot_cross_barrier"] = {
        "final_x": float(env.world.position[0]), "impulse": env.private_metrics()["normal_contact_impulse"]}

    env.reset("barrier", (2., .6, math.pi/2), "green")
    env.world.velocity[:] = (1., 2.)  # audit-only state setup; never an action API
    assert np.allclose(env.sensor_packet()["observation"]["velocity"], (2., -1.))
    results["velocity_is_body_frame"] = True

    frequencies = []
    env.set_tone_permutation((2,0,1))
    for name in BEACON_NAMES:
        packet = env.reset("barrier", "left_middle", name)
        x = packet["waveform"]
        frequencies.append(float(np.fft.rfftfreq(len(x),1/SAMPLE_RATE)[np.argmax(np.abs(np.fft.rfft(x)))]))
    assert frequencies == [660., 220., 440.]
    results["tone_mapping_can_be_permuted"] = frequencies

    geometries = {name:_geometry_reachability(scene) for name,scene in SCENES.items()}
    assert all(g["all_starts_and_beacons_connected"] for g in geometries.values())
    results["geometry_reachability"] = geometries
    # Arrival is based on actual body coordinates and labelled only privately.
    env.reset("barrier", (8.5,1.5,0.), "red")
    assert env.private_metrics()["target_reached"]
    assert env.private_metrics()["first_target_time"] == 0
    assert "target_reached" not in env.sensor_packet()["observation"]
    results["private_arrival_scoring"] = True
    return results


def verify_fast_observation():
    """Compare every frame against the original full-observation stepping path.

    The baseline still calls the original World.step/World.observe for each
    10ms scoring substep. Timing covers the complete environment step and final
    sensor packet; it does not exclude collision checks or sound generation.
    """
    import time
    rng = np.random.default_rng(614713)
    actions = np.vstack((np.tile([.6,0,.6,0], (100,1)),
                         np.tile([0,.3,.3,0], (50,1)), rng.uniform(0,.8,(100,4))))
    durations = np.full(len(actions), .1)
    durations[-50:] = np.tile([.037,.083,.1,.2,.05],10)

    def equal(a, b, path="root"):
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for key in a:
                equal(a[key], b[key], path+"."+key)
        elif isinstance(a, (list,tuple)):
            assert len(a) == len(b), path
            for i, (x,y) in enumerate(zip(a,b)):
                equal(x,y,path+f"[{i}]")
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a,b), path
        else:
            assert a == b, path

    comparisons = []
    for scenario in SCENES:
        baseline = BeaconNavigationEnv(scenario,seed=901,fast_internal_observe=False)
        fast = BeaconNavigationEnv(scenario,seed=901,fast_internal_observe=True)
        equal(baseline.reset(start="left_middle",target="red"),
              fast.reset(start="left_middle",target="red"))
        for action, dt in zip(actions,durations):
            equal(baseline.step(action,float(dt)),fast.step(action,float(dt)),"packet")
            equal(baseline.get_view_state(),fast.get_view_state(),"full_world_and_score")
        comparisons.append({"scenario":scenario,"frames":len(actions),
                            "all_physics_sensors_and_private_scores_exact":True,
                            "contact_steps":fast.private_metrics()["contact_steps"],
                            "path_length":fast.private_metrics()["path_length"]})

    timings = {"baseline":[],"optimized":[]}
    # Alternate order so the timing is less sensitive to system warm-up/load.
    for repetition in range(3):
        order = (False,True) if repetition % 2 == 0 else (True,False)
        for optimize in order:
            env = BeaconNavigationEnv(seed=901,fast_internal_observe=optimize)
            env.reset(start="left_middle",target="red")
            begun = time.perf_counter()
            for action,dt in zip(actions,durations):
                env.step(action,float(dt))
            elapsed = time.perf_counter()-begun
            timings["optimized" if optimize else "baseline"].append(elapsed)
    baseline_seconds = float(np.median(timings["baseline"]))
    fast_seconds = float(np.median(timings["optimized"]))
    # Verify exception paths cannot accidentally leave public observations slim.
    env = BeaconNavigationEnv()
    try:
        env.world.step_physics_only([np.nan,0,0,0],.01)
    except ValueError:
        pass
    assert "ray_distances" in env.world.observe()
    return {"comparisons":comparisons,"physics_substep_changed":False,
            "arrival_sampling_changed":False,"maximum_value_difference":0,
            "exception_restores_complete_observation":True,
            "timing_frames_per_run":len(actions),"raw_timing_seconds":timings,
            "median_baseline_seconds":baseline_seconds,"median_optimized_seconds":fast_seconds,
            "speedup":baseline_seconds/fast_seconds,
            "baseline_ms_per_step":baseline_seconds/len(actions)*1000,
            "optimized_ms_per_step":fast_seconds/len(actions)*1000,
            "ray_evaluations_per_point_one_second_step":{"baseline":11,"optimized":1}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--benchmark", action="store_true",help="Compare optimized and original observation paths exactly, then time them")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent/"results"/"environment_checks.json")
    args = parser.parse_args()
    result = self_check()
    if args.benchmark:
        result["observation_optimization"] = verify_fast_observation()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"PASS: {len(result)} environment check groups")
