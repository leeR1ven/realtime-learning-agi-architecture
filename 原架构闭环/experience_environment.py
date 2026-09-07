"""Continuous sensory experience using the existing physical four-muscle body.

This environment contains stimulus scheduling, rendering and private measurement,
not a controller. Only ``sensor_packet()``/``step()`` results may enter a brain:
the unchanged six-field observation plus 8 kHz mono PCM. Beacon names, tone
frequencies, coordinates, presentation events and visits stay on the experiment
side. There is no automatic cue, reward, teacher action or navigation success.

``setup_pairing`` changes ordinary visible objects and emits a tone only after
checking actual ray visibility. ``hide_beacons`` and ``emit_tone`` then permit a
sound-only probe in the SAME world, at its current time and physical state.
``restore_beacons`` permits subsequent exploration. None resets a neural model
or the world clock. An explicit ``reset`` starts a new physical episode.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAVIGATION_DIR = ROOT / "色标导航"
if str(NAVIGATION_DIR) not in sys.path:
    sys.path.insert(0, str(NAVIGATION_DIR))
from environment import (BeaconNavigationEnv, Scene, BEACON_NAMES,
                         BEACON_COLORS, SAMPLE_RATE, SENSOR_KEYS)


PAIRING_SCENE = Scene(
    "pairing_open", 10., 8., (),
    ((6., 4.), (6., 6.), (6., 2.)),
    {"observation": (3., 4., 0.)},
)


class ExperienceEnvironment(BeaconNavigationEnv):
    """31 physical rays, four antagonist muscles, explicitly scheduled sound.

    ``scenario`` also accepts the existing barrier/offset/crossbar scenes or a
    Scene instance. Optional ``texture_seed`` uses the already audited wall
    pigment renderer; it changes only visible wall RGB, never the physics.
    ``target`` in reset is compatibility-only private metadata. It generates
    neither audio nor reward. Visits are measured against actual visible circles.
    """

    def __init__(self, scenario="pairing_open", *, seed=20260906, ray_count=31,
                 texture_seed=None, **kwargs):
        self.texture_seed = texture_seed
        self._tone = None
        self._stimulus_events = []
        self._exposure_samples = []
        self._visits = []
        self._inside_visible = set()
        self._original_objects = []
        super().__init__(scenario=scenario, seed=seed, ray_count=ray_count, **kwargs)

    @staticmethod
    def _scene(scenario):
        if isinstance(scenario, str) and scenario == "pairing_open":
            return PAIRING_SCENE
        return BeaconNavigationEnv._scene(scenario)

    def reset(self, scenario=None, start=None, target=None, **kwargs):
        self._tone = None
        self._stimulus_events = []
        self._exposure_samples = []
        self._visits = []
        self._inside_visible = set()
        # Avoid consuming a target-selection RNG draw for an unused task label.
        super().reset(scenario=scenario, start=start,
                      target="red" if target is None else target, **kwargs)
        self._original_objects = copy.deepcopy(self.world.objects)
        if self.texture_seed is not None:
            from textured_route_environment import SmoothWallPigment, _TexturedPhysicsWorld
            pigment = SmoothWallPigment(self.scene.width, self.scene.height,
                                        seed=int(self.texture_seed))
            self.world = _TexturedPhysicsWorld.from_world(self.world, pigment, True)
        return self.sensor_packet()

    @staticmethod
    def _validate_tone(frequency, duration, amplitude):
        frequency, duration, amplitude = map(float, (frequency, duration, amplitude))
        if not np.isfinite(frequency) or not 0 < frequency < SAMPLE_RATE / 2:
            raise ValueError("frequency must be finite and strictly between 0 and 4000 Hz")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        if not np.isfinite(amplitude) or not 0 < amplitude <= 1:
            raise ValueError("amplitude must be finite and in (0,1]")
        return frequency, duration, amplitude

    def emit_tone(self, frequency, duration=.6, amplitude=.45):
        """Experiment-side mono tone, with no direction, label or amplitude cue.

        Arbitrary positive frequencies below Nyquist are accepted. A new call
        replaces the previous tone. It does not change objects, pose or time.
        Repeated packet reads at the same state yield exactly the same PCM.
        """
        frequency, duration, amplitude = self._validate_tone(frequency, duration, amplitude)
        self._tone = {"frequency_hz": frequency, "start_time": float(self.world.time),
                      "end_time": float(self.world.time + duration), "amplitude": amplitude}
        self._stimulus_events.append({"event": "emit_tone", **self._tone})
        return self.sensor_packet()

    def stop_tone(self):
        self._tone = None
        self._stimulus_events.append({"event": "stop_tone", "time": float(self.world.time)})
        return self.sensor_packet()

    def sensor_packet(self, dt=None):
        packet = super().sensor_packet(dt=dt)
        # Explicitly discard the base class's initial target cue. Only this
        # generic scheduler may generate audio in an ExperienceEnvironment.
        waveform = np.zeros_like(packet["waveform"])
        tone = self._tone
        if tone is not None:
            times = self.world.time + np.arange(len(waveform)) / SAMPLE_RATE
            active = ((times >= tone["start_time"] - 1e-10)
                      & (times < tone["end_time"] - 1e-10))
            phase_time = times[active] - tone["start_time"]
            waveform[active] = tone["amplitude"] * np.sin(
                2 * math.pi * tone["frequency_hz"] * phase_time)
        packet["waveform"] = waveform
        return packet

    def visible_beacon_rays(self):
        """PRIVATE experiment evidence from actual nearest-hit RGB rendering."""
        _, colors, _, kinds = self.world._rays()
        objects = np.asarray(kinds) == "object"
        return {name: np.flatnonzero(objects & np.all(np.isclose(colors, color,
                atol=1e-12, rtol=0), axis=1)).tolist()
                for name, color in zip(BEACON_NAMES, BEACON_COLORS)}

    def setup_pairing(self, beacon="red", frequency=220., duration=.6, *,
                      presentation_position=(6., 4.), amplitude=.45,
                      hide_others=True, require_visible=True):
        """Present a normal colored circle and its explicit simultaneous sound.

        This is stimulus presentation, not an action demonstration. Body pose,
        velocity, muscle state and time are untouched. A required-but-occluded
        stimulus raises an error and restores the old objects before any sound.
        ``presentation_position=None`` uses that beacon's original scene place.
        """
        if beacon not in BEACON_NAMES:
            raise ValueError("beacon must be red, green or blue")
        self._validate_tone(frequency, duration, amplitude)
        selected = copy.deepcopy(next(obj for obj in self._original_objects
                                      if obj["name"] == beacon))
        if presentation_position is not None:
            selected["position"] = np.asarray(presentation_position, dtype=float)
        position = selected["position"]
        radius = selected["radius"]
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("presentation_position must contain two finite coordinates")
        if not (radius <= position[0] <= self.scene.width - radius and
                radius <= position[1] <= self.scene.height - radius):
            raise ValueError("The presented circle must stay inside the visible arena")
        for wall in self.world.walls:
            bounds = wall["bounds"]
            nearest = np.clip(position, bounds[:2], bounds[2:])
            if np.linalg.norm(position - nearest) < radius:
                raise ValueError("The presented circle overlaps a solid wall")
        previous = copy.deepcopy(self.world.objects)
        objects = [] if hide_others else [copy.deepcopy(obj) for obj in self.world.objects
                                         if obj["name"] != beacon]
        self.world.set_geometry(objects=objects + [selected])
        ray_indices = self.visible_beacon_rays()[beacon]
        if require_visible and not ray_indices:
            self.world.set_geometry(objects=previous)
            raise ValueError("The presented beacon is not actually visible on any ray")
        # Presentation is not a locomotor arrival; mark overlapping visible
        # objects as already inside so it cannot invent a visit after insertion.
        self._inside_visible = self._visible_inside()
        self._stimulus_events.append({"event": "present", "time": float(self.world.time),
            "beacon": beacon, "position": position.tolist(), "visible_rays": ray_indices})
        return self.emit_tone(frequency, duration, amplitude)

    def hide_beacons(self):
        """Remove visual stimuli while preserving body, clock and ongoing tone."""
        self.world.set_geometry(objects=[])
        self._inside_visible = set()
        self._stimulus_events.append({"event": "hide_beacons", "time": float(self.world.time)})
        return self.sensor_packet()

    def restore_beacons(self):
        self.world.set_geometry(objects=copy.deepcopy(self._original_objects))
        self._inside_visible = self._visible_inside()
        self._stimulus_events.append({"event": "restore_beacons", "time": float(self.world.time)})
        return self.sensor_packet()

    def _visible_inside(self):
        return {obj["name"] for obj in self.world.objects
                if np.linalg.norm(self.world.position - obj["position"]) <=
                obj["radius"] + self.world.config.radius + self.arrival_tolerance}

    def _score_arrivals(self):
        # Evaluate only objects physically present now, not old scene positions
        # after a stimulus was hidden or moved. No target or reward is generated.
        inside = self._visible_inside()
        for name in sorted(inside - self._inside_visible):
            self._visits.append({"beacon": name, "time": float(self.world.time),
                                 "position": self.world.position.tolist()})
        self._inside_visible = inside

    def step(self, muscles, dt=.1):
        # A private trace of stimuli available at this real action's start.
        # Pure sensor reads are never counted as additional experienced frames.
        packet = self.sensor_packet(dt=dt)
        audible = bool(np.any(packet["waveform"] != 0))
        sample = {"time": float(self.world.time), "visible_rays": self.visible_beacon_rays(),
                  "audible": audible,
                  "frequency_hz": self._tone["frequency_hz"] if audible else None}
        result = super().step(muscles, dt=dt)
        self._exposure_samples.append(sample)
        return result

    def private_metrics(self):
        active_tone = (self._tone if self._tone is not None and
                       self.world.time < self._tone["end_time"] - 1e-10 else None)
        return {"scenario": self.scene.name, "elapsed_seconds": float(self.world.time),
                "control_steps": self._steps, "initial_pose": self._initial_pose.tolist(),
                "final_position": self.world.position.tolist(), "path_length": self._path_length,
                "contact_steps": self._contact_steps, "normal_contact_impulse": self._contact_impulse,
                "visits": copy.deepcopy(self._visits), "visible_beacon_rays": self.visible_beacon_rays(),
                "active_tone": copy.deepcopy(active_tone),
                "stimulus_events": copy.deepcopy(self._stimulus_events),
                "exposure_samples": copy.deepcopy(self._exposure_samples),
                "automatic_reward": False, "teacher_actions": 0,
                "texture_seed": self.texture_seed}

    def get_view_state(self):
        state = self.world.get_state()
        tone = self._tone
        state.update({"scenario": self.scene.name, "private_metrics": self.private_metrics(),
                      "cue_active": tone is not None and self.world.time < tone["end_time"] - 1e-10})
        return state


def self_check():
    env = ExperienceEnvironment(seed=7)
    packet = env.sensor_packet()
    assert set(packet) == {"observation", "waveform"}
    assert set(packet["observation"]) == SENSOR_KEYS
    assert packet["observation"]["ray_colors"].shape == (31, 3)
    assert packet["observation"]["touch"].shape == (4,)
    assert not np.any(packet["waveform"]), "There must be no automatic base target cue"
    initial_world = id(env.world)
    initial_pose = env.world.position.copy()
    exposures = []
    for trial in range(12):
        name = BEACON_NAMES[trial % 3]
        frequency = (220., 777., 1503.)[trial % 3]
        before = env.world.time
        packet = env.setup_pairing(name, frequency)
        assert env.visible_beacon_rays()[name] == [15]
        assert env.world.time == before and id(env.world) == initial_world
        for frame in range(6):
            packet = env.sensor_packet()
            assert np.any(packet["waveform"])
            assert env.visible_beacon_rays()[name]
            assert np.array_equal(packet["waveform"], env.sensor_packet()["waveform"])
            env.step(np.zeros(4))
        assert not np.any(env.sensor_packet()["waveform"])
        exposures.append({"beacon": name, "frequency_hz": frequency,
                          "start_time": before, "audible_visible_frames": 6})
        env.hide_beacons()
        silent = env.sensor_packet()
        audible = env.emit_tone(frequency)
        for key in SENSOR_KEYS:
            assert np.array_equal(silent["observation"][key], audible["observation"][key])
        assert not any(env.visible_beacon_rays().values())
        assert np.any(audible["waveform"])
        env.step(np.zeros(4))
        env.stop_tone()
    assert np.array_equal(initial_pose, env.world.position)
    assert id(env.world) == initial_world
    assert not env.private_metrics()["visits"]
    # Private target names and legacy target-frequency maps cannot affect this
    # packet: only the explicit generic scheduler is permitted to emit sound.
    probe_clone = copy.deepcopy(env)
    probe_clone._target_name = "blue"
    probe_clone._target_index = 2
    probe_clone.set_tone_map({"red": 181., "green": 953., "blue": 2159.})
    for key in SENSOR_KEYS:
        assert np.array_equal(env.sensor_packet()["observation"][key],
                              probe_clone.sensor_packet()["observation"][key])
    assert np.array_equal(env.sensor_packet()["waveform"], probe_clone.sensor_packet()["waveform"])
    continuous_elapsed = env.world.time
    # Arbitrary non-list frequency, clean single tone, exact 0.6 s stop.
    env.emit_tone(777.)
    samples = []
    for _ in range(6):
        samples.append(env.sensor_packet()["waveform"])
        env.step(np.zeros(4))
    pcm = np.concatenate(samples)
    expected = .45 * np.sin(2 * np.pi * 777 * np.arange(4800) / SAMPLE_RATE)
    pcm_error = float(np.max(np.abs(pcm - expected)))
    assert pcm_error < 1e-9 and not np.any(env.sensor_packet()["waveform"])
    # Impossible co-presentation must fail without moving the body or objects.
    snapshot = env.get_view_state()
    try:
        env.setup_pairing("red", 220., presentation_position=(1., 4.))
    except ValueError:
        pass
    else:
        raise AssertionError("An invisible presentation must not be called joint exposure")
    assert env.get_view_state() == snapshot
    # Compare original physics and endpoint sensory RGB on a wall collision and
    # turning sequence. No action depends on a goal, coordinate or planner.
    probe = ExperienceEnvironment("barrier", seed=8)
    reference = BeaconNavigationEnv("barrier", seed=8)
    probe.reset(start=(3.7, 3.4, 0.), target="red")
    reference.reset(start=(3.7, 3.4, 0.), target="red")
    sequence = ([1., 0., 1., 0.],) * 25 + ([0., 1., 1., 0.],) * 12
    sequence += ([.3, 0., .5, 0.],) * 20
    for muscles in sequence:
        a, b = probe.step(muscles), reference.step(muscles)
        for key in SENSOR_KEYS:
            assert np.array_equal(a["observation"][key], b["observation"][key]), key
        for key in ("position", "heading", "velocity", "angular_velocity", "activations", "time"):
            assert np.array_equal(getattr(probe.world, key), getattr(reference.world, key)), key
    assert probe._contact_steps > 0
    assert probe._path_length == reference._path_length
    assert probe._contact_impulse == reference._contact_impulse
    probe.hide_beacons()
    assert not any(probe.visible_beacon_rays().values())
    probe.restore_beacons()
    assert np.array_equal(probe.sensor_packet()["observation"]["ray_colors"],
                          reference.sensor_packet()["observation"]["ray_colors"])
    for frequency in (0, -1, 4000, float("nan")):
        try:
            probe.emit_tone(frequency)
        except ValueError:
            continue
        raise AssertionError("Invalid tone accepted")
    textured = ExperienceEnvironment("barrier", seed=8, texture_seed=44017)
    assert np.array_equal(textured.sensor_packet()["observation"]["ray_colors"],
                          np.asarray(textured.get_view_state()["ray_colors"]))
    return {"passed": True, "model_training": False, "teacher_actions": 0,
            "continuous_exposure_probe_seconds": continuous_elapsed,
            "joint_exposures": exposures, "actual_joint_frames": 72,
            "sound_only_probe_frames": 12, "same_world_identity_and_clock": True,
            "sensor_whitelist_exact": True, "sound_only_changes_only_pcm": True,
            "private_target_and_legacy_tone_map_do_not_change_packet": True,
            "arbitrary_777_hz_pcm_max_error": pcm_error, "exact_point6_stop": True,
            "invisible_pairing_rejected_without_mutation": True,
            "physical_and_observation_exact_comparison_frames": len(sequence),
            "physics_contact_frames": probe._contact_steps, "texture_renderer_matches_sensor": True,
            "no_automatic_reward_or_terminal": True,
            "limitation": "Stimulus exposure and physical integrity checks do not establish learning or navigation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "experience_environment_audit.json")
    args = parser.parse_args()
    if args.self_check:
        report = self_check()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
