"""Optional visible wall texture; not enabled in the formal experiment or UI.

The existing world computes visibility and ray intersections first. Only actual
wall hits receive fixed smooth pigment sampled at that surface point. Objects,
range, collision, route scoring, audio and controller packet keys are unchanged.
This adds visible environmental landmarks; it does not add coordinates or a map
to the brain, and it does not establish improved learning or spatial reasoning.

``TexturedRouteEnv(texture_seed=..., texture_enabled=False)`` is an exact sensory
control using the original wall colors. Texture seed has its own RNG and is
independent of environment seed, instructions, time and target identity.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from environment import _PhysicsWorld, SENSOR_KEYS, BEACON_COLORS
from route_switch_environment import RouteSwitchEnv


class SmoothWallPigment:
    """Local C1 interpolation of independent random paint samples, in [0,1].

    Two grid scales make patches less repetitive. Each scale uses only the four
    neighboring grid samples. No semantic labels, wall IDs, goal colors, route
    directions, distance-to-goal or camera pose participate in this function.
    """

    def __init__(self, width, height, seed=44017, *, cell_sizes=(.9, .34)):
        self.width, self.height = float(width), float(height)
        self.seed = int(seed)
        self.cell_sizes = tuple(float(s) for s in cell_sizes)
        if len(self.cell_sizes) != 2 or not all(np.isfinite(s) and s > 0 for s in self.cell_sizes):
            raise ValueError("Two finite positive texture cell sizes are required")
        rng = np.random.default_rng(self.seed)
        self.grids = [rng.uniform(-1., 1., (math.ceil(self.width/s)+2,
                      math.ceil(self.height/s)+2, 3)) for s in self.cell_sizes]

    def sample(self, points):
        """Renderer-only world-space points -> RGB pigment, never model input."""
        p = np.asarray(points, dtype=float)
        single = p.shape == (2,)
        p = p.reshape(-1, 2) if single else p
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("Texture points must have finite shape (N,2) or (2,)")
        if (p < -1e-7).any() or (p[:, 0] > self.width+1e-7).any() or (p[:, 1] > self.height+1e-7).any():
            raise ValueError("Texture sample must lie on the finite world's surfaces")
        p = np.clip(p, (0., 0.), (self.width, self.height))
        value = np.zeros((len(p), 3))
        for weight, size, grid in zip((.75, .25), self.cell_sizes, self.grids):
            q = p / size
            index = np.floor(q).astype(int)
            fraction = q-index
            fraction = fraction*fraction*(3.-2.*fraction)
            u, v = fraction[:, 0, None], fraction[:, 1, None]
            ix, iy = index[:, 0], index[:, 1]
            interpolated = ((1-u)*(1-v)*grid[ix, iy] + u*(1-v)*grid[ix+1, iy]
                            + (1-u)*v*grid[ix, iy+1] + u*v*grid[ix+1, iy+1])
            value += weight*interpolated
        rgb = .46 + .28*value  # bounded .18..74; avoids saturated beacon colors
        return rgb[0] if single else rgb


class _TexturedPhysicsWorld(_PhysicsWorld):
    """Rendering extension; integration/contact methods remain inherited."""

    @classmethod
    def from_world(cls, world, pigment, enabled):
        # Preserve every existing physical field and RNG-bearing object. The
        # old world object has no external references during environment reset.
        instance = cls.__new__(cls)
        instance.__dict__ = world.__dict__.copy()
        instance.wall_pigment = pigment
        instance.texture_enabled = bool(enabled)
        return instance

    def _rays(self):
        distances, colors, points, kinds = super()._rays()
        if self.texture_enabled:
            wall = np.asarray([kind == "wall" for kind in kinds], dtype=bool)
            if np.any(wall):
                colors[wall] = self.wall_pigment.sample(np.asarray(points)[wall])
        return distances, colors, points, kinds


class TexturedRouteEnv(RouteSwitchEnv):
    """Opt-in candidate environment with the unchanged six-field sensor packet."""

    def __init__(self, *, texture_seed=44017, texture_enabled=True, **kwargs):
        self.texture_seed = int(texture_seed)
        self.texture_enabled = bool(texture_enabled)
        self._wall_pigment = None
        self._texture_surface_cache = None
        super().__init__(**kwargs)

    def reset(self, *args, texture_seed=None, texture_enabled=None, **kwargs):
        if texture_seed is not None and int(texture_seed) != self.texture_seed:
            self.texture_seed = int(texture_seed)
            self._wall_pigment = None
        if texture_enabled is not None:
            self.texture_enabled = bool(texture_enabled)
        super().reset(*args, **kwargs)
        if self._wall_pigment is None:
            self._wall_pigment = SmoothWallPigment(self.scene.width, self.scene.height, self.texture_seed)
        self.world = _TexturedPhysicsWorld.from_world(self.world, self._wall_pigment, self.texture_enabled)
        self._texture_surface_cache = None
        return self.sensor_packet()

    def set_texture_enabled(self, enabled):
        """Evaluator-only visual intervention; no reset or physical change."""
        self.texture_enabled = bool(enabled)
        self.world.texture_enabled = self.texture_enabled

    def wall_texture_colors(self, points):
        """For drawing the actual paint on wall surfaces; not a brain API."""
        return self._wall_pigment.sample(points)

    def wall_texture_surfaces(self, spacing=.08):
        """Private renderer polylines with per-segment sampled surface colors.

        A new renderer can draw these over the wall outlines. Existing renderers
        already show matching ray colors and receptor strips via get_view_state.
        They do not automatically paint the solid wall rectangles themselves.
        """
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be positive")
        edges = []
        for x0, y0, x1, y1 in ((0., 0., self.scene.width, self.scene.height), *self.scene.walls):
            edges.extend((((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                          ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))))
        surfaces = []
        for a, b in edges:
            a, b = np.asarray(a), np.asarray(b)
            count = max(1, math.ceil(np.linalg.norm(b-a)/spacing))
            points = a[None, :] + np.linspace(0., 1., count+1)[:, None]*(b-a)[None, :]
            colors = self.wall_texture_colors((points[:-1]+points[1:])/2)
            surfaces.append(dict(points=points.tolist(), colors=colors.tolist()))
        return surfaces

    def get_view_state(self):
        state = super().get_view_state()
        if self._texture_surface_cache is None:
            self._texture_surface_cache = self.wall_texture_surfaces()
        state.update(texture_enabled=self.texture_enabled, texture_seed=self.texture_seed,
                     wall_texture_surfaces=copy.deepcopy(self._texture_surface_cache) if self.texture_enabled else [])
        return state

    def private_metrics(self, **kwargs):
        result = super().private_metrics(**kwargs)
        result.update(texture_enabled=self.texture_enabled, texture_seed=self.texture_seed,
                      texture_semantics="Fixed visible pigment at actual wall ray hits; no additional brain fields")
        return result


def self_check():
    result = {}
    base = RouteSwitchEnv(seed=76)
    textured = TexturedRouteEnv(seed=76, texture_seed=44017)
    a = base.reset(route="bottom")
    b = textured.reset(route="bottom")
    assert set(b) == {"observation", "waveform"} and set(b["observation"]) == SENSOR_KEYS
    assert b["observation"]["ray_colors"].shape == (31, 3)
    assert not np.array_equal(a["observation"]["ray_colors"], b["observation"]["ray_colors"])
    for key in SENSOR_KEYS-{"ray_colors"}:
        assert np.array_equal(a["observation"][key], b["observation"][key])
    assert np.array_equal(a["waveform"], b["waveform"])
    # Every color derives from the exact same ray endpoint and hit classification
    # that the viewer sees. This includes objects occluding more distant walls.
    textured.reset(start=(7., 1.5, 0.), route="bottom")
    view = textured.get_view_state()
    packet = textured.sensor_packet()
    assert np.array_equal(packet["observation"]["ray_colors"], view["ray_colors"])
    distances, colors, endpoints, kinds = textured.world._rays()
    assert "object" in kinds and "wall" in kinds
    for i, kind in enumerate(kinds):
        expected = (textured.wall_texture_colors(endpoints[i]) if kind == "wall" else
                    next(np.asarray(col) for col in BEACON_COLORS if np.array_equal(colors[i], col)))
        assert np.array_equal(colors[i], expected)
    result["only_actual_wall_hits_colored_objects_occlude_normally"] = True
    result["model_and_viewer_ray_rgb_identical"] = True

    # Same physical surface point viewed from two different valid camera poses.
    endpoint = np.array([4.8, 3.0])
    observed = []
    for position in (np.array([2., 2.]), np.array([3., 3.5])):
        direction = endpoint-position
        heading = math.atan2(direction[1], direction[0])
        textured.reset(start=(*position, heading), route="bottom")
        _, cols, ends, hits = textured.world._rays()
        assert hits[15] == "wall" and np.linalg.norm(np.asarray(ends[15])-endpoint) < 1e-12
        observed.append(cols[15])
    assert np.allclose(*observed, atol=1e-12, rtol=0)
    p = np.array([[4.8, y] for y in np.linspace(1.1, 6.19, 510)])
    pigment = textured.wall_texture_colors(p)
    max_delta = float(np.max(np.abs(np.diff(pigment, axis=0))))
    assert max_delta < .03
    assert pigment.min() >= .18 and pigment.max() <= .74
    # Boundary derivatives match: cubic smoothstep does not jump at grid edges.
    for boundary in (.9, 1.8, 2.7, .34, .68):
        adjacent = textured.wall_texture_colors(np.array([[0., boundary-1e-7], [0., boundary+1e-7]]))
        assert np.max(np.abs(adjacent[1]-adjacent[0])) < 1e-6
    result["same_surface_color_across_camera_views"] = True
    result["continuous_local_pigment"] = {"maximum_RGB_change_per_centimetre": max_delta,
                                         "observed_min": float(pigment.min()), "observed_max": float(pigment.max())}
    levels = np.linspace(1/17, 16/17, 16, dtype=np.float32)
    quantized = np.sum(pigment.astype(np.float32)[:, :, None] >= levels[None, None, :], axis=2)
    unique_rgb_patterns = len(np.unique(quantized, axis=0))
    assert unique_rgb_patterns > 10
    result["unchanged_16_threshold_color_receptors"] = {
        "samples_on_one_fixed_wall": len(pigment),
        "original_uniform_wall_RGB_patterns": 1,
        "textured_wall_RGB_patterns": unique_rgb_patterns,
        "meaning": "Visible color distinctions survive existing quantization; this is not a full-observation ambiguity or learning test"}

    # Start both RNG streams with identical reset histories for the control.
    base = RouteSwitchEnv(seed=76)
    textured = TexturedRouteEnv(seed=76, texture_seed=44017, texture_enabled=False)
    base.reset(route="bottom")
    textured.reset(route="bottom")
    for key in SENSOR_KEYS:
        assert np.array_equal(base.sensor_packet()["observation"][key], textured.sensor_packet()["observation"][key])
    assert np.array_equal(base.sensor_packet()["waveform"], textured.sensor_packet()["waveform"])
    textured.set_texture_enabled(True)
    # Independent texture construction/reset does not consume environment RNG.
    assert base.rng.bit_generator.state == textured.rng.bit_generator.state
    actions = np.random.default_rng(332).uniform(0., .75, (120, 4))
    for frame, action in enumerate(actions):
        if frame == 70:
            base.emit_route_cue("top")
            textured.emit_route_cue("top")
        a, b = base.step(action), textured.step(action)
        for key in SENSOR_KEYS-{"ray_colors"}:
            assert np.array_equal(a["observation"][key], b["observation"][key])
        assert np.array_equal(a["waveform"], b["waveform"])
        for name in ("position", "velocity", "heading", "angular_velocity", "time", "activations", "last_action", "contacts"):
            assert np.array_equal(getattr(base.world, name), getattr(textured.world, name))
        base_metrics, texture_metrics = base.private_metrics(), textured.private_metrics()
        assert all(texture_metrics[key] == value for key, value in base_metrics.items())
    result["120_identical_actions_exact_physics_audio_and_route_score"] = True
    result["disabled_texture_exact_original_packet"] = True

    alternate = copy.deepcopy(textured)
    before = alternate.sensor_packet()
    alternate.emit_route_cue("bottom")
    after = alternate.sensor_packet()
    for key in SENSOR_KEYS:
        assert np.array_equal(before["observation"][key], after["observation"][key])
    changed_seed = TexturedRouteEnv(seed=76, texture_seed=44018)
    points = np.array([[4.8, 2.], [4.8, 3.], [4.8, 4.]])
    assert not np.array_equal(changed_seed.wall_texture_colors(points), textured.wall_texture_colors(points))
    result["route_change_cannot_change_visible_pigment"] = True
    result["independent_texture_seed_changes_only_paint"] = True
    result["renderer_surface_segments_available"] = sum(len(s["colors"]) for s in textured.wall_texture_surfaces())
    result["limit"] = "Adds visible local cues at unchanged 31 rays and RGB thresholds; learning improvement is untested. Formal experiment and viewer defaults remain unchanged."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/'results'/'textured_environment_audit.json')
    args = parser.parse_args()
    result = self_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
