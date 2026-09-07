"""A small, deterministic embodied world, without a behavioral controller.

This is an explicit *planar thruster body*, not a humanoid or walking model.
Four antagonistic muscle activations create two longitudinal thruster forces.
Forces change momentum; no policy action changes position or velocity directly.
The surrounding medium supplies linear/angular drag. Solid walls supply unilateral
normal impulses and Coulomb friction. Colored circles are visual beacons only.

Coordinates: metres, seconds, kilograms; heading=0 faces +x, positive turns CCW.
Actions: [left_forward, left_backward, right_forward, right_backward], each 0..1.
Ray order: rightmost (-60 deg) through forward (0) to leftmost (+60 deg).
Touch order: front, back, left, right. RGB channels are in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    width: float = 8.0
    height: float = 6.0
    radius: float = 0.18
    mass: float = 1.0
    max_muscle_force: float = 2.0  # newtons per thruster
    lever_arm: float = 0.12
    linear_drag: float = 1.8  # N per (m/s)
    angular_drag: float = 0.12  # N m per (rad/s)
    muscle_time_constant: float = 0.04
    physics_dt: float = 0.005
    ray_range: float = 5.0
    ray_angles: tuple[float, ...] = tuple(math.radians(x) for x in (-60, -30, 0, 30, 60))
    wall_color: tuple[float, float, float] = (0.35, 0.35, 0.35)
    restitution: float = 0.0
    friction: float = 0.25
    pain_impulse_scale: float = 0.25  # N s across one requested control interval


def _vec(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be {size} finite numbers")
    return result.copy()


class World:
    """World(config, walls, objects), reset(), step(action, dt), observe().

    Walls are (xmin, ymin, xmax, ymax), or dictionaries with a ``bounds`` key.
    Objects are dictionaries: ``position``=(x,y), ``radius``=.25,
    ``color``=(r,g,b), optional ``name``. They are visible, non-solid circles.
    The rectangular outer border is always solid and visible.
    """

    def __init__(self, config: WorldConfig | Mapping | None = None,
                 walls: Sequence | None = None, objects: Sequence | None = None):
        self.config = (WorldConfig(**config) if isinstance(config, Mapping)
                       else config if config is not None else WorldConfig())
        self._validate_config()
        self.inertia = 0.5 * self.config.mass * self.config.radius ** 2
        self.walls: list[dict] = []
        self.objects: list[dict] = []
        self.set_geometry(walls=walls if walls is not None else [],
                          objects=objects if objects is not None else [])
        self.reset()

    def _validate_config(self):
        c = self.config
        positive = (c.width, c.height, c.radius, c.mass, c.max_muscle_force,
                    c.physics_dt, c.ray_range, c.pain_impulse_scale)
        nonnegative = (c.lever_arm, c.linear_drag, c.angular_drag,
                       c.muscle_time_constant, c.friction)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("World dimensions, mass, timestep and force scales must be positive")
        if not all(np.isfinite(x) and x >= 0 for x in nonnegative):
            raise ValueError("Drag, lever arm, muscle lag and friction must be nonnegative")
        if not 0 <= c.restitution <= 1:
            raise ValueError("restitution must be in [0, 1]")
        if min(c.width, c.height) <= 2 * c.radius:
            raise ValueError("World must be larger than the body diameter")
        if not c.ray_angles or not np.isfinite(c.ray_angles).all():
            raise ValueError("At least one finite ray angle is required")
        color = _vec(c.wall_color, 3, "wall_color")
        if np.any((color < 0) | (color > 1)):
            raise ValueError("Colors must be in [0,1]")

    def set_geometry(self, *, walls: Sequence | None = None,
                     objects: Sequence | None = None):
        """Replace geometry; caller must reset to a non-overlapping initial pose."""
        if walls is not None:
            parsed = []
            for item in walls:
                item = item if isinstance(item, Mapping) else {"bounds": item}
                b = _vec(item["bounds"], 4, "wall bounds")
                if b[0] >= b[2] or b[1] >= b[3]:
                    raise ValueError("Wall bounds must have xmin<xmax and ymin<ymax")
                col = _vec(item.get("color", self.config.wall_color), 3, "wall color")
                if np.any((col < 0) | (col > 1)):
                    raise ValueError("Colors must be in [0,1]")
                parsed.append({"bounds": b, "color": col})
            self.walls = parsed
        if objects is not None:
            parsed = []
            for i, item in enumerate(objects):
                p = _vec(item["position"], 2, "object position")
                r = float(item.get("radius", 0.25))
                col = _vec(item.get("color", (1, 0, 0)), 3, "object color")
                if not np.isfinite(r) or r <= 0 or np.any((col < 0) | (col > 1)):
                    raise ValueError("Object radius must be positive and color in [0,1]")
                parsed.append({"position": p, "radius": r, "color": col,
                               "name": str(item.get("name", f"object_{i}"))})
            self.objects = parsed

    def reset(self, position: Sequence[float] | None = None, heading: float = 0.0,
              velocity: Sequence[float] = (0.0, 0.0), angular_velocity: float = 0.0,
              *, walls: Sequence | None = None, objects: Sequence | None = None):
        self.set_geometry(walls=walls, objects=objects)
        c = self.config
        self.position = _vec(position if position is not None else (c.width / 2, c.height / 2),
                             2, "position")
        self.velocity = _vec(velocity, 2, "velocity")
        if not np.isfinite(heading) or not np.isfinite(angular_velocity):
            raise ValueError("heading and angular_velocity must be finite")
        self.heading = float(heading)
        self.angular_velocity = float(angular_velocity)
        self.activations = np.zeros(4)
        self.last_action = np.zeros(4)
        self.time = 0.0
        self.pain = 0.0
        self.contact = False
        self.touch = np.zeros(4)
        self.contact_impulse = 0.0
        self.contacts: list[dict] = []
        if self._penetrations():
            raise ValueError("Initial body intersects a wall; choose a free initial pose")
        return self.observe()

    def step(self, muscles: Sequence[float], dt: float = 0.02):
        action = _vec(muscles, 4, "muscles")
        if np.any((action < 0) | (action > 1)):
            raise ValueError("Muscle activation must be between 0 and 1")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if dt > 1.0:
            raise ValueError("Use control intervals of at most one second")
        c = self.config
        self.last_action = action
        self.pain = 0.0
        self.touch[:] = 0.0
        self.contact_impulse = 0.0
        self.contacts = []
        self.contact = False
        count = max(1, int(math.ceil(dt / c.physics_dt)))
        h = dt / count
        # Prevent discrete collision tunneling. Subdivision adapts to momentum;
        # it never clamps or directly sets the body's commanded velocity.
        speed_bound = float(np.linalg.norm(self.velocity)) + 2 * c.max_muscle_force / c.mass * dt
        count = max(count, int(math.ceil(speed_bound * dt / (c.radius * 0.25))))
        h = dt / count
        for _ in range(count):
            blend = 1.0 if c.muscle_time_constant == 0 else -math.expm1(-h / c.muscle_time_constant)
            self.activations += blend * (action - self.activations)
            left = c.max_muscle_force * (self.activations[0] - self.activations[1])
            right = c.max_muscle_force * (self.activations[2] - self.activations[3])
            forward = np.array([math.cos(self.heading), math.sin(self.heading)])
            thrust = (left + right) * forward
            torque = c.lever_arm * (right - left)
            # Implicit linear drag dissipates energy at every supported step size.
            self.velocity = (self.velocity + h * thrust / c.mass) / (1 + h * c.linear_drag / c.mass)
            self.angular_velocity = ((self.angular_velocity + h * torque / self.inertia)
                                     / (1 + h * c.angular_drag / self.inertia))
            self.position += h * self.velocity
            self.heading = (self.heading + h * self.angular_velocity + math.pi) % (2 * math.pi) - math.pi
            self._resolve_contacts()
            self.time += h
        self.pain = float(np.clip(self.contact_impulse / c.pain_impulse_scale, 0, 1))
        if not np.isfinite(np.r_[self.position, self.velocity, self.heading, self.angular_velocity]).all():
            raise FloatingPointError("Non-finite world state")
        if self._penetrations(tolerance=1e-7):
            raise RuntimeError("Unresolved penetration; geometry may trap the body")
        return self.observe()

    def _penetrations(self, tolerance: float = 0.0):
        p, r, c = self.position, self.config.radius, self.config
        out = []
        for depth, n in ((r - p[0], (1, 0)), (p[0] + r - c.width, (-1, 0)),
                         (r - p[1], (0, 1)), (p[1] + r - c.height, (0, -1))):
            if depth > tolerance:
                out.append((float(depth), np.array(n, dtype=float)))
        for wall in self.walls:
            x0, y0, x1, y1 = wall["bounds"]
            closest = np.clip(p, (x0, y0), (x1, y1))
            delta = p - closest
            dist = float(np.linalg.norm(delta))
            if dist > 1e-12:
                if r - dist > tolerance:
                    out.append((r - dist, delta / dist))
            else:
                depths = (p[0] - x0 + r, x1 - p[0] + r, p[1] - y0 + r, y1 - p[1] + r)
                k = int(np.argmin(depths))
                normal = ((-1, 0), (1, 0), (0, -1), (0, 1))[k]
                if depths[k] > tolerance:
                    out.append((float(depths[k]), np.array(normal, dtype=float)))
        return out

    def _resolve_contacts(self):
        c = self.config
        for _ in range(12):
            hits = self._penetrations(tolerance=1e-12)
            if not hits:
                return
            # Recompute after each constraint correction to handle corners.
            depth, n = max(hits, key=lambda hit: hit[0])
            self.position += n * (depth + 1e-11)
            vn = float(self.velocity @ n)
            impulse = max(0.0, -(1 + c.restitution) * c.mass * vn)
            self.velocity += impulse / c.mass * n
            tangent = np.array([-n[1], n[0]])
            # Contact lies at -radius*n; tangential impulse also changes spin.
            vt = float(self.velocity @ tangent) - c.radius * self.angular_velocity
            tangent_mass_inv = 1 / c.mass + c.radius ** 2 / self.inertia
            jt = float(np.clip(-vt / tangent_mass_inv, -c.friction * impulse, c.friction * impulse))
            self.velocity += jt / c.mass * tangent
            self.angular_velocity -= c.radius * jt / self.inertia
            self.contact_impulse += impulse
            self.contact = True
            f = np.array([math.cos(self.heading), math.sin(self.heading)])
            l = np.array([-f[1], f[0]])
            side = -n  # direction from body centre towards the contacted wall
            self.touch = np.maximum(self.touch, np.maximum(0, [side @ f, -side @ f, side @ l, -side @ l]))
            self.contacts.append({"normal": n.tolist(), "normal_impulse": impulse,
                                  "tangent_impulse": jt})

    @staticmethod
    def _ray_box(origin: np.ndarray, direction: np.ndarray, bounds: np.ndarray):
        near, far = -math.inf, math.inf
        for k in range(2):
            lo, hi = bounds[k], bounds[k + 2]
            if abs(direction[k]) < 1e-12:
                if not lo <= origin[k] <= hi:
                    return math.inf
            else:
                a, b = (lo - origin[k]) / direction[k], (hi - origin[k]) / direction[k]
                near, far = max(near, min(a, b)), min(far, max(a, b))
        if far < max(near, 0):
            return math.inf
        return max(near, 0.0)

    @staticmethod
    def _ray_circle(origin: np.ndarray, direction: np.ndarray, centre: np.ndarray, radius: float):
        offset = origin - centre
        b, c = float(offset @ direction), float(offset @ offset - radius ** 2)
        if c <= 0:
            return 0.0
        disc = b * b - c
        if disc < 0:
            return math.inf
        near = -b - math.sqrt(disc)
        return near if near >= 0 else math.inf

    def _rays(self):
        c = self.config
        ds, cols, points, kinds = [], [], [], []
        for angle in c.ray_angles:
            direction = np.array([math.cos(self.heading + angle), math.sin(self.heading + angle)])
            # Camera at centre: ranges include the body radius, documented above.
            origin = self.position
            distance, color, kind = c.ray_range, np.zeros(3), "none"
            for k, limit in ((0, 0), (0, c.width), (1, 0), (1, c.height)):
                if abs(direction[k]) > 1e-12:
                    t = (limit - origin[k]) / direction[k]
                    if 0 <= t < distance:
                        distance, color, kind = float(t), np.array(c.wall_color), "wall"
            for wall in self.walls:
                t = self._ray_box(origin, direction, wall["bounds"])
                if t < distance:
                    distance, color, kind = float(t), wall["color"], "wall"
            for obj in self.objects:
                t = self._ray_circle(origin, direction, obj["position"], obj["radius"])
                if t < distance:
                    distance, color, kind = float(t), obj["color"], "object"
            ds.append(distance)
            cols.append(color.copy())
            points.append((origin + distance * direction).tolist())
            kinds.append(kind)
        return np.asarray(ds), np.asarray(cols), points, kinds

    def observe(self):
        ds, cols, _, _ = self._rays()
        f = np.array([math.cos(self.heading), math.sin(self.heading)])
        return {"time": float(self.time), "position": self.position.copy(),
                "heading": float(self.heading), "velocity": self.velocity.copy(),
                "angular_velocity": float(self.angular_velocity),
                "forward_speed": float(self.velocity @ f),
                "ray_angles": np.asarray(self.config.ray_angles).copy(),
                "ray_distances": ds, "ray_colors": cols,
                "ray_distances_normalized": ds / self.config.ray_range,
                "touch": self.touch.copy(), "pain": float(self.pain),
                "contact": bool(self.contact), "contact_impulse": float(self.contact_impulse),
                "muscle_activations": self.activations.copy()}

    def get_state(self):
        """JSON-serializable geometry/state for a renderer or an evaluator."""
        obs = self.observe()
        state = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        _, _, points, kinds = self._rays()
        state.update({"width": self.config.width, "height": self.config.height,
                      "radius": self.config.radius,
                      "walls": [{"bounds": w["bounds"].tolist(), "color": w["color"].tolist()}
                                for w in self.walls],
                      "objects": [{"position": o["position"].tolist(), "radius": o["radius"],
                                   "color": o["color"].tolist(), "name": o["name"]}
                                  for o in self.objects],
                      "ray_endpoints": points, "ray_hit_kinds": kinds,
                      "last_action": self.last_action.tolist(), "contacts": list(self.contacts)})
        return state


SimpleWorld = World
