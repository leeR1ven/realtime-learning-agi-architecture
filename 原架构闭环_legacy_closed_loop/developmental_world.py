"""Evaluator-owned discrete sensorimotor world for developmental navigation.

The first part of the sensor is a local floor texture patch and the remainder
are panoramic wall-distance rays.  The layout is parameterized so tests can scale
world size, local patch resolution, and obstacle complexity while preserving the
original protocol that external cues are not embedded into state labels.
"""
from __future__ import annotations

import copy
import math
import numpy as np


class DevelopmentalWorld:
    def __init__(self, seed=71, width=11, height=9, rays=31, patch_samples=9,
                 obstacle_complexity=0.08):
        self.seed = int(seed)
        self.width = int(width)
        self.height = int(height)
        self.rays = int(rays)
        self.patch_samples = int(patch_samples)
        self.obstacle_complexity = float(obstacle_complexity)
        if not (7 <= self.width and 7 <= self.height):
            raise ValueError('World width/height must be at least 7')
        if not (8 <= self.rays <= 512):
            raise ValueError('Rays must be between 8 and 512')
        if not (1 <= self.patch_samples <= self.rays):
            raise ValueError('Patch samples must be at least 1 and not exceed ray count')
        if not (0 <= self.obstacle_complexity <= 1):
            raise ValueError('obstacle_complexity must be in [0, 1]')

        self.patch_side = int(math.isqrt(self.patch_samples))
        if self.patch_side ** 2 != self.patch_samples:
            raise ValueError('patch_samples must be a perfect square')
        rng = np.random.default_rng(seed)
        self.grid = np.ones((self.height, self.width), bool)
        self.grid[1:-1, 1:-1] = False
        self._carve_obstacles(rng)
        self.wall_pigment = rng.uniform(.12, .88, (self.height, self.width, 3))
        self.pigment = rng.uniform(.12, .88, (self.height, self.width, self.patch_samples, 3))
        self.colors = np.array([[1., .04, .04], [.04, .04, 1.], [.04, 1., .04]])
        self.tones = [220., 440., 880.]
        self._place_beacons(rng)
        self.position = np.array(self._pick_start(), int)
        # Each world's motor wiring is fixed at birth, but hidden from brain.
        self.wiring = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])[rng.permutation(4)]
        self.velocity = np.zeros(2)
        self.touch = np.zeros(4)
        self.cue = None
        self.local_sounds = True
        self.ticks = 0
        self.history = []

    def _carve_obstacles(self, rng):
        if self.obstacle_complexity <= 0.:
            return
        interior_y = np.arange(1, self.height - 1)
        interior_x = np.arange(1, self.width - 1)
        yx = [(int(x), int(y)) for y in interior_y for x in interior_x]
        interior = len(yx)
        obstacle_count = int(interior * self.obstacle_complexity)
        for x, y in rng.permutation(yx)[:obstacle_count]:
            self.grid[y, x] = True
        # Add a coarse vertical barrier family for richer geometry; keep exits by carving.
        barrier_count = min(2, (self.width - 1) // 4)
        for axis in range(barrier_count):
            x = 2 + axis * (self.width - 4) // max(1, barrier_count - 1)
            if 1 < x < self.width - 2:
                breaks = rng.random(self.height - 2) > 0.55
                self.grid[1:self.height - 1, x] |= breaks
                self.grid[1, x] = False
                self.grid[self.height - 2, x] = False

    def _place_beacons(self, rng):
        opens = np.argwhere(~self.grid)
        opens = [(int(x), int(y)) for y, x in opens]
        if len(opens) < 9:
            raise ValueError('World has insufficient open cells for beacons and exploration')
        starts = sorted(opens, key=lambda p: self._manhattan(p, (self.width / 2, self.height / 2)))
        # Keep a center-like start candidate while forcing beacons to be spread apart.
        if starts:
            start = starts[0]
        else:
            raise ValueError('No open cells were produced')
        beacons = []
        remaining = [p for p in opens if p != start]
        for _ in range(3):
            if not remaining:
                raise ValueError('Failed to place required beacons')
            if not beacons:
                beacon = remaining[rng.integers(len(remaining))]
            else:
                beacon = max(remaining, key=lambda c: min(self._manhattan(c, b) for b in beacons))
            beacons.append(beacon)
            remaining.remove(beacon)
            self._force_connectivity(start, beacon)

        self.beacons = beacons

    def _pick_start(self):
        opens = np.argwhere(~self.grid)
        if opens.size == 0:
            raise ValueError('No open cells in world')
        cx, cy = self.width // 2, self.height // 2
        x, y = min(((int(x), int(y)) for y, x in opens),
                   key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        return x, y

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    @staticmethod
    def _reconstruct_path(parents, target):
        path = [target]
        while target in parents:
            target = parents[target]
            path.append(target)
        path.reverse()
        return path

    def _shortest_path(self, start, target):
        parents = {}
        queue = [(int(start[0]), int(start[1]))]
        target = (int(target[0]), int(target[1]))
        visited = {queue[0]}
        while queue:
            p = queue.pop(0)
            if p == target:
                return self._reconstruct_path(parents, p)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + dx, p[1] + dy)
                if not (0 <= q[0] < self.width and 0 <= q[1] < self.height):
                    continue
                if q in visited or self.grid[q[1], q[0]]:
                    continue
                visited.add(q)
                parents[q] = p
                queue.append(q)
        return []

    def _force_connectivity(self, start, target):
        path = self._shortest_path(start, target)
        if path:
            return
        x, y = start
        tx, ty = target
        step = 0
        while (x, y) != (tx, ty) and step < self.width * self.height * 2:
            step += 1
            moves = []
            if x != tx:
                moves.append((1 if tx > x else -1, 0))
            if y != ty:
                moves.append((0, 1 if ty > y else -1))
            if x == tx:
                moves.append((0, 1 if ty > y else -1))
            if y == ty:
                moves.append((1 if tx > x else -1, 0))
            dx, dy = moves[step % len(moves)]
            x = max(1, min(self.width - 2, x + dx))
            y = max(1, min(self.height - 2, y + dy))
            self.grid[y, x] = False

    def _local_patch(self):
        patch = self.pigment[self.position[1], self.position[0]]
        if patch.shape[0] < self.patch_samples:
            pad = np.zeros((self.patch_samples - patch.shape[0], 3))
            patch = np.concatenate((patch, pad), axis=0)
        elif patch.shape[0] > self.patch_samples:
            patch = patch[:self.patch_samples]
        return patch

    def _rays(self):
        x, y = self.position
        rgb = np.zeros((self.rays, 3))
        depth = np.ones(self.rays)
        rgb[:self.patch_samples] = self._local_patch()
        for i, beacon in enumerate(self.beacons):
            if tuple(self.position) == beacon:
                tint = self.colors[i]
                indices = np.unique(np.linspace(self.patch_samples - 3, self.patch_samples - 1, 3,
                                               dtype=int, endpoint=True))
                indices = [int(idx) for idx in indices if 0 <= idx < self.patch_samples]
                for index in indices:
                    rgb[index] = .35 * rgb[index] + .65 * tint
        panorama = self.rays - self.patch_samples
        if panorama > 0:
            for i, angle in enumerate(np.linspace(-math.pi, math.pi, panorama, endpoint=False),
                                     self.patch_samples):
                direction = np.array([math.cos(angle), math.sin(angle)])
                for distance in np.arange(.12, max(self.width, self.height) + 1, .08):
                    q = np.rint(self.position + direction * distance).astype(int)
                    if not (0 <= q[0] < self.width and 0 <= q[1] < self.height):
                        break
                    if self.grid[q[1], q[0]]:
                        depth[i] = distance
                        rgb[i] = self.wall_pigment[q[1], q[0]]
                        break
        return rgb, depth

    def packet(self):
        rgb, distances = self._rays()
        tone = self.cue
        if tone is None and self.local_sounds:
            for i, p in enumerate(self.beacons):
                if tuple(self.position) == p:
                    tone = self.tones[i]
        wave = np.zeros(800)
        if tone is not None:
            times = self.ticks * .1 + np.arange(800) / 8000.
            wave = .45 * np.sin(2 * math.pi * tone * times)
        return dict(observation=dict(ray_colors=rgb, ray_distances=distances,
            velocity=self.velocity.copy(), angular_velocity=np.array(0.),
            pain=np.array(float(self.touch.any())), touch=self.touch.copy()),
            waveform=wave)

    def step(self, muscles):
        muscles = np.asarray(muscles, float)
        if muscles.shape != (4,) or not np.isfinite(muscles).all() or np.any((muscles < 0) | (muscles > 1)):
            raise ValueError('Four finite 0..1 motor controls required')
        old = self.position.copy()
        self.touch[:] = 0
        if muscles.max() > .5:
            action = int(np.argmax(muscles))
            destination = self.position + self.wiring[action]
            if self.grid[destination[1], destination[0]]:
                self.touch[action] = 1.
            else:
                self.position = destination
        self.velocity = (self.position - old).astype(float)
        self.ticks += 1
        self.history.append(dict(tick=self.ticks, position=self.position.tolist(),
                                 muscles=muscles.tolist(), collision=bool(self.touch.any())))
        return self.packet()

    def snapshot(self):
        return {k: copy.deepcopy(v) for k, v in vars(self).items() if k != 'history'}

    @classmethod
    def from_snapshot(cls, state):
        obj = cls(state['seed'])
        for k, v in state.items():
            setattr(obj, k, copy.deepcopy(v))
        obj.beacons = [tuple(p) for p in obj.beacons]
        obj.history = []
        return obj
