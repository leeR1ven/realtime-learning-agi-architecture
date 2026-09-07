"""Privileged demonstration teacher; NEVER a component of the learned brain.

The teacher uses simulator geometry to plan demonstrations. Only the sensory
packet and actually applied four-muscle vector become learning experiences.
Autonomous evaluation must not construct this class or call its planner.
"""
from __future__ import annotations

import heapq
import math
import numpy as np


class MuscleTeacher:
    def __init__(self, world, destination, *, spacing=.22):
        self.world = world
        self.destination = np.asarray(destination, dtype=float).copy()
        self.spacing = float(spacing)
        self.clearance = world.config.radius + .12
        self.path = self._plan()
        self.cursor = 0

    def _free(self, point):
        x, y = point
        c, r = self.world.config, self.clearance
        if not (r <= x <= c.width-r and r <= y <= c.height-r):
            return False
        for wall in self.world.walls:
            x0, y0, x1, y1 = wall['bounds']
            if x0-r <= x <= x1+r and y0-r <= y <= y1+r:
                return False
        return True

    def _visible(self, a, b):
        n = max(2, int(np.linalg.norm(b-a) / .06)+1)
        return all(self._free(a*(1-t)+b*t) for t in np.linspace(0, 1, n))

    def _plan(self):
        c = self.world.config
        points = {}
        for ix, x in enumerate(np.arange(self.clearance, c.width-self.clearance+.001, self.spacing)):
            for iy, y in enumerate(np.arange(self.clearance, c.height-self.clearance+.001, self.spacing)):
                point = np.array([x, y])
                if self._free(point):
                    points[ix, iy] = point
        start_position = self.world.position.copy()
        start = min(points, key=lambda k: np.linalg.norm(points[k]-start_position))
        goal = min(points, key=lambda k: np.linalg.norm(points[k]-self.destination))
        queue = [(0., start)]
        cost, parent = {start: 0.}, {}
        while queue:
            _, node = heapq.heappop(queue)
            if node == goal:
                break
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                nxt = (node[0]+dx, node[1]+dy)
                if nxt not in points:
                    continue
                if dx and dy and ((node[0]+dx, node[1]) not in points or (node[0], node[1]+dy) not in points):
                    continue
                new_cost = cost[node] + self.spacing * math.hypot(dx,dy)
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt], parent[nxt] = new_cost, node
                    heuristic = np.linalg.norm(points[nxt]-points[goal])
                    heapq.heappush(queue, (new_cost+heuristic, nxt))
        if goal not in cost:
            raise ValueError('Demonstration teacher found no route with body clearance')
        path = [self.destination.copy(), points[goal]]
        while goal != start:
            goal = parent[goal]
            path.append(points[goal])
        path.append(start_position)
        path.reverse()
        # Visibility simplification belongs exclusively to the teacher.
        simplified = [path[0]]
        current = 0
        while current < len(path)-1:
            following = current+1
            for candidate in range(current+2, len(path)):
                if self._visible(path[current], path[candidate]):
                    following = candidate
            simplified.append(path[following])
            current = following
        return simplified[1:]

    def muscles(self):
        w = self.world
        while self.cursor < len(self.path)-1 and np.linalg.norm(w.position-self.path[self.cursor]) < .28:
            self.cursor += 1
        delta = self.path[self.cursor]-w.position
        distance = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0])
        angle = (desired_heading-w.heading+math.pi) % (2*math.pi)-math.pi
        omega = float(np.clip(2.5*angle, -1.8, 1.8))
        speed = .65 * max(0., math.cos(angle))**3
        if abs(angle) > .65:
            speed = 0.
        if self.cursor == len(self.path)-1:
            speed *= min(1., distance/.5)
            if distance < .16:
                omega = 0.
                speed = 0.
        forward = np.array([math.cos(w.heading), math.sin(w.heading)])
        vf = float(w.velocity @ forward)
        thrust = w.config.linear_drag*speed + w.config.mass*(speed-vf)/.35
        torque = w.config.angular_drag*omega + w.inertia*(omega-w.angular_velocity)/.2
        differential = torque/max(w.config.lever_arm, 1e-9)
        left = np.clip((thrust-differential)/2/w.config.max_muscle_force, -1, 1)
        right = np.clip((thrust+differential)/2/w.config.max_muscle_force, -1, 1)
        return np.array([max(left,0.), max(-left,0.), max(right,0.), max(-right,0.)])


def _self_check():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '闭环仿真'))
    from world import World
    w = World()
    w.reset(position=(1.1,3.), heading=0., walls=[(3.5, 1.2, 4.2, 4.8)])
    target = np.array([6.8,3.])
    teacher = MuscleTeacher(w, target)
    contacts = 0
    for step in range(500):
        w.step(teacher.muscles(), .1)
        contacts += w.contact
        if np.linalg.norm(w.position-target) < .3:
            break
    result = {'teacher_only': True, 'steps': step+1, 'distance': float(np.linalg.norm(w.position-target)),
              'contacts': contacts, 'waypoints': [p.tolist() for p in teacher.path]}
    print(result)
    assert result['distance'] < .3
    assert contacts == 0


if __name__ == '__main__':
    _self_check()
