"""Autonomous exploration and held-out goal combinations; evaluator stays outside brain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from collections import deque
import numpy as np

from developmental_navigation import DevelopmentalNavigator
from developmental_world import DevelopmentalWorld
from checkpoint import save_tree


HERE = Path(__file__).resolve().parent


def json_number(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported result value: ' + type(value).__name__)


def shortest_distance(world, start, target):
    # Evaluator ONLY. This is never called by a controller or used for training.
    seen = {tuple(start)}
    queue = deque([(tuple(start), 0)])
    while queue:
        p, distance = queue.popleft()
        if p == tuple(target):
            return distance
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + d[0], p[1] + d[1])
            if q not in seen and 0 <= q[0] < world.width and 0 <= q[1] < world.height and not world.grid[q[1], q[0]]:
                seen.add(q)
                queue.append((q, distance + 1))
    return None


def train(seed, capacity, steps, world_kwargs):
    world = DevelopmentalWorld(seed, **world_kwargs)
    brain = DevelopmentalNavigator(dict(seed=seed, capacity=capacity,
                                       visual_samples=world.patch_samples,
                                       visual_ray_count=world.rays))
    brain.runtime.set_time_ring_policy('grow')
    elapsed = []
    visits, pairs, actual = set(), set(), []
    command, trace = brain.observe(world.packet(), learning=True)
    for k in range(steps):
        start = time.perf_counter()
        before = tuple(world.position)
        packet = world.step(command)
        visits.add(tuple(world.position))
        pairs.add((before, tuple(world.position)))
        actual.append(dict(source=list(before), destination=world.position.tolist(),
                           command=command.tolist(), record_time=int(brain.runtime.clock.当前时间)))
        command, trace = brain.observe(packet, actual_muscles=command, learning=True)
        elapsed.append((time.perf_counter() - start) * 1000.)
    state = dict(brain=brain.snapshot(), world=world.snapshot())
    summary = dict(seed=seed, capacity=capacity, steps=steps, assemblies=len(brain.prototypes),
        relations=len(brain.edges), visited_places=len(visits), walkable_places=int((~world.grid).sum()),
        visited_beacons=[tuple(p) in visits for p in world.beacons],
        movements=sum(row['source'] != row['destination'] for row in actual),
        clock=int(brain.runtime.clock.当前时间), current_time_cells=int(brain.runtime.clock.时间激活.sum()),
        median_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed, 95)))
    return state, summary, actual


def _valid_starts(world, count=4):
    candidates = [(2, 2), (2, world.height - 3), (world.width - 3, 4), (4, world.height // 2)]
    starts = []
    for x, y in candidates:
        if 0 <= x < world.width and 0 <= y < world.height and not world.grid[y, x]:
            starts.append((x, y))
    if len(starts) < count:
        for y, x in np.argwhere(~world.grid):
            candidate = (int(x), int(y))
            if candidate not in starts and candidate not in world.beacons:
                starts.append(candidate)
            if len(starts) == count:
                break
    return starts[:count]


def evaluate(state, *, start=(2, 2), target=0, switch_at=None, switch_target=None,
             blocked=None, propagation=True, online=False, max_steps=100):
    brain = DevelopmentalNavigator.from_snapshot(state['brain'])
    world = DevelopmentalWorld.from_snapshot(state['world'])
    # Evaluator changes actual start/stimulus; no brain memory or learned
    # connections are reset. A frozen probe begins with an exogenous observation.
    world.position = np.array(start, int)
    world.velocity[:] = 0
    world.touch[:] = 0
    world.local_sounds = False
    world.cue = world.tones[target]
    brain.pending = None
    command, first = brain.observe(world.packet(), learning=False, pursue=True, propagation=propagation)
    world.cue = None
    trials = [first]
    path = [list(start)]
    target_before = world.beacons[target]
    expected = world.beacons[switch_target] if switch_target is not None else target_before
    distance = shortest_distance(world, start, expected)
    switched = False
    blocked_applied = False
    for k in range(max_steps):
        if switch_at is not None and k == switch_at:
            world.cue = world.tones[switch_target]
            # A new real external cue is an actual observation. Previous
            # proposed action was not executed, so it cannot become experience.
            brain.pending = None
            command, trace = brain.observe(world.packet(), learning=False, pursue=True, propagation=propagation)
            trials.append(trace)
            world.cue = None
            switched = True
        if blocked is not None and k == blocked[0]:
            bx, by = blocked[1]
            if (bx, by) != tuple(world.position):
                world.grid[by, bx] = True
                blocked_applied = True
        packet = world.step(command)
        command, trace = brain.observe(packet, actual_muscles=command, learning=online,
                                      pursue=True, propagation=propagation)
        trials.append(trace)
        path.append(world.position.tolist())
        if tuple(world.position) == expected and (switch_at is None or switched):
            break
    return dict(start=list(start), target=target, switch_at=switch_at, switch_target=switch_target,
        propagation=propagation, online=online, block=blocked, block_applied=blocked_applied,
        success=tuple(world.position) == expected and (switch_at is None or switched),
        steps=len(path)-1, evaluator_shortest_distance=distance, path=path, traces=trials,
        collisions=sum(row['collision'] for row in world.history),
        final_clock=int(brain.runtime.clock.当前时间), learning_steps_before=state['brain']['learning_steps'],
        learning_steps_after=brain.learning_steps,
        assemblies_after=len(brain.prototypes), relations_after=len(brain.edges))


def shortest_path(world, start, target):
    # Returned for testing; includes both start and target.
    seen = {tuple(start): None}
    queue = deque([tuple(start)])
    while queue:
        p = queue.popleft()
        if p == tuple(target):
            path = [p]
            while seen[p] is not None:
                p = seen[p]
                path.append(p)
            return list(reversed(path))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (p[0] + dx, p[1] + dy)
            if q not in seen and 0 <= q[0] < world.width and 0 <= q[1] < world.height and not world.grid[q[1], q[0]]:
                seen[q] = p
                queue.append(q)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 83, 97])
    parser.add_argument('--capacity', type=int, default=128)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--world-width', type=int, default=11)
    parser.add_argument('--world-height', type=int, default=9)
    parser.add_argument('--world-rays', type=int, default=31)
    parser.add_argument('--world-patch-samples', type=int, default=9)
    parser.add_argument('--world-obstacle-complexity', type=float, default=0.08)
    parser.add_argument('--name', default='developmental_navigation')
    args = parser.parse_args()
    results = HERE / 'results'
    results.mkdir(exist_ok=True)
    protected = HERE / 'results' / 'living_original.npz'
    previous_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
    summary = []
    world_kwargs = dict(width=args.world_width, height=args.world_height, rays=args.world_rays,
                       patch_samples=args.world_patch_samples, obstacle_complexity=args.world_obstacle_complexity)
    for seed in args.seeds:
        state, training, actual = train(seed, args.capacity, args.steps, world_kwargs)
        save_tree(results / f'{args.name}_{seed}.npz', state)
        print(json.dumps(dict(training=training), ensure_ascii=False), flush=True)
        outcomes = []
        world = DevelopmentalWorld(seed, **world_kwargs)
        for start in _valid_starts(world, count=4):
            for goal in range(3):
                outcomes.append(evaluate(state, start=start, target=goal))
        starts = _valid_starts(world, count=1)
        start_for_switch = starts[0] if starts else (2, 4)
        route_block = shortest_path(world, start_for_switch, tuple(world.beacons[0]))
        blocked_cell = route_block[1] if len(route_block) > 1 else tuple(world.beacons[0])
        switches = [evaluate(state, start=start_for_switch, target=a, switch_at=3, switch_target=b)
                    for a, b in ((0, 1), (1, 0))]
        lesions = [evaluate(state, start=start_for_switch, target=i, propagation=False)
                   for i in range(3)]
        blocked = [evaluate(state, start=start_for_switch, target=0, blocked=(3, blocked_cell), online=True)]
        report = dict(training=training, autonomous_records=actual, navigation=outcomes,
                      switches=switches, propagation_lesion=lesions, blocked_route=blocked)
        (results / f'{args.name}_{seed}.json').write_text(json.dumps(report, ensure_ascii=False,
            allow_nan=False, separators=(',', ':'), default=json_number), encoding='utf-8')
        compact = dict(**training, navigation=sum(row['success'] for row in outcomes),
            navigation_trials=len(outcomes), switches=sum(row['success'] for row in switches),
            switch_trials=len(switches), lesion_success=sum(row['success'] for row in lesions),
            lesion_trials=len(lesions), blocked=sum(row['success'] for row in blocked))
        summary.append(compact)
        print(json.dumps(compact, ensure_ascii=False), flush=True)
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == previous_hash
    (results / f'{args.name}_summary.json').write_text(json.dumps(dict(results=summary,
        original_life_unchanged=True, original_life_sha256=previous_hash,
        limits=['Engineered new PFC propagation rule, not spontaneous original-PFC emergence',
                'New lattice body with downward texture camera; no continuous-body transfer proof',
                'Start-goal combinations were not trained as commanded routes; local transitions were explored',
                'Goal change is a previously paired tone, not natural-language understanding']),
        ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
