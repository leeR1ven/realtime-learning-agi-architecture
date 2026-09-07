"""Small independent capacity, chronology and safe-resume checks.

Artificial neural patterns check storage mechanics; they are never written to
the user's living model and are not reported as navigation or intelligence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from checkpoint import save_tree, load_tree
from growing_time_ring import GrowingTimeRing
from original_runtime import OriginalRuntime, ROOT, SOURCE_FILES


HERE = Path(__file__).resolve().parent


def equal(left, right):
    if isinstance(left, dict):
        return (isinstance(right, dict) and list(left) == list(right)
                and all(equal(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    return type(left) is type(right) and left == right


def pattern(index, width=64):
    value = np.zeros(width, dtype=bool)
    value[index % width] = True
    return value


def runtime(capacity, steps=2, policy='grow'):
    return OriginalRuntime(64, 64, 64, 64, time_neurons=capacity,
                           steps_per_frame=steps, time_ring_policy=policy)


def learn(obj, index, learning=True):
    value = pattern(index)
    obj.pfc_microstep(value, 4. * value, diagnostics=False)
    return obj.record(value, value, value, value, learning=learning, index_pfc_bool=value)


def main():
    protected = [ROOT / name for name in SOURCE_FILES] + [HERE / 'results/living_original.npz']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    growth_cases = []
    for capacity, steps in ((1, 1), (2, 2), (3, 2), (5, 3), (8, 9)):
        obj = runtime(capacity, steps)
        capacities, old_rows = [capacity], None
        for frame in range(36):
            info = learn(obj, frame)
            assert info['index_time'] == frame * steps
            assert info['time_steps'] == [(frame * steps + j, frame * steps + j + 1) for j in range(steps)]
            assert obj.clock.当前时间 == (frame + 1) * steps
            assert int(obj.clock.时间激活.sum()) == 1
            assert obj.clock.时间激活[obj.clock.当前时间]
            if capacities[-1] != obj.clock.时间神经元数量:
                capacities.append(obj.clock.时间神经元数量)
            if frame == 4:
                old_rows = {name: deepcopy(memory.时间到特征.表)
                            for name, memory in obj.memories.items()}
        for name, memory in obj.memories.items():
            for time, row in old_rows[name].items():
                assert row == memory.时间到特征.表[time]
            for time in range(36 * steps):
                assert memory.时间到特征.表[time] == {time // steps: 1.0}
                assert np.flatnonzero(obj.recall_at_time(time, .5)[name]).tolist() == [time // steps]
        assert obj.index.时间到输出 == {frame * steps: {frame} for frame in range(36)}
        growth_cases.append({'initial_capacity': capacity, 'steps_per_frame': steps,
                             'capacities': capacities, 'final_time': int(obj.clock.当前时间),
                             'old_time_rows_unchanged': True, 'all_recorded_frames_distinct': True})

    # Standalone stepping uses the original implementation after capacity reserve.
    standalone = GrowingTimeRing(2, .05, 3)
    transitions = [standalone.走一步() for _ in range(21)]
    assert transitions == [(j, j + 1) for j in range(21)]
    assert standalone.一帧() == (23, 24)
    assert int(standalone.时间激活.sum()) == 1

    # A failed allocation must precede any PFC/clock/memory mutation in record.
    full = runtime(3)
    learn(full, 0)
    snapshot_before = full.snapshot()
    with patch('growing_time_ring.np.zeros', side_effect=MemoryError('simulated capacity exhaustion')):
        try:
            value = np.ones(64, bool)
            full.record(value, value, value, value)
        except MemoryError:
            pass
        else:
            raise AssertionError('Expected simulated allocation failure')
    assert equal(snapshot_before, full.snapshot())

    # Fixed is exactly the old policy; no new snapshot field is required.
    fixed = runtime(3, policy='fixed')
    for i in range(8):
        learn(fixed, i)
    assert fixed.clock.当前时间 == 16 % 3 and fixed.clock.时间神经元数量 == 3
    assert type(fixed.clock).__name__ == '时间环'
    old_style = fixed.snapshot()
    assert 'time_ring_policy' not in old_style['parameters']
    assert equal(old_style, OriginalRuntime.from_snapshot(old_style).snapshot())

    migration = runtime(4, policy='fixed')
    learn(migration, 0)
    migration_before = migration.snapshot()
    migration.set_time_ring_policy('grow')
    migration_state = migration.snapshot()
    assert migration_state['parameters'].pop('time_ring_policy') == 'grow'
    assert equal(migration_before, migration_state)
    migration.set_time_ring_policy('fixed')
    assert equal(migration_before, migration.snapshot())
    migration.set_time_ring_policy('grow')

    # Safe numeric serialization at an imminent frontier, followed by many more
    # growths, recurrent steps and actual-pattern writes in two independent lives.
    with tempfile.TemporaryDirectory(prefix='growing_ring_audit_', dir=HERE / 'results') as folder:
        path = Path(folder) / 'runtime.npz'
        save_tree(path, migration.snapshot())
        restored = OriginalRuntime.from_snapshot(load_tree(path))
        assert equal(migration.snapshot(), restored.snapshot())
        for i in range(1, 40):
            assert equal(learn(migration, i), learn(restored, i))
            assert equal(migration.snapshot(), restored.snapshot())
        save_tree(path, migration.snapshot())
        assert equal(migration.snapshot(), OriginalRuntime.from_snapshot(load_tree(path)).snapshot())

    # Recall does not allocate, advance the real clock, or wrap into old birth
    # content when its requested playback reaches unallocated future positions.
    prior_recall = migration.snapshot()
    playback = migration.recall_from_audio(pattern(39), steps=512, threshold=.5)['playback']
    times = [item['time'] for item in playback]
    assert times and times == list(range(times[0], migration.clock.时间神经元数量))
    assert equal(prior_recall, migration.snapshot())

    # Frozen experience still has a continuing clock but no new learned edges.
    frozen = runtime(1)
    learn(frozen, 0)
    counts = frozen.connection_counts()
    for i in range(1, 20):
        learn(frozen, i, learning=False)
    assert frozen.connection_counts() == counts and frozen.clock.当前时间 == 40

    rejected = 0
    for policy in ('', 'other', None, 7):
        try:
            runtime(3, policy=policy)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid policy accepted')
    malformed = deepcopy(migration.snapshot())
    malformed['clock']['activity'][0] = True
    try:
        OriginalRuntime.from_snapshot(malformed)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError('Multiple active time cells accepted')
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    assert before == after
    result = {'kind': 'optional_single_growing_time_ring_audit_v1', 'passed': True,
              'growth_cases': growth_cases, 'standalone_transitions': 24,
              'allocation_failure_precedes_mutation': True,
              'fixed_original_class_and_wrap_preserved': True,
              'old_snapshot_without_policy_round_trip_exact': True,
              'policy_switch_preserves_entire_existing_state': True,
              'safe_npz_resume_learning_steps_exact': 39,
              'post_growth_safe_npz_round_trip_exact': True,
              'read_only_recall_has_no_frontier_wrap': True,
              'frozen_clock_continues_without_new_connections': True,
              'invalid_inputs_rejected': rejected,
              'protected_source_and_default_life_hashes_unchanged': before,
              'limitation': 'Optional engineering capacity policy. Finite machine memory still limits duration; existing fixed-ring collisions are not repaired. No navigation claim.'}
    target = HERE / 'results/growing_time_ring_audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'cases': len(growth_cases), 'continued_steps_exact': 39,
                      'report': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
