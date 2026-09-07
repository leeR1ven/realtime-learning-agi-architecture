"""Deterministic white-box acceptance checks; no navigation or task labels.

Run with the workspace .venv Python. Results describe this finite neural
mechanism and its computational cost, not general intelligence or navigation.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from recurrent_core import SparseHebbianPFC


HERE = Path(__file__).resolve().parent


def equal_snapshots(left, right):
    assert left['metadata'] == right['metadata']
    assert left['arrays'].keys() == right['arrays'].keys()
    for key in left['arrays']:
        np.testing.assert_array_equal(left['arrays'][key], right['arrays'][key], err_msg=key)


def weight_digest(core):
    snapshot = core.snapshot()
    digest = hashlib.sha256()
    for key, value in sorted(snapshot['arrays'].items()):
        if key.startswith(('excitatory_', 'disinhibitory_')):
            digest.update(key.encode())
            digest.update(value.tobytes())
    for key in ('excitatory', 'disinhibitory'):
        digest.update(json.dumps(snapshot['metadata'][key], sort_keys=True).encode())
    return digest.hexdigest()


def clone(core):
    return SparseHebbianPFC.from_snapshot(core.snapshot())


def safe_npz_roundtrip(core):
    snapshot = core.snapshot()
    payload = dict(snapshot['arrays'])
    payload['metadata_json'] = np.frombuffer(json.dumps(snapshot['metadata']).encode('utf-8'), dtype=np.uint8)
    memory_file = io.BytesIO()
    np.savez_compressed(memory_file, **payload)
    size = memory_file.tell()
    memory_file.seek(0)
    with np.load(memory_file, allow_pickle=False) as archive:
        restored = {'metadata': json.loads(archive['metadata_json'].tobytes().decode('utf-8')),
                    'arrays': {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}}
    return SparseHebbianPFC.from_snapshot(restored), size


def small_checks():
    core = SparseHebbianPFC(width=32, site_width=4)
    a, b, c, d = [core.activity_from_indices(ids) for ids in ([0, 1], [8, 9], [16, 17], [24, 25])]
    zero = np.zeros(core.width, dtype=np.float32)
    core.observe_transition(a, b)
    core.observe_transition(b, c)
    sequence = []
    for drive in (a, zero, zero, zero):
        activity, diag = core.advance(drive)
        sequence.append(np.flatnonzero(activity).tolist())
    assert sequence == [[0, 1], [8, 9], [16, 17], []]
    assert core.excitatory.edge_count == 8 and core.disinhibitory.edge_count == 4

    # The same weak factual A->B association is sufficient only when its
    # learned cell->site path relieves inhibition on B.
    core.reset_activity()
    core.advance(a)
    base = core.snapshot()
    on = SparseHebbianPFC.from_snapshot(base)
    no_disinhibition = SparseHebbianPFC.from_snapshot(base)
    no_disinhibition.disinhibition_enabled = False
    no_recurrence = SparseHebbianPFC.from_snapshot(base)
    no_recurrence.recurrent_enabled = False
    y_on, diag_on = on.advance(zero)
    y_no_dis, diag_no_dis = no_disinhibition.advance(zero)
    y_no_rec, _ = no_recurrence.advance(zero)
    np.testing.assert_array_equal(y_on, b)
    assert not y_no_dis.any() and not y_no_rec.any()
    assert np.all(diag_on['net_inhibitory_site_current'][2] == 0.)
    assert diag_no_dis['net_inhibitory_site_current'][2] > 0.

    # Multiple true successors can be coactive, with no winner or top-K cut.
    branching = SparseHebbianPFC(width=32, site_width=4)
    branching.observe_transition(a, b)
    branching.observe_transition(a, d)
    branching.advance(a)
    multi, _ = branching.advance(zero)
    np.testing.assert_array_equal(multi, b | d)
    assert branching.excitatory.edge_count == 8

    cycling = SparseHebbianPFC(width=32, site_width=4)
    cycling.observe_transition(a, b)
    cycling.observe_transition(b, a)
    cycling.advance(a)
    persistent = []
    for index in range(12):
        activity, _ = cycling.advance(zero)
        np.testing.assert_array_equal(activity, b if index % 2 == 0 else a)
        persistent.append(np.flatnonzero(activity).tolist())
    cycle_edges = cycling.excitatory.edge_count
    cycling.reset_activity()
    activity, _ = cycling.advance(zero)
    assert not activity.any() and cycling.excitatory.edge_count == cycle_edges
    cycling.set_activity(b)
    activity, _ = cycling.advance(zero)
    np.testing.assert_array_equal(activity, a)

    # External evidence and recalled evidence are separate causal sources.
    virgin = SparseHebbianPFC(width=32, site_width=4)
    assert not virgin.advance(a.astype(np.float32) * .4)[0].any()
    virgin.reset_activity()
    heard, d_ext = virgin.advance(a)
    np.testing.assert_array_equal(heard, a)
    assert not d_ext['recalled_current'].any()
    virgin.reset_activity()
    thought, d_replay = virgin.advance(zero, b)
    np.testing.assert_array_equal(thought, b)
    assert not d_replay['external_current'].any()
    virgin.reset_activity()
    virgin.recalled_enabled = False
    assert not virgin.advance(zero, b)[0].any()
    virgin.reset_activity()
    virgin.external_enabled = False
    assert not virgin.advance(a)[0].any()

    # Imagined activation never silently becomes observed-transition Hebb.
    before_digest = weight_digest(core)
    for _ in range(5):
        core.advance(zero, d)
    assert weight_digest(core) == before_digest
    core.observe_transition(c, d)
    assert weight_digest(core) != before_digest

    # Uniform decay is the only deletion trigger, including fractional weak
    # facts: activity competition itself never removes learned connections.
    decaying = SparseHebbianPFC(width=8, site_width=2, decay_interval=1,
                               decay_factor=.1, disappearance_threshold=.01,
                               prune_rows_per_tick=8)
    first = decaying.activity_from_indices([0])
    second = decaying.activity_from_indices([4])
    decaying.observe_transition(first, second)
    assert decaying.excitatory.edge_count == 1
    np.testing.assert_allclose(decaying.excitatory.rows[0][4] * decaying.excitatory.global_scale, .025)
    decaying.advance(np.zeros(8), learning=True)
    assert decaying.excitatory.edge_count == 0 and decaying.disinhibitory.edge_count == 0
    assert decaying.excitatory.removed_by_decay == 1
    assert decaying.disinhibitory.removed_by_decay == 1

    # Broad input raises an overall inhibitory state; local feedback uses
    # density, so site_size never multiplies the numerical resting threshold.
    homeostasis = SparseHebbianPFC(width=32, site_width=4)
    _, home = homeostasis.advance(np.ones(32))
    assert home['inhibition_strength_after'] > home['inhibition_strength_before']
    _, home = homeostasis.advance(np.ones(32))
    assert np.all(home['local_site_density'] == 1.)
    assert float(home['inhibitory_site_current'].max()) <= home['inhibition_strength_before'] * .450001

    # Complete continuation, including live dynamics, disabled pathways,
    # partial uniform-decay scan, maintenance counters and diagnostics.
    saved = SparseHebbianPFC(width=32, site_width=4, decay_interval=3,
                             decay_factor=.9, prune_rows_per_tick=3)
    for pre, post in ((a, b), (b, c), (c, d), (d, a)):
        saved.observe_transition(pre, post)
    saved.advance(a)
    saved.advance(zero, d.astype(np.float32) * .2)
    saved.external_enabled = False
    restored, npz_bytes = safe_npz_roundtrip(saved)
    equal_snapshots(saved.snapshot(), restored.snapshot())
    for index in range(12):
        for brain in (saved, restored):
            brain.advance(a if index % 2 else zero, b.astype(np.float32) * .1)
            brain.observe_transition((a, b, c, d)[index % 4], (b, c, d, a)[index % 4])
        equal_snapshots(saved.snapshot(), restored.snapshot())

    return {
        'fact_sequence_with_no_task_reward': sequence,
        'same_state_disinhibition_on': np.flatnonzero(y_on).tolist(),
        'same_state_disinhibition_off': np.flatnonzero(y_no_dis).tolist(),
        'same_state_recurrence_off': np.flatnonzero(y_no_rec).tolist(),
        'coactive_true_successors': np.flatnonzero(multi).tolist(),
        'persistent_blank_steps': len(persistent),
        'activity_reset_and_intervention_causal': True,
        'external_and_replay_sources_independently_causal': True,
        'internal_replay_does_not_strengthen_synapses': True,
        'observed_transition_without_reward_does_strengthen_synapses': True,
        'uniform_decay_only_deletion': True,
        'global_inhibition_and_site_density_checked': True,
        'safe_npz_size_bytes': npz_bytes,
        'continued_npz_restore_identical_steps': 12,
    }


def normalization_checks():
    maxima = []
    for n_active in (16, 512):
        core = SparseHebbianPFC(width=1024, site_width=32, decay_interval=1000)
        before = core.activity_from_indices(np.arange(n_active))
        after = core.activity_from_indices(np.arange(512, 512 + n_active))
        for _ in range(24):
            core.observe_transition(before, after)
        core.set_activity(before)
        _, diag = core.advance(np.zeros(core.width))
        maximum = float(diag['recurrent_current'].max())
        assert maximum <= 1.00001
        assert float(diag['disinhibitory_site_current'].max()) <= 1.30001
        maxima.append({'active_sources': n_active, 'maximum_recurrent_current': maximum,
                       'excitatory_edges': core.excitatory.edge_count})
    assert abs(maxima[0]['maximum_recurrent_current'] - maxima[1]['maximum_recurrent_current']) < 1e-5
    return maxima


def performance_checks():
    core = SparseHebbianPFC(width=8192)
    patterns = [core.activity_from_indices(np.arange(i * 500, (i + 1) * 500)) for i in range(6)]
    zero = np.zeros(core.width, dtype=np.float32)
    learning_times, advance_times, combined_times = [], [], []
    allocation_start = time.perf_counter()
    for index in range(6):
        core.observe_transition(patterns[index], patterns[(index + 1) % 6])
    allocation_seconds = time.perf_counter() - allocation_start
    for index in range(30):
        pre, post = patterns[index % 6], patterns[(index + 1) % 6]
        core.set_activity(pre)
        begin = time.perf_counter()
        core.observe_transition(pre, post)
        learned = time.perf_counter()
        activity, diag = core.advance(post.astype(np.float32) * .75)
        advanced = time.perf_counter()
        learning_times.append((learned - begin) * 1000)
        advance_times.append((advanced - learned) * 1000)
        combined_times.append((advanced - begin) * 1000)
        assert np.isfinite(diag['membrane']).all()
        assert float(diag['recurrent_current'].max()) <= 1.00002
        assert diag['presynaptic_activity_sum'] == 500.
    stats = lambda values: {'median_ms': float(np.median(values)),
                            'p95_ms': float(np.percentile(values, 95)),
                            'max_ms': float(np.max(values))}
    result = {
        'width': core.width, 'active_sources_per_step': 500,
        'stored_cyclic_patterns': 6, 'measured_steps': len(combined_times),
        'initial_allocation_seconds': allocation_seconds,
        'observe_transition': stats(learning_times), 'advance': stats(advance_times),
        'combined': stats(combined_times), 'frame_budget_ms': 100.,
        'within_budget_at_p95': bool(np.percentile(combined_times, 95) < 100),
        'excitatory_edges': core.excitatory.edge_count,
        'disinhibitory_edges': core.disinhibitory.edge_count,
        'allocated_synapse_bytes': diag['allocated_synapse_bytes'],
        'fully_materialized_synapse_bytes': 4 * core.width * (core.width + core.site_count),
        'note': 'Synthetic 500-active patterns; integration lookup/physics/serialization costs are excluded.',
    }
    assert result['within_budget_at_p95'], result
    return result


def main():
    start = time.perf_counter()
    result = {'status': 'passed',
              'mechanism': 'Persistent temporal-Hebbian cell excitation and cell-to-site disinhibition; finite local efficacy and population-mean current normalization.',
              'code_sha256': hashlib.sha256((HERE / 'recurrent_core.py').read_bytes()).hexdigest(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'small_causal_checks': small_checks(),
              'activity_size_normalization': normalization_checks(),
              'performance': performance_checks(),
              'limits': [
                  'These constructed neural fixtures do not establish navigation, flexible subgoals, language, or AGI.',
                  'Transitions are unsupervised facts. Task value and candidate selection must remain a separate mechanism in the integrator.',
                  'A finite width cannot create information absent from its input; overlapping experiences may interfere and produce mixed successors.',
                  'Population-mean normalization suppresses associations when many unrelated presynaptic cells are concurrently active; this is a stability tradeoff.',
                  'Default uniform decay gradually weakens inactive facts; this is not indefinite retention or an unbounded capacity claim.',
                  'Full 8192-row storage is about 275 MB before Python/state/CSR snapshot copies. Full 32768 rows would exceed 4 GB.',
                  'advance learning=True means maintenance only. observe_transition must receive actual before/after activity after executed physics, never an imagined successor.',
              ]}
    result['elapsed_seconds'] = time.perf_counter() - start
    result['default_weight_half_life_observed_steps'] = 200 * math.log(.5) / math.log(.995)
    output = HERE / 'results'
    output.mkdir(exist_ok=True)
    (output / 'core_acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    perf = result['performance']
    prose = (
        '# Recurrent core acceptance\n\n'
        'The deterministic causal and complete-state checks passed. This is an isolated mechanism check, not a navigation or AGI result.\n\n'
        '- Real A → B → C transitions produce A, B, C, then silence without further sensory input or task reward.\n'
        '- At the same state and weights, disabling disinhibition or recurrence stops the weak next-pattern activation.\n'
        '- Multiple observed successors activate together. A learned two-pattern cycle persists for 12 blank steps; resetting activity stops it.\n'
        '- External and remembered current have separately checked effects. Internal replay does not strengthen weights.\n'
        '- A safe NPZ round trip continues 12 steps with exact arrays, flags, diagnostics and decay state.\n'
        '- 16 versus 512 active sources give the same bounded recurrent-current maximum within floating-point tolerance.\n\n'
        f'8192 cells, 500 active sources, {perf["excitatory_edges"]:,} excitatory edges and {perf["disinhibitory_edges"]:,} disinhibitory edges: '
        f'combined observe/advance median {perf["combined"]["median_ms"]:.2f} ms, p95 {perf["combined"]["p95_ms"]:.2f} ms. '
        f'Allocated synapse storage {perf["allocated_synapse_bytes"] / 1e6:.1f} MB. This excludes the rest of the controller.\n\n'
        'Numerical changes from the original raw-sum class are explicit: saturating Hebb efficacy, mean presynaptic current, normalized local site activity, leaky membrane, cell adaptation and bounded overall inhibition. '
        'Every nonzero learned edge is retained until uniform decay places it below the common disappearance threshold; there is no top-K connection deletion.\n\n'
        'Default decay gives a weight half-life of about 27,657 actual observation steps without reinforcement. '
        'Fixed-width interference, mixture ambiguity, finite memory and actual embodied integration remain to be evaluated.\n'
    )
    (output / 'core_acceptance.md').write_text(prose, encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'performance': perf, 'code_sha256': result['code_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
