"""Independent read-only audit of the explicitly engineered lattice candidate.

Uses a saved pilot and actual training records; no new training corpus, source
mutation or write to the pilot/default life. Evaluation clones are disposable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_growing_time_ring import equal
from checkpoint import save_tree, load_tree
from developmental_navigation import DevelopmentalNavigator
from developmental_world import DevelopmentalWorld


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def learned_state(brain):
    saved = brain.snapshot()
    runtime = saved['runtime']
    adapter = saved['adapter']
    return {**{k: deepcopy(saved[k]) for k in ('config', 'prototypes', 'visits', 'edges',
        'attempts', 'inhibition', 'learning_steps')},
        'memory': {name: {k: deepcopy(v) for k, v in item.items() if k != 'activity'}
                   for name, item in runtime['memories'].items()},
        'index': deepcopy(runtime['index']),
        'parameters': deepcopy(runtime['parameters']),
        'excitatory': deepcopy(runtime['pfc']['excitatory']),
        'disinhibitory': deepcopy(runtime['pfc']['disinhibitory']),
        'adapter_metadata': deepcopy(adapter['metadata']),
        'adapter_connections': {k: v.copy() for k, v in adapter['arrays'].items()
                                if not k.startswith('state_')}}


def verify_selected(brain, trace, actual_records):
    edge = trace['selected_edge']
    if edge is None:
        return False
    time = edge['motor_time']
    recalled = brain.runtime.recall_at_time(time, .5)['motor'].astype(float)
    assert np.array_equal(recalled, trace['command'])
    assert int(recalled.sum()) == 1 and recalled[edge['action']] == 1
    assert time in actual_records
    assert np.array_equal(recalled, actual_records[time]['command'])
    assert edge['source'] == trace['place']
    return True


def exercise_fixes_and_serialization(state):
    brain = DevelopmentalNavigator.from_snapshot(state['brain'])
    world = DevelopmentalWorld.from_snapshot(state['world'])
    before = brain.snapshot()
    changed = brain.snapshot()
    changed['pending'][0] = .123
    changed['last_trace']['source'] = 'audit-only mutation'
    assert equal(before, brain.snapshot())
    invalid = brain.snapshot()
    invalid['goal'] = 1000000
    try:
        DevelopmentalNavigator.from_snapshot(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid goal accepted on restoration')
    brain.goal_activity[:] = .3
    brain._propagate(False)
    assert not brain.goal_activity.any()
    brain.goal = 1
    world.cue = 1511.
    packet = world.packet()
    cue = brain._cue(packet, brain.adapter.encode(packet, brain.last_action))
    assert cue['recognized'] is False and brain.goal is None and brain.goal_time is None

    # Unauthorized privileged fields must be rejected before any brain mutation.
    boundary_cases = 0
    clean_brain = DevelopmentalNavigator.from_snapshot(state['brain'])
    clean_world = DevelopmentalWorld.from_snapshot(state['world'])
    for field in ('world', 'position', 'target', 'route', 'teacher_muscles', 'reward'):
        packet = clean_world.packet()
        packet[field] = [1, 2]
        previous = clean_brain.snapshot()
        try:
            clean_brain.observe(packet, actual_muscles=clean_brain.pending)
        except ValueError:
            pass
        else:
            raise AssertionError('Privileged packet field accepted: ' + field)
        assert equal(previous, clean_brain.snapshot())
        boundary_cases += 1
    for field in ('world_position', 'absolute_heading', 'target_id'):
        packet = clean_world.packet()
        packet['observation'][field] = 1.
        previous = clean_brain.snapshot()
        try:
            clean_brain.observe(packet, actual_muscles=clean_brain.pending)
        except ValueError:
            pass
        else:
            raise AssertionError('Privileged observation field accepted: ' + field)
        assert equal(previous, clean_brain.snapshot())
        boundary_cases += 1

    a = DevelopmentalNavigator.from_snapshot(state['brain'])
    wa = DevelopmentalWorld.from_snapshot(state['world'])
    with tempfile.TemporaryDirectory(prefix='candidate_audit_', dir=HERE / 'results') as directory:
        path = Path(directory) / 'temporary_candidate.npz'
        save_tree(path, dict(brain=a.snapshot(), world=wa.snapshot()))
        loaded = load_tree(path)
        b = DevelopmentalNavigator.from_snapshot(loaded['brain'])
        wb = DevelopmentalWorld.from_snapshot(loaded['world'])
        assert equal(a.snapshot(), b.snapshot()) and equal(wa.snapshot(), wb.snapshot())
        for _ in range(12):
            ca, cb = a.pending.copy(), b.pending.copy()
            pa, pb = wa.step(ca), wb.step(cb)
            oa = a.observe(pa, actual_muscles=ca, learning=True)
            ob = b.observe(pb, actual_muscles=cb, learning=True)
            assert equal(oa, ob)
            assert equal(a.snapshot(), b.snapshot()) and equal(wa.snapshot(), wb.snapshot())
    return dict(snapshot_arrays_and_trace_detached=True, invalid_goal_rejected=True,
                unknown_sound_clears_goal=True, disabled_propagation_clears_field=True,
                privileged_packet_cases_rejected_without_mutation=boundary_cases,
                safe_numeric_checkpoint_resume_steps_exact=12)


def audit_training_alignment(state, report):
    brain = DevelopmentalNavigator.from_snapshot(state['brain'])
    records = {int(row['record_time']): row for row in report['autonomous_records']}
    assert len(records) == len(report['autonomous_records'])
    times = sorted(records)
    assert times == list(range(2, 2 * (len(times) + 1), 2))
    for time, row in records.items():
        actual = brain.runtime.recall_at_time(time, .5)['motor'].astype(float)
        assert np.array_equal(actual, row['command'])
    checked = 0
    for condition in ('navigation', 'switches', 'propagation_lesion'):
        for trial in report[condition]:
            assert trial['learning_steps_before'] == trial['learning_steps_after']
            for trace in trial['traces']:
                assert trace['learning'] is False and trace['current_time_cells'] == 1
                checked += int(verify_selected(brain, trace, records))
    for (source, action, target), edge in brain.edges.items():
        row = records[edge['motor_time']]
        assert row['command'][action] == 1.
        assert row['source'] != row['destination'] or source == target
        source_group = {cell // brain.config.assembly_size for cell in
                        brain.runtime.index.时间到输出[edge['source_time']]
                        if cell < brain.audio_offset}
        target_group = {cell // brain.config.assembly_size for cell in
                        brain.runtime.index.时间到输出[edge['motor_time']]
                        if cell < brain.audio_offset}
        assert source_group == {source} and target_group == {target}
        if edge['source_time'] in records:
            assert records[edge['source_time']]['destination'] == row['source']
    return records, dict(actual_training_actions_aligned=len(records),
                         frozen_selected_motor_references_aligned=checked,
                         learned_relations_source_successor_time_aligned=len(brain.edges))


def blocked_matched_prefix(state, records):
    brain = DevelopmentalNavigator.from_snapshot(state['brain'])
    world = DevelopmentalWorld.from_snapshot(state['world'])
    world.position = np.array([2, 2], int)
    world.velocity[:] = 0.
    world.touch[:] = 0.
    world.local_sounds = False
    world.cue = world.tones[0]
    brain.pending = None  # Abandon an unexecuted proposal before an external start change.
    command, trace = brain.observe(world.packet(), learning=False, pursue=True)
    world.cue = None
    initial_learned = learned_state(brain)
    prefix_path = [world.position.tolist()]
    for _ in range(3):
        packet = world.step(command)
        command, trace = brain.observe(packet, actual_muscles=command, learning=False, pursue=True)
        prefix_path.append(world.position.tolist())
        assert equal(initial_learned, learned_state(brain))
    assert tuple(world.position) != (5, 1)
    blocked_source, blocked_action = trace['place'], int(np.argmax(command))
    target = tuple(world.beacons[0])
    prefix_brain, prefix_world = brain.snapshot(), world.snapshot()
    outputs = []
    clones = []
    for _ in range(3):
        b = DevelopmentalNavigator.from_snapshot(prefix_brain)
        w = DevelopmentalWorld.from_snapshot(prefix_world)
        assert equal(prefix_brain, b.snapshot()) and equal(prefix_world, w.snapshot())
        clones.append((b, w))
    for (name, blocked, online), (b, w) in zip((('open_frozen', False, False),
            ('blocked_frozen', True, False), ('blocked_online', True, True)), clones):
        if blocked:
            w.grid[1, 5] = True
        assert equal(prefix_brain, b.snapshot())
        learned_before = learned_state(b)
        start_clock = int(b.runtime.clock.当前时间)
        path, entries, checked = [w.position.tolist()], [], 0
        actual_records = deepcopy(records)
        for step in range(100):
            command = b.pending.copy()
            previous = w.position.tolist()
            packet = w.step(command)
            timestamp = int(b.runtime.clock.当前时间)
            if online:
                actual_records[timestamp] = dict(source=previous, destination=w.position.tolist(),
                                                command=command.tolist(), record_time=timestamp)
            _, tr = b.observe(packet, actual_muscles=command, learning=online, pursue=True)
            assert tr['current_time_cells'] == 1
            assert b.runtime.clock.当前时间 == start_clock + 2 * (step + 1)
            checked += int(verify_selected(b, tr, actual_records))
            if not online:
                assert equal(learned_before, learned_state(b))
            path.append(w.position.tolist())
            entries.append(dict(step=step, time=timestamp, position=w.position.tolist(),
                                action=command.tolist(), contact=bool(w.touch.any()),
                                blocked_connection_inhibition=float(b.inhibition[blocked_source, blocked_action]),
                                selected_edge=tr['selected_edge']))
            if tuple(w.position) == target:
                break
        outputs.append(dict(condition=name, online=online, success=tuple(w.position) == target,
                            steps=len(entries), collisions=sum(item['contact'] for item in entries),
                            path=path, trace=entries, selected_memory_checks=checked,
                            final_clock=int(b.runtime.clock.当前时间),
                            learned_state_unchanged=equal(learned_before, learned_state(b))))
    open_case, frozen_case, online_case = outputs
    assert open_case['success'] and not frozen_case['success'] and online_case['success']
    assert [5, 1] in open_case['path'] and [5, 7] in online_case['path']
    assert online_case['collisions'] == 1 and frozen_case['collisions'] > 1
    assert open_case['learned_state_unchanged'] and frozen_case['learned_state_unchanged']
    assert online_case['trace'][0]['blocked_connection_inhibition'] == 1.
    assert prefix_brain['inhibition'][blocked_source, blocked_action] == 0.
    return dict(prefix_actual_path=prefix_path, prefix_clock=prefix_brain['runtime']['clock']['current_time'],
                all_three_clones_full_state_equal_before_environment_change=True,
                blocked_source_assembly=blocked_source, blocked_motor_channel=blocked_action,
                intervention='After three frozen physical steps, only two worlds gain a wall at (5,1); one brain then learns from actual outcomes.',
                outcomes=outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', default='developmental_pilot_71')
    parser.add_argument('--output', default='developmental_navigation_independent_audit.json')
    args = parser.parse_args()
    checkpoint = HERE / 'results' / (args.pilot + '.npz')
    report_path = HERE / 'results' / (args.pilot + '.json')
    protected = [HERE / 'developmental_navigation.py', HERE / 'developmental_world.py',
                 HERE / 'original_runtime.py', HERE / 'growing_time_ring.py',
                 HERE / 'results/living_original.npz', checkpoint, report_path]
    hashes_before = {str(path): sha(path) for path in protected}
    state = load_tree(checkpoint)
    report = json.loads(report_path.read_text(encoding='utf8'))
    fixes = exercise_fixes_and_serialization(state)
    records, alignment = audit_training_alignment(state, report)
    blocked = blocked_matched_prefix(state, records)
    hashes_after = {str(path): sha(path) for path in protected}
    assert hashes_before == hashes_after, 'Source or protected checkpoint changed during independent audit'
    result = dict(kind='developmental_navigation_independent_audit_v1', passed=True,
        scope='Disposable evaluation clones of the saved pilot; no default-life or pilot mutation.',
        source_and_checkpoint_hashes=hashes_before, fixes_and_serialization=fixes,
        actual_memory_alignment=alignment, same_state_blocked_route=blocked,
        attribution=['OriginalRuntime supplies the single chronological multimodal record and actual motor recall.',
                     'New sensory-prototype assemblies, action-conditioned experience edges, maximum discounted goal propagation and contact-dependent edge inhibition supply candidate navigation.',
                     'The original spiking PFC receives transition learning but is not called to choose these actions; planner success is not evidence of its spontaneous emergence.',
                     'This is a new lattice body with nine downward pigment samples, not transfer to the original continuous physical body.',
                     'Pilot switches change the destination; the matched-prefix blocked-route test keeps the destination fixed and changes the available path.',
                     'No language or AGI conclusion follows from this bounded navigation result.'])
    destination = HERE / 'results' / args.output
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(passed=True, aligned_training_actions=alignment['actual_training_actions_aligned'],
        aligned_frozen_selections=alignment['frozen_selected_motor_references_aligned'],
        blocked=[{k: item[k] for k in ('condition', 'success', 'steps', 'collisions')}
                 for item in blocked['outcomes']], output=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
