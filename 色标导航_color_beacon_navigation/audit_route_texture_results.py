"""Read-only recount of gray/textured results; writes only a new audit report."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
BRANCHES = ('stay', 'switch', 'no_cue', 'swapped_acoustic_mapping', 'clear_rule_after_cue')
EPS = 1e-8


def physical_trajectory(row):
    # request_index is evaluator annotation and intentionally differs for a
    # repeated cue versus no cue; remove it, retain every physical sample.
    return [{k: sample[k] for k in ('time', 'position', 'heading')}
            for sample in row['trajectory']]


def inspect_arrival(row):
    reached = bool(row['target_reached'])
    assert bool(row['terminated']) == reached
    if not reached:
        assert row['arrival_route_score'] is None and not row['correct_arrival']
        return False
    target_entries = [event for event in row['arrivals'] if event['name'] == 'red']
    assert target_entries and abs(target_entries[0]['time']-row['first_target_time']) < EPS
    arrival_time = row['first_target_time']
    # Derive the approach from raw completed crossings, not used_passage or
    # correct_arrival. Only crossings completed before first red entry qualify.
    completed = [p for p in row['passages'] if p['completed_time'] <= arrival_time+EPS]
    last = completed[-1] if completed else None
    request = row['route_requests'][-1] if row['route_requests'] else None
    expected = bool(last and request and last['direction'] == 'left_to_right'
                    and last['route'] == request['route']
                    and last['time'] >= request['requirement_since']-EPS)
    assert row['correct_arrival'] == expected
    assert row['arrival_route_score']['correct_arrival'] == expected
    return expected


def recount_strict(row, expected_route):
    correct = inspect_arrival(row)
    first = next((p for p in row['passages'] if p['direction'] == 'left_to_right'), None)
    arrival_time = row['first_target_time']
    at_arrival = [p for p in row['passages'] if arrival_time is not None
                  and p['completed_time'] <= arrival_time+EPS]
    last = at_arrival[-1] if at_arrival else None
    return bool(correct and first and last
                and first['time'] >= 10.6-EPS and last['time'] >= 10.6-EPS
                and first['route'] == expected_route and last['route'] == expected_route
                and last['direction'] == 'left_to_right' and row['goal_preserved'])


def audit_condition(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] != 'complete':
        raise SystemExit(f'{path.name} is not complete; no audit output changed')
    protocol = data['protocol']
    assert len(data['training']) == 36 and len(data['pairs']) == 6
    assert protocol['evaluation_learning'] is False and protocol['evaluation_reward'] is False
    assert protocol['teacher_demonstrations'] == 0
    assert {(p['seed'], p['initial_route']) for p in data['pairs']} == {
        (seed, route) for seed in (831, 931, 1031) for route in ('bottom', 'top')}
    coverage = {(start, route): {'trials': 0, 'rewarded': 0, 'physical_red_arrivals': 0}
                for start in protocol['starts_independent_of_route'] for route in ('bottom', 'top')}
    indirect_training = []
    for index, row in enumerate(data['training']):
        schedule = protocol['schedule'][index]
        assert row['episode'] == schedule['episode'] == index
        assert row['seed'] == schedule['seed']
        assert row['start_name'] == schedule['start'] and row['repeat'] == schedule['repeat']
        assert row['requested_route'] == schedule['route']
        assert row['initial_pose'] == protocol['starts_independent_of_route'][row['start_name']]
        assert row['teacher_supplied'] is False and row['learning'] is True
        correct = inspect_arrival(row)
        assert row['reward_supplied'] == correct
        if correct:
            assert row['reward_details']['reward'] == 1.
            assert row['first_target_time'] <= row['elapsed_seconds']+EPS
        cell = coverage[row['start_name'], row['requested_route']]
        cell['trials'] += 1
        cell['rewarded'] += int(correct)
        cell['physical_red_arrivals'] += int(row['target_reached'])
        first = next((p for p in row['passages'] if p['direction'] == 'left_to_right'), None)
        if correct and first and first['route'] != row['requested_route']:
            indirect_training.append(index)
    assert all(cell['trials'] == 6 for cell in coverage.values())

    branch_counts = Counter({branch: 0 for branch in BRANCHES})
    pair_results, frozen_rows, clone_checks, exact_controls = [], 0, 0, []
    for pair in data['pairs']:
        prefix = pair['prefix']
        valid = (prefix['control_steps'] == 100 and not prefix['terminated']
                 and not any(p['direction'] == 'left_to_right' for p in prefix['midline_crossings'])
                 and prefix['final_position'][0] <= prefix['route_gate_geometry']['left_x']+1e-10)
        assert valid == pair['valid_prefix']
        rows = [prefix]
        calculated = {}
        if valid:
            assert abs(pair['switch_time']-10.) < EPS
            assert set(pair['branches']) == set(BRANCHES)
            for branch, row in pair['branches'].items():
                expected = pair['initial_route'] if branch in ('stay', 'no_cue') else next(
                    route for route in ('bottom', 'top') if route != pair['initial_route'])
                assert row['clone_state_sha256'] == pair['full_prefix_state_sha256']
                clone_checks += 1
                assert physical_trajectory(row)[:len(prefix['trajectory'])] == physical_trajectory(prefix)
                calculated[branch] = recount_strict(row, expected)
                assert calculated[branch] == row['strict_success']
                branch_counts[branch] += int(calculated[branch])
                rows.append(row)
            stay = pair['branches']['stay']
            for other in ('no_cue', 'swapped_acoustic_mapping'):
                candidate = pair['branches'][other]
                assert physical_trajectory(stay) == physical_trajectory(candidate)
                assert stay['trace'] == candidate['trace']
                exact_controls.append(dict(seed=pair['seed'], initial_route=pair['initial_route'],
                                           compared_to_stay=other, full_physics_samples=len(stay['trajectory']),
                                           sampled_muscles_and_logged_brain_info_equal=True))
            cleared = pair['branches']['clear_rule_after_cue']
            assert cleared['rule_state_cleared']
            first_cleared = next(sample for sample in cleared['trace'] if sample['time'] >= 10.7-EPS)
            assert first_cleared['info']['rule_active'] == [0]
            assert first_cleared['info']['goal_active'] == [0]
        else:
            assert not pair['branches']
        for row in rows:
            assert row['learning'] is False and row['teacher_supplied'] is False
            assert row['reward_supplied'] is False
            assert row['events_before'] == row['events_after']
            assert row['strength_before'] == row['strength_after']
            frozen_rows += 1
        success = bool(valid and calculated.get('stay') and calculated.get('switch'))
        assert pair['pair_success'] == success
        pair_results.append(dict(seed=pair['seed'], initial_route=pair['initial_route'],
                                 valid_prefix=valid, strict_branches=calculated, pair_success=success))

    checkpoint = Path(data['checkpoint'])
    if not checkpoint.exists():
        checkpoint = path.with_suffix('.npz')
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == data['checkpoint_sha256']
    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].tobytes().decode('utf-8'))
        assert metadata['birth_hash'] == data['birth_hash']
        assert metadata['mixed_neurons'] == protocol['mixed_neurons'] == 8192
        assert metadata['max_events'] == protocol['max_events'] == 100000
        assert int(np.count_nonzero(archive['event_kind'] == 2)) == 0
        assert int(np.count_nonzero(archive['event_ids'] >= 0)) == data['memory_inventory']['events']
        assert int(np.count_nonzero(archive['event_strength'] > 0)) == data['memory_inventory']['reinforced_events']
        assert float(archive['event_strength'].sum()) == data['memory_inventory']['weight_sum']
    summary = dict(training_successes=sum(cell['rewarded'] for cell in coverage.values()),
        training_trials=36, paired_successes=sum(pair['pair_success'] for pair in pair_results),
        paired_denominator=6, valid_prefixes=sum(pair['valid_prefix'] for pair in pair_results),
        branch_successes=dict(branch_counts))
    assert summary['paired_successes'] == data['summary']['paired_successes']
    assert summary['branch_successes'] == data['summary']['branch_successes']
    return data, dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint_hash, birth_hash=data['birth_hash'], summary=summary,
        pairs=pair_results, coverage=[dict(start=start, route=route, **cell) for (start, route), cell in coverage.items()],
        rewarded_training_episodes_with_opposite_first_passage=indirect_training,
        matching_full_clone_hashes=clone_checks, frozen_evaluation_rows=frozen_rows,
        exact_same_physical_trajectory_controls=exact_controls,
        checkpoint_inventory_matches_report=True, teacher_event_count=0,
        frozen_evaluation_evidence='Recorded event counts and strength sums unchanged, no rewards, learning=False; this audit does not claim per-array before/after evaluation hashes')


def main():
    gray_data, gray = audit_condition(RESULTS/'route_switch_evaluation.json')
    texture_data, texture = audit_condition(RESULTS/'route_texture_evaluation.json')
    assert gray_data['birth_hash'] == texture_data['birth_hash']
    assert gray_data['protocol']['schedule'] == texture_data['protocol']['schedule']
    gray_protocol = {k: v for k, v in gray_data['protocol'].items() if k != 'wall_texture_seed'}
    texture_protocol = {k: v for k, v in texture_data['protocol'].items() if k != 'wall_texture_seed'}
    assert gray_protocol == texture_protocol
    assert gray_data['protocol'].get('wall_texture_seed') is None
    assert texture_data['protocol']['wall_texture_seed'] == 44017
    stable_modules = ('environment.py', 'associative_controller.py', 'route_switch_environment.py', 'route_switch_controller.py')
    assert all(gray_data['code_sha256'][name] == texture_data['code_sha256'][name] for name in stable_modules)
    assert next(cell for cell in texture['coverage'] if cell['start'] == 'upper' and cell['route'] == 'bottom')['rewarded'] == 0
    report = dict(
        status='complete', gray=gray, textured=texture,
        comparison=dict(same_birth_hash=True, same_all_36_episode_schedule=True,
            same_protocol_except_visible_wall_texture_seed=True,
            identical_physics_scoring_and_controller_module_hashes=list(stable_modules),
            texture_seed=44017,
            runner_hash_differs_reason='Texture factory/option plus first-crossing scoring correction and digest provenance; both result sets independently recounted using the same first+last >=10.6 rule.',
            training_success_change=f"{gray['summary']['training_successes']}/36 -> {texture['summary']['training_successes']}/36",
            strict_pair_success_change=f"{gray['summary']['paired_successes']}/6 -> {texture['summary']['paired_successes']}/6"),
        interpretation=[
            'Visible texture improved this fixed birth, training schedule and probe set; six pairs are not broad reliability evidence.',
            'Texture condition still has zero rewarded upper-start/bottom-route episodes out of six; successful static examples elsewhere do not establish coverage at every switching pose.',
            'No-cue retains the old requirement. Identical trajectories for stay/no-cue/swapped acoustics support the absence of a private route-label behavior bypass.',
            'Clearing rule state also clears replay activity; its failure tests that combined intervention, not isolated proof of an anatomical PFC mechanism.',
            'Both recorded conditions have zero rewarded training episodes whose first complete passage was the opposite route; successful episodes still do not establish shortest-path planning.',
            'No new training, parameter changes or source-result edits were performed by this audit.'])
    output = RESULTS/'route_texture_independent_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'gray': gray['summary'], 'textured': texture['summary'],
                      'textured_coverage': texture['coverage'], 'comparison': report['comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
