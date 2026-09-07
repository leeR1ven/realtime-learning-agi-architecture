"""Autonomous integration training, development probes and held-out evaluation.

Training includes real fixed-time auditory changes. No teacher, spatial action
oracle, intermediate progress reward, or answer fields enter the controller.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from integrated_controller import IntegratedController
from integration_environment import IntegrationEnvironment
from route_state_digest import digest

HERE = Path(__file__).resolve().parent
STARTS = {'middle': (1.3, 3.4, 0.), 'lower': (1.3, 1.4, 0.), 'upper': (1.3, 6.6, 0.)}
ROUTES = ('bottom', 'top')
DEVELOPMENT_SEEDS = (831, 931, 1031)
HELD_OUT_SEEDS = (1831, 1931, 2031)


def compact(info):
    result = {key: info[key] for key in ('source', 'goal_active', 'rule_active',
        'pfc_active_count', 'external_pfc_count', 'fact_recall_unrewarded',
        'fact_recall_events', 'recall_events', 'recall_weights', 'motor_error',
        'motor_group', 'last_prediction_error', 'muscle_currents') if key in info}
    core = info.get('core', {})
    result['core'] = {key: core[key] for key in ('excitatory_edges', 'disinhibitory_edges',
        'active_count', 'used_excitatory_edges', 'used_disinhibitory_edges',
        'inhibition_strength_after', 'allocated_synapse_bytes') if key in core}
    return result


def roll(env, brain, steps, *, learning):
    before = {'events': brain.next_event_id, 'strength': float(brain.event_strength.sum()),
              'transitions': brain.core.transition_count}
    packet = env.sensor_packet()
    trace, times, sources = [], [], {}
    all_goal = True
    supplied_rewards = 0.
    for frame in range(steps):
        if env.private_metrics()['terminated']:
            break
        begin = time.perf_counter()
        muscles, info = brain.observe_then_act(packet, learning=learning)
        transition = env.advance(muscles)
        feedback = transition['feedback']
        if not learning:
            feedback = {**feedback, 'reward': 0.}
        outcome_info = brain.observe_outcome(transition['packet'], muscles, **feedback, learning=learning)
        packet = transition['packet']
        supplied_rewards += feedback['reward']
        elapsed = (time.perf_counter()-begin)*1000
        times.append(elapsed)
        all_goal &= info['goal_active'] == [0]
        sources[info['source']] = sources.get(info['source'], 0)+1
        if frame < 7 or frame % 20 == 0 or feedback['terminal']:
            trace.append({'time': float(env.world.time), 'position': env.world.position.tolist(),
                'heading': float(env.world.heading), 'muscles': muscles.tolist(),
                'info': compact(info), 'outcome': outcome_info})
        if feedback['terminal']:
            break
    after = {'events': brain.next_event_id, 'strength': float(brain.event_strength.sum()),
             'transitions': brain.core.transition_count}
    if not learning:
        assert before == after and supplied_rewards == 0
    metrics = env.private_metrics()
    metrics.update(learning=learning, teacher_supplied=False, before=before, after=after,
        supplied_rewards=supplied_rewards, goal_preserved=bool(all_goal), source_counts=sources,
        p50_ms=float(np.percentile(times, 50)) if times else 0.,
        p95_ms=float(np.percentile(times, 95)) if times else 0.,
        compute_seconds=sum(times)/1000, trace=trace, trajectory=env.private_trajectory(),
        actual_outcomes_observed=brain.outcomes_observed,
        retired_facts=brain.fact_capacity_retirements,
        core_edges=brain.core.excitatory.edge_count,
        disinhibition_edges=brain.core.disinhibitory.edge_count)
    return metrics


def inventory(brain):
    return {'facts': len(brain.event_slot_by_id), 'retirements': brain.fact_capacity_retirements,
        'teacher_events': int(np.count_nonzero(brain.event_kind == 2)),
        'valued_facts': int(np.count_nonzero(brain.event_strength > 0)),
        'terminal_outcomes': int(brain.event_terminal.sum()),
        'excitatory_edges': brain.core.excitatory.edge_count,
        'disinhibition_edges': brain.core.disinhibitory.edge_count,
        'real_observed_transitions': brain.core.transition_count}


def write(path, report):
    rows = report['training']
    pairs = report['pairs']
    report['summary'] = {'training_successes': sum(r['correct_arrival'] for r in rows),
        'training_trials': len(rows),
        'training_coverage': {start: {route: sum(r['correct_arrival'] and r['start_name'] == start
            and r['requested_route'] == route for r in rows) for route in ROUTES} for start in STARTS},
        'completed_pairs': len(pairs), 'predeclared_pairs': 6,
        'valid_prefixes': sum(p['valid_prefix'] for p in pairs),
        'pair_successes': sum(p['pair_success'] for p in pairs),
        'branch_successes': {b: sum(p['branches'].get(b, {}).get('strict_success', False) for p in pairs)
                             for b in ('stay', 'switch', 'switch_no_recurrence')}}
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def schedule():
    rng = np.random.default_rng(73194)
    warm = [(start, route, None, repeat) for repeat in range(2) for start in STARTS for route in ROUTES]
    later = [(start, route, None, repeat) for repeat in range(2, 6) for start in STARTS for route in ROUTES]
    later += [(start, route, 5.+5.*repeat, repeat) for repeat in range(2) for start in STARTS for route in ROUTES]
    rng.shuffle(warm)
    rng.shuffle(later)
    return [{'episode': i, 'seed': 7000+i, 'start': start, 'initial_route': route,
             'switch_time': switch, 'repeat': repeat} for i, (start, route, switch, repeat) in enumerate(warm+later)]


def probe(path, seed, initial, max_steps, progress):
    brain = IntegratedController.load(path)
    brain.reset_activity()
    brain.rng = np.random.default_rng(seed+100000)
    env = IntegrationEnvironment(seed=seed)
    env.reset(start=STARTS['middle'], route=initial)
    prefix = roll(env, brain, 100, learning=False)
    reasons = []
    if prefix['control_steps'] != 100 or prefix['terminated']:
        reasons.append('ended_before_fixed_intervention')
    if any(p['direction'] == 'left_to_right' for p in prefix['midline_crossings']):
        reasons.append('already_crossed_midline')
    if env.world.position[0] > prefix['route_gate_geometry']['left_x']+1e-10:
        reasons.append('already_entered_passage')
    pair = {'seed': seed, 'initial_route': initial, 'valid_prefix': not reasons,
            'invalid_reasons': reasons, 'prefix': prefix, 'branches': {}, 'pair_success': False}
    if reasons:
        return pair
    h = digest((env, brain))
    pair['prefix_hash'] = h
    for branch in ('stay', 'switch', 'switch_no_recurrence'):
        e, b = copy.deepcopy((env, brain))
        clone_hash = digest((e, b))
        assert clone_hash == h
        requested = initial if branch == 'stay' else ('top' if initial == 'bottom' else 'bottom')
        e.emit_route_cue(requested)
        if branch == 'switch_no_recurrence':
            # Remove only learned PFC recurrent/disinhibitory currents. Retain
            # external and recalled drives, state, facts and motor values.
            b.core.recurrent_enabled = False
            b.core.disinhibition_enabled = False
        row = roll(e, b, max_steps-100, learning=False)
        first = next((p for p in row['passages'] if p['direction'] == 'left_to_right'), None)
        last = (row['arrival_route_score'] or {}).get('passage')
        row.update(clone_hash=clone_hash, expected_route=requested,
            strict_success=bool(row['correct_arrival'] and row['goal_preserved'] and first and last
                and first['route'] == requested and first['time'] >= 10.6-1e-8 and last['time'] >= 10.6-1e-8))
        pair['branches'][branch] = row
        progress(pair, branch)
        del e, b
        gc.collect()
    pair['pair_success'] = all(pair['branches'][name]['strict_success'] for name in ('stay', 'switch'))
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'integration_development.json')
    parser.add_argument('--episodes', type=int, default=48)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--held-out', action='store_true')
    args = parser.parse_args()
    if args.evaluate_only and args.resume is None:
        raise ValueError('Evaluation requires --resume')
    if not 0 <= args.episodes <= 48:
        raise ValueError('The registered curriculum contains 48 episodes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    planned = schedule()
    seeds = HELD_OUT_SEEDS if args.held_out else DEVELOPMENT_SEEDS
    report = {'status': 'running', 'protocol': {'curriculum': planned, 'requested_episodes': args.episodes,
        'max_steps': args.steps, 'seed': 2026, 'width': 8192, 'texture_seed': 44017,
        'evaluation_seeds': seeds, 'evaluation_split': 'held_out' if args.held_out else 'development',
        'switch_frame': 100, 'cue_end': 10.6, 'evaluation_learning': False, 'evaluation_reward': 0,
        'teacher': False, 'model_private_inputs': False,
        'training_observes_actual_terminal_consequences': True,
        'limitation': 'Fixed acoustic goal/rule receptors; restored core equations with explicit numerical normalization, not complete original sensory networks.'},
        'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('integrated_controller.py', 'integration_environment.py', 'recurrent_core.py', 'run_integration.py')},
        'training': [], 'pairs': [], 'resume': None if args.resume is None else str(args.resume.resolve())}
    write(args.output, report)
    checkpoint = args.output.with_suffix('.npz')
    if not args.evaluate_only:
        brain = IntegratedController.load(args.resume) if args.resume else IntegratedController(seed=2026)
        # Resume is tied to the completed curriculum prefix in the prior JSON;
        # never silently repeat initial seeds after loading a pilot checkpoint.
        offset = 0
        if args.resume:
            previous_report = args.resume.with_suffix('.json')
            previous = json.loads(previous_report.read_text(encoding='utf-8'))
            report['training'] = previous['training']
            report['training_source_sha256'] = previous['source_sha256']
            offset = len(report['training'])
        report['birth_hash'] = brain.birth_hash
        for item in planned[offset:args.episodes]:
            brain.reset_activity()
            brain.rng = np.random.default_rng(item['seed']+100000)
            env = IntegrationEnvironment(seed=item['seed'])
            other = 'top' if item['initial_route'] == 'bottom' else 'bottom'
            cues = [] if item['switch_time'] is None else [(item['switch_time'], other)]
            env.reset(start=STARTS[item['start']], route=item['initial_route'], switch_schedule=cues)
            row = roll(env, brain, args.steps, learning=True)
            row.update(**item, start_name=item['start'])
            report['training'].append(row)
            write(args.output, report)
            print(f'TRAIN {len(report["training"])}/{args.episodes} {item["start"]} '
                  f'{item["initial_route"]} switch={item["switch_time"]} '
                  f'red={row["target_reached"]} success={row["correct_arrival"]} '
                  f'steps={row["control_steps"]} p95={row["p95_ms"]:.1f}ms '
                  f'edges={row["core_edges"]}', flush=True)
            if len(report['training']) % 6 == 0:
                brain.save(checkpoint)
        brain.reset_activity()
        brain.save(checkpoint)
        report['inventory'] = inventory(brain)
        del brain
        gc.collect()
    else:
        checkpoint = args.resume.resolve()
    report['checkpoint'] = str(checkpoint.resolve())
    if not args.skip_evaluation:
        def progress(pair, branch):
            report['in_progress'] = pair
            write(args.output, report)
            row = pair['branches'][branch]
            print(f'PROBE {pair["seed"]} {pair["initial_route"]} {branch} '
                  f'route={row["chosen_route"]} success={row["strict_success"]} '
                  f'p95={row["p95_ms"]:.1f}ms', flush=True)
        for seed in seeds:
            for initial in ROUTES:
                pair = probe(checkpoint, seed, initial, args.steps, progress)
                report['pairs'].append(pair)
                report.pop('in_progress', None)
                write(args.output, report)
                print(f'PAIR {seed} {initial} valid={pair["valid_prefix"]} pass={pair["pair_success"]}', flush=True)
    report['status'] = 'complete'
    report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write(args.output, report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
