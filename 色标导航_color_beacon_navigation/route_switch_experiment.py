"""Predeclared autonomous two-route learning and paired mid-journey cues.

Only sensory packets enter the learner. Evaluator geometry is used for scoring
and prefix validity, never to choose muscles. Evaluation is frozen and rewardless.
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

from route_switch_controller import RouteSwitchController
from route_switch_environment import RouteSwitchEnv, ROUTE_NAMES

HERE = Path(__file__).resolve().parent
STARTS = {'middle': (1.3, 3.4, 0.), 'lower': (1.3, 1.4, 0.),
          'upper': (1.3, 6.6, 0.)}
SEEDS = (831, 931, 1031)
BRANCHES = ('stay', 'switch', 'no_cue', 'swapped_acoustic_mapping', 'clear_rule_after_cue')
INFO_FIELDS = ('source', 'goal_active', 'rule_active', 'pfc_active_count',
               'recall_count', 'recall_peak', 'muscle_currents',
               'feedback_pfc_active_count', 'sequence_supported_events', 'context_signature')


def create_environment(seed, texture_seed=None):
    if texture_seed is None:
        return RouteSwitchEnv(seed=seed)
    from textured_route_environment import TexturedRouteEnv
    return TexturedRouteEnv(seed=seed, texture_seed=texture_seed)


def write_report(path, report):
    training = report['training']
    pairs = report['pairs']
    report['summary'] = {
        'training_successes': sum(r['correct_arrival'] for r in training),
        'training_trials': len(training),
        'rewarded_routes': {route: sum(r['correct_arrival'] and r['requested_route'] == route
                                     for r in training) for route in ROUTE_NAMES},
        'paired_successes': sum(p.get('pair_success', False) for p in pairs),
        'predeclared_pairs': 6, 'completed_pairs': len(pairs),
        'valid_prefixes': sum(p['valid_prefix'] for p in pairs),
        'branch_successes': {branch: sum(p.get('branches', {}).get(branch, {}).get(
            'strict_success', False) for p in pairs) for branch in BRANCHES},
    }
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run_segment(env, brain, steps, *, learning, reward_enabled=False,
                clear_rule_time=None):
    trace, timings, sources = [], [], {}
    before_events = int(np.count_nonzero(brain.event_ids >= 0))
    before_strength = float(brain.event_strength.sum())
    reward_details = None
    cleared = False
    goal_preserved = True
    packet = env.sensor_packet()
    for frame in range(steps):
        if env.private_metrics()['terminated']:
            break
        begin = time.perf_counter()
        if clear_rule_time is not None and not cleared and env.world.time >= clear_rule_time-1e-8:
            brain.intervene_rule_state([1., 0., 0.])
            cleared = True
        muscles, info = brain.observe_then_act(packet, learning=learning)
        assert not info['teaching']
        goal_preserved &= info['goal_active'] == [0]
        sources[info['source']] = sources.get(info['source'], 0)+1
        packet = env.step(muscles, .1)
        metrics = env.private_metrics()
        if metrics['terminated'] and metrics['correct_arrival'] and reward_enabled:
            assert learning
            brain.receive_reward(1.)
            reward_details = copy.deepcopy(brain.last_reward_info)
        timings.append((time.perf_counter()-begin)*1000)
        if frame % 10 == 0 or frame < 8 or metrics['terminated']:
            trace.append({'time': float(env.world.time), 'position': env.world.position.tolist(),
                'heading': float(env.world.heading), 'muscles': np.asarray(muscles).tolist(),
                'info': {key: info[key] for key in INFO_FIELDS if key in info}})
        if metrics['terminated']:
            break
    result = env.private_metrics()
    result.update(learning=learning, reward_supplied=reward_details is not None,
        reward_details=reward_details, teacher_supplied=False,
        events_before=before_events, events_after=int(np.count_nonzero(brain.event_ids >= 0)),
        strength_before=before_strength, strength_after=float(brain.event_strength.sum()),
        p50_ms=float(np.percentile(timings, 50)) if timings else None,
        p95_ms=float(np.percentile(timings, 95)) if timings else None,
        wall_seconds=sum(timings)/1000, goal_preserved=bool(goal_preserved),
        rule_state_cleared=cleared, source_counts=sources, trace=trace,
        trajectory=env.private_trajectory())
    if not learning:
        assert result['events_before'] == result['events_after']
        assert result['strength_before'] == result['strength_after']
        assert not result['reward_supplied']
    return result


def paired_probe(checkpoint, *, seed, initial_route, max_steps, record_progress,
                 texture_seed=None):
    from route_state_digest import digest
    brain = RouteSwitchController.load(checkpoint)
    brain.reset_activity()
    brain.rng = np.random.default_rng(seed+100000)
    env = create_environment(seed, texture_seed)
    env.reset(start='left_middle', target='red', route=initial_route)
    prefix = run_segment(env, brain, 100, learning=False)
    reasons = []
    if prefix['control_steps'] != 100 or prefix['terminated']:
        reasons.append('ended_before_fixed_100_frame_intervention')
    if any(p['direction'] == 'left_to_right' for p in prefix['midline_crossings']):
        reasons.append('already_crossed_midline')
    if env.world.position[0] > prefix['route_gate_geometry']['left_x']+1e-10:
        reasons.append('no_longer_on_left_approach')
    result = dict(seed=seed, initial_route=initial_route, switch_frame=100,
        switch_time=float(env.world.time), cue_end_time=10.6, valid_prefix=not reasons,
        invalid_reasons=reasons, prefix=prefix, branches={}, pair_success=False)
    if reasons:
        return result
    prefix_hash = digest((env, brain))
    result['full_prefix_state_sha256'] = prefix_hash
    alternative = next(route for route in ROUTE_NAMES if route != initial_route)
    for branch in BRANCHES:
        branch_env, branch_brain = copy.deepcopy((env, brain))
        # Full arrays, indices, caches, neural/physical state and RNG, before cue.
        clone_hash = digest((branch_env, branch_brain))
        assert clone_hash == prefix_hash
        requested = initial_route if branch in ('stay', 'no_cue') else alternative
        if branch == 'swapped_acoustic_mapping':
            branch_env.set_route_tone_permutation((1, 0))
        if branch != 'no_cue':
            branch_env.emit_route_cue(requested)
        row = run_segment(branch_env, branch_brain, max_steps-100, learning=False,
            clear_rule_time=10.6 if branch == 'clear_rule_after_cue' else None)
        passage = (row['arrival_route_score'] or {}).get('passage')
        post_cue = bool(passage and passage['time'] >= 10.6-1e-8)
        first_passage = next((p for p in row['passages'] if p['direction'] == 'left_to_right'), None)
        first_after_cue = bool(first_passage and first_passage['time'] >= 10.6-1e-8)
        # Require the FIRST complete passage to be the requested one too: no
        # opposite crossing followed by a late return masquerades as clean switch.
        row.update(branch=branch, clone_state_sha256=clone_hash,
            expected_route=requested, qualifying_passage_after_cue=post_cue,
            first_forward_passage=first_passage, first_passage_after_cue=first_after_cue,
            strict_success=bool(row['correct_arrival'] and post_cue and first_after_cue and
                row['chosen_route'] == requested and row['goal_preserved']))
        result['branches'][branch] = row
        record_progress(result, branch)
        del branch_env, branch_brain
        gc.collect()
    result['pair_success'] = all(result['branches'][b]['strict_success'] for b in ('stay', 'switch'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'route_switch_evaluation.json')
    parser.add_argument('--evaluate-only', type=Path)
    parser.add_argument('--texture-seed', type=int,
                        help='Optional visible wall texture; physics, cues and training budget stay fixed.')
    parser.add_argument('--width', type=int, default=8192)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix('.npz')
    schedule = [(route, name, repeat) for repeat in range(6)
                for name in STARTS for route in ROUTE_NAMES]
    np.random.default_rng(2026+937).shuffle(schedule)
    report = {'status': 'running', 'protocol': {
        'birth_seed': 2026, 'mixed_neurons': args.width, 'max_events': 100000,
        'training_episodes': 36, 'max_steps': 1600, 'dt': .1,
        'starts_independent_of_route': STARTS,
        'reward_rule': 'One +1 only at first red arrival after requested passage; wrong first arrival ends trial.',
        'schedule': [{'episode': i, 'seed': 4000+i, 'route': route, 'start': name, 'repeat': repeat}
                     for i, (route, name, repeat) in enumerate(schedule)],
        'evaluation_seeds': SEEDS, 'initial_routes': ROUTE_NAMES,
        'fixed_switch_frame': 100, 'denominator_includes_invalid_prefixes': True,
        'pair_success': 'Both stay and switch choose their opposite requested first passages after 10.6s and reach same red.',
        'evaluation_learning': False, 'evaluation_reward': False,
        'branches': BRANCHES, 'teacher_demonstrations': 0,
        'wall_texture_seed': args.texture_seed,
        'limitation': 'Two learned absolute auditory route cues, not a generic relative instruction or natural language.'},
        'code_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('environment.py', 'associative_controller.py', 'route_switch_environment.py',
             'route_switch_controller.py', 'route_switch_experiment.py')},
        'training': [], 'pairs': []}
    if args.texture_seed is not None:
        report['code_sha256']['textured_route_environment.py'] = hashlib.sha256(
            (HERE/'textured_route_environment.py').read_bytes()).hexdigest()
    report['code_sha256']['route_state_digest.py'] = hashlib.sha256(
        (HERE/'route_state_digest.py').read_bytes()).hexdigest()
    # Persist protocol before any training or probe outcome is known.
    write_report(args.output, report)
    if args.evaluate_only:
        checkpoint = args.evaluate_only.resolve()
        report['evaluation_only_checkpoint'] = str(checkpoint)
    else:
        brain = RouteSwitchController(seed=2026, mixed_neurons=args.width, max_events=100000)
        report['birth_hash'] = brain.birth_hash
        for episode, (route, name, repeat) in enumerate(schedule):
            brain.reset_activity()
            brain.rng = np.random.default_rng(104000+episode)
            env = create_environment(4000+episode, args.texture_seed)
            env.reset(start=STARTS[name], target='red', route=route)
            row = run_segment(env, brain, 1600, learning=True, reward_enabled=True)
            row.update(episode=episode, seed=4000+episode, start_name=name, repeat=repeat)
            report['training'].append(row)
            write_report(args.output, report)
            print(f'TRAIN {episode+1}/36 {name} {route} red={row["target_reached"]} '
                  f'correct={row["correct_arrival"]} steps={row["control_steps"]} '
                  f'p95={row["p95_ms"]:.2f}ms events={row["events_after"]}', flush=True)
            if (episode+1) % 6 == 0:
                brain.save(checkpoint)
        brain.reset_activity()
        brain.save(checkpoint)
        report['memory_inventory'] = {
            'events': int(np.count_nonzero(brain.event_ids >= 0)),
            'teacher_events': int(np.count_nonzero(brain.event_kind == 2)),
            'reinforced_events': int(np.count_nonzero(brain.event_strength > 0)),
            'weight_sum': float(brain.event_strength.sum())}
        assert report['memory_inventory']['teacher_events'] == 0
        del brain, env
        gc.collect()
    report['checkpoint'] = str(checkpoint.resolve())
    write_report(args.output, report)
    def progress(pair, branch):
        report['in_progress_pair'] = pair
        write_report(args.output, report)
        row = pair['branches'][branch]
        print(f'PROBE {pair["seed"]} {pair["initial_route"]} {branch} '
              f'red={row["target_reached"]} route={row["chosen_route"]} '
              f'strict={row["strict_success"]}', flush=True)
    for seed in SEEDS:
        for initial in ROUTE_NAMES:
            pair = paired_probe(checkpoint, seed=seed, initial_route=initial,
                                max_steps=1600, record_progress=progress,
                                texture_seed=args.texture_seed)
            report['pairs'].append(pair)
            report.pop('in_progress_pair', None)
            write_report(args.output, report)
            print(f'PAIR {seed} {initial} valid={pair["valid_prefix"]} '
                  f'passed={pair["pair_success"]}', flush=True)
    report['status'] = 'complete'
    report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_report(args.output, report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
