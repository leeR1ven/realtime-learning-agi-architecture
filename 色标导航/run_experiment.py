"""Demonstration collection, neural learning and teacher-free navigation tests.

The privileged teacher is imported ONLY by collect_demonstrations. Autonomous
rollouts receive sensor packets and have no access to teacher actions or routes.
Ground-truth positions are collected by the evaluator solely for scoring/traces.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from environment import BeaconNavigationEnv, BEACON_NAMES

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def collect_demonstrations(*, scenario='barrier', repeats=1):
    from teacher import MuscleTeacher
    env = BeaconNavigationEnv(scenario=scenario, seed=260906)
    rng = np.random.default_rng(476)
    sensory = {key: [] for key in ('ray_distances', 'ray_colors', 'velocity',
                                  'angular_velocity', 'pain', 'touch', 'waveform', 'teacher_muscles')}
    offsets, rows = [0], []
    schedule = [(start, target, repetition) for repetition in range(repeats)
                for start in env.scene.starts for target in BEACON_NAMES]
    rng.shuffle(schedule)
    started = time.perf_counter()
    for start, target, repetition in schedule:
        pose = np.array(env.scene.starts[start], dtype=float)
        if repetition:
            pose += rng.uniform([-.13, -.13, -.18], [.13, .13, .18])
        packet = env.reset(start=pose, target=target)
        destination = env.scene.beacon_positions[BEACON_NAMES.index(target)]
        teacher = MuscleTeacher(env.world, destination)
        for frame in range(450):
            muscles = teacher.muscles()
            for key in ('ray_distances', 'ray_colors', 'velocity', 'angular_velocity', 'pain', 'touch'):
                sensory[key].append(np.asarray(packet['observation'][key]).copy())
            sensory['waveform'].append(packet['waveform'].astype(np.float32))
            sensory['teacher_muscles'].append(muscles)
            packet = env.step(muscles, .1)
            if env.private_metrics()['target_reached']:
                break
        metric = env.private_metrics()
        metric.update(start_label=start, demonstration_only=True, repetition=repetition,
                      begin=offsets[-1], end=len(sensory['waveform']))
        offsets.append(len(sensory['waveform']))
        rows.append(metric)
        print(f"DEMONSTRATION {start}/{target} frames={frame+1} arrived={metric['target_reached']} contacts={metric['contact_steps']}", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / 'demonstrations.npz'
    np.savez_compressed(path, **{key: np.asarray(value) for key,value in sensory.items()},
                        offsets=np.asarray(offsets, dtype=np.int64))
    meta = {'scope': 'Privileged teacher demonstrations, not autonomous model results',
            'scenario': scenario, 'repeats': repeats, 'episodes': rows,
            'frames': len(sensory['waveform']), 'wall_seconds': time.perf_counter()-started,
            'all_teacher_routes_arrived': all(row['target_reached'] for row in rows),
            'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'demonstrations.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return meta


def packet_at(data, frame):
    return {'observation': {key: data[key][frame] for key in
            ('ray_distances', 'ray_colors', 'velocity', 'angular_velocity', 'pain', 'touch')},
            'waveform': data['waveform'][frame]}


def train_controller():
    from associative_controller import AssociativeController
    model = AssociativeController(ray_count=31, seed=2026)
    started = time.perf_counter()
    durations = []
    with np.load(OUT/'demonstrations.npz', allow_pickle=False) as data:
        offsets = data['offsets']
        for begin, end in zip(offsets[:-1], offsets[1:]):
            model.reset_activity()
            for frame in range(int(begin), int(end)):
                t0 = time.perf_counter()
                _, info = model.observe_then_act(packet_at(data,frame),
                    teacher_muscles=data['teacher_muscles'][frame], learning=True)
                durations.append((time.perf_counter()-t0)*1000)
            print(f"LEARN frames={int(end)} info={json.dumps(info, ensure_ascii=False, default=lambda v: np.asarray(v).tolist())[:350]}", flush=True)
    model.reset_activity()
    model.save(OUT/'navigation_brain.npz')
    stats = {'frames': len(durations), 'wall_seconds': time.perf_counter()-started,
             'p50_ms': float(np.percentile(durations,50)), 'p95_ms': float(np.percentile(durations,95)),
             'max_ms': max(durations), 'last_info': info,
             'teacher_is_removed_for_evaluation': True}
    (OUT/'training.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False,
        default=lambda v: np.asarray(v).tolist()), encoding='utf-8')
    return stats


def rollout(model, *, start, target, scenario='barrier', max_steps=420,
            seed=873, condition='intact', learning=False, tone_permutation=None):
    model = copy.deepcopy(model)
    env = BeaconNavigationEnv(scenario=scenario, seed=seed, tone_permutation=tone_permutation)
    packet = env.reset(start=start, target=target)
    model.reset_activity()
    flags = {'context_enabled': condition != 'no_context',
             'view_enabled': condition != 'no_view',
             'motor_memory_enabled': condition != 'no_motor_memory'}
    for name, value in flags.items():
        setattr(model, name, value)
    trace, times = [], []
    for frame in range(max_steps):
        t0 = time.perf_counter()
        muscles, info = model.observe_then_act(packet, learning=learning)
        packet = env.step(muscles, .1)
        times.append((time.perf_counter()-t0)*1000)
        if frame % 2 == 0:
            trace.append({'frame':frame, 'position':env.world.position.tolist(),
                          'heading':env.world.heading, 'muscles':np.asarray(muscles).tolist(),
                          'info':info})
        if env.private_metrics()['target_reached']:
            break
    result = env.private_metrics()
    result.update(condition=condition, teacher_supplied=False, learning=learning,
                  p50_ms=float(np.percentile(times,50)), p95_ms=float(np.percentile(times,95)),
                  max_ms=max(times), trace=trace)
    return result


def evaluate_controller(*, quick=False):
    from associative_controller import AssociativeController
    model = AssociativeController.load(OUT/'navigation_brain.npz')
    env = BeaconNavigationEnv(seed=943)
    cases = [(start, target) for start in env.scene.starts for target in BEACON_NAMES]
    if quick:
        cases = [('left_middle', target) for target in BEACON_NAMES]
    rows = []
    for start, target in cases:
        row = rollout(model, start=start, target=target)
        row['split'] = 'demonstrated_start_new_rollout'
        rows.append(row)
        print(f"AUTONOMOUS {start}/{target}: arrived={row['target_reached']} remaining={row['remaining_target_distance']:.2f} contacts={row['contact_steps']}", flush=True)
    if not quick:
        for start in ('left_middle','right_middle'):
            pose = np.array(env.scene.starts[start]) + np.array([.19,-.17,.12])
            for target in BEACON_NAMES:
                row = rollout(model, start=pose, target=target)
                row['split'] = 'untrained_start_offset'
                rows.append(row)
                print(f"OFFSET {start}/{target}: {row['target_reached']}", flush=True)
        for condition in ('no_context','no_view','no_motor_memory'):
            for target in BEACON_NAMES:
                row = rollout(model, start='left_middle', target=target, condition=condition)
                row['split'] = 'ablation'
                rows.append(row)
                print(f"ABLATION {condition}/{target}: {row['target_reached']}", flush=True)
    report = {'scope':'Sensor-only associative navigation after explicit muscle demonstrations; not AGI or language',
              'demonstrations':str(OUT/'demonstrations.json'), 'rows':rows,
              'summary': {}}
    for split, condition in sorted({(r['split'],r['condition']) for r in rows}):
        chosen = [r for r in rows if r['split']==split and r['condition']==condition]
        report['summary'][split+'/'+condition] = {'arrived':sum(r['target_reached'] for r in chosen),
            'trials':len(chosen), 'first_beacon_correct':sum(r['first_beacon_correct'] for r in chosen),
            'mean_contacts':float(np.mean([r['contact_steps'] for r in chosen])),
            'mean_p95_ms':float(np.mean([r['p95_ms'] for r in chosen]))}
    path = OUT / ('evaluation_quick.json' if quick else 'evaluation.json')
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,
        default=lambda v: np.asarray(v).tolist()),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('collect','train','evaluate','all'), default='all')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.stage in ('collect','all'):
        collect_demonstrations(repeats=args.repeats)
    if args.stage in ('train','all'):
        train_controller()
    if args.stage in ('evaluate','all'):
        evaluate_controller(quick=args.quick)


if __name__=='__main__':
    main()
