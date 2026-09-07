"""Pure self-exploration: sensory packets, real muscles, terminal reward only.

No teacher/planner imports, demonstrations, coordinate inputs, progress reward,
or target-dependent innate motor policy. Private world state is evaluator-only.
Learning and action use the same controller step. Frozen probes are explicit
evaluation interventions; live trials keep experience recording enabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from environment import BeaconNavigationEnv, BEACON_NAMES
from associative_controller import AssociativeController

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'


def compact_info(info):
    fields = ('frame','context_active','pfc_active_count','recall_count','recall_peak',
              'source','memory_events','trusted_events','muscle_currents',
              'feedback_pfc_active_count','sequence_supported_events','feedback_support')
    return {key:info[key] for key in fields if key in info}


def live_trial(model, *, target, start='left_middle', seed=771, max_steps=900,
               learning=True, reward_enabled=True, condition='intact', trace_stride=10):
    env = BeaconNavigationEnv(seed=seed)
    packet = env.reset(start=start, target=target)
    model.reset_activity()
    model.context_enabled = condition != 'no_context'
    model.view_enabled = condition != 'no_view'
    model.motor_memory_enabled = condition != 'no_motor_memory'
    model.sequence_enabled = condition != 'no_sequence'
    model.recall_feedback_enabled = condition != 'no_recall_feedback'
    if condition == 'shuffled_sequence':
        model.shuffle_sequence_connections(seed=351)
    # Matched exploration randomness across conditions; no target-derived seed.
    model.rng = np.random.default_rng(seed+100000)
    times, trace, source_counts = [], [], {}
    rewarded = False
    events_before = int(np.count_nonzero(model.event_ids >= 0))
    strength_before = float(model.event_strength.sum())
    for frame in range(max_steps):
        t0 = time.perf_counter()
        muscles, info = model.observe_then_act(packet, learning=learning)
        source = info.get('source','unknown')
        source_counts[source] = source_counts.get(source,0)+1
        packet = env.step(muscles, .1)
        reached = env.private_metrics()['target_reached']
        if reached and learning and reward_enabled and not rewarded:
            model.receive_reward(1.)
            rewarded = True
        times.append((time.perf_counter()-t0)*1000)
        if frame % trace_stride == 0 or reached:
            trace.append({'frame':frame,'position':env.world.position.tolist(),
                          'heading':env.world.heading,'muscles':np.asarray(muscles).tolist(),
                          'info':compact_info(info)})
        if reached:
            break
    result = env.private_metrics()
    result.update(condition=condition, learning=learning, teacher_supplied=False,
        reward_supplied=rewarded, max_steps=max_steps, seed=seed,
        p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),
        max_ms=max(times),wall_seconds=sum(times)/1000,
        events_before=events_before,events_after=int(np.count_nonzero(model.event_ids>=0)),
        strength_before=strength_before,strength_after=float(model.event_strength.sum()),
        source_counts=source_counts,reward_details=getattr(model,'last_reward_info',{}),trace=trace)
    return result


def stats(rows):
    if not rows:
        return {}
    return {'successes':sum(r['target_reached'] for r in rows),'trials':len(rows),
            'first_beacon_correct':sum(r['first_beacon_correct'] for r in rows),
            'mean_steps':float(np.mean([r['control_steps'] for r in rows])),
            'mean_contact_steps':float(np.mean([r['contact_steps'] for r in rows])),
            'mean_p95_ms':float(np.mean([r['p95_ms'] for r in rows]))}


def write_report(path, report):
    report['summary'] = {split:stats([r for r in report['rows'] if r['split']==split])
                         for split in sorted({r['split'] for r in report['rows']})}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--steps',type=int,default=900)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--width',type=int,default=4096)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--evaluation-only',action='store_true')
    parser.add_argument('--skip-ablations',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    report_path = args.output or OUT/'autonomous_evaluation.json'
    checkpoint = report_path.with_suffix('.npz')
    model = (AssociativeController.load(args.resume) if args.resume else
             AssociativeController(seed=args.seed,mixed_neurons=args.width,max_events=50000))
    report = {'scope':'Pure exploration and scalar terminal reward; no teacher or planner in this run',
        'seed':args.seed,'width':args.width,'steps_per_trial':args.steps,
        'resume':None if args.resume is None else str(args.resume.resolve()),
        'rows':[], 'code_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ('environment.py','associative_controller.py','autonomous_experiment.py')},
        'reward_rule':'Exactly one +1 after first actual arrival at the signalled beacon; no distance shaping',
        'rows_include_failures':True}
    def record(row, split):
        row['split'] = split
        report['rows'].append(row)
        write_report(report_path,report)
        print(f"{split} target={row['target']} reached={row['target_reached']} "
              f"steps={row['control_steps']} contacts={row['contact_steps']} "
              f"reinforced={row['strength_after']-row['strength_before']:.1f}",flush=True)
    if not args.evaluation_only:
        for target in BEACON_NAMES:
            newborn = AssociativeController(seed=args.seed,mixed_neurons=args.width,max_events=50000)
            record(live_trial(newborn,target=target,seed=831,max_steps=args.steps,learning=False),
                   'newborn_matched_probe')
        rng = np.random.default_rng(args.seed+937)
        schedule=[]
        while len(schedule)<args.episodes:
            schedule.extend(rng.permutation(BEACON_NAMES).tolist())
        for episode,target in enumerate(schedule[:args.episodes]):
            record(live_trial(model,target=target,seed=2000+episode,max_steps=args.steps),
                   'self_exploration_training')
            model.save(checkpoint)
    if args.evaluation_only and not args.resume:
        raise ValueError('--evaluation-only requires --resume')
    model.save(checkpoint)
    for target in BEACON_NAMES:
        probe = AssociativeController.load(checkpoint)
        record(live_trial(probe,target=target,seed=831,max_steps=args.steps,learning=False),
               'learned_matched_probe')
    # No prior demonstration or experience at these two altered poses.
    for pose in ((1.49,3.23,.12),(1.1,3.56,-.14)):
        for target in BEACON_NAMES:
            probe = AssociativeController.load(checkpoint)
            record(live_trial(probe,target=target,start=pose,seed=931,max_steps=args.steps,learning=False),
                   'new_start_probe')
    if not args.skip_ablations:
        for condition in ('no_context','no_view','no_motor_memory','no_sequence','shuffled_sequence','no_recall_feedback'):
            for target in BEACON_NAMES:
                probe = AssociativeController.load(checkpoint)
                record(live_trial(probe,target=target,seed=831,max_steps=args.steps,
                                  learning=False,condition=condition),condition)
    model.reset_activity()
    model.save(OUT/'navigation_brain.npz')
    write_report(report_path,report)
    print(json.dumps({'report':str(report_path),'summary':report['summary']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
