"""Replay recorded muscles from the exact initial world, without a brain.

This is verification of recorded physics, not a demonstration or teacher run.
Visual stimuli are non-solid: their experiment-side relocation cannot affect
body integration. The replay compares position, heading, speed and touch only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_continuous_learning import _load_runner, _sha


def audit(report_path):
    runner=_load_runner()
    report_path=Path(report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status']=='complete'
    input_paths=[report_path,Path(report['base_life']),Path(report['complete_life_checkpoint']),
                 Path(report['actual_sensor_corpus'])]
    hashes={str(p):_sha(p) for p in input_paths}
    assert hashes[str(input_paths[1])]==report['base_sha256']
    assert hashes[str(input_paths[2])]==report['complete_life_sha256']
    initial=runner.LifeSession.load(input_paths[1])
    final=runner.LifeSession.load(input_paths[2])
    env=initial.environment
    start=env.world.position.copy()
    start_time=env.world.time
    start_path=env._path_length
    start_contacts=env._contact_steps
    n=report['current_steps']
    rows=report['records']
    assert len(rows)==n
    assert np.array_equal(start,report['starts']['position'])
    with np.load(input_paths[3],allow_pickle=False) as archive:
        corpus={key:archive[key] for key in ('muscles','time','body','index_time')}
    assert len(corpus['muscles'])==n
    trajectory=[start.copy()]
    matched=0
    for i,(action,row) in enumerate(zip(corpus['muscles'],rows)):
        assert row['step']==i+1
        assert abs(corpus['time'][i]-env.world.time)<1e-8
        before=env.world.position.copy()
        packet=env.step(action,dt=.1)
        assert np.array_equal(env.world.position,row['position']),i
        assert env.world.heading==row['heading'],i
        assert float(np.linalg.norm(env.world.velocity))==row['speed'],i
        assert float(np.linalg.norm(env.world.position-before))==row['path_increment'],i
        assert np.array_equal(packet['observation']['touch'],row['touch']),i
        assert abs(row['time']-env.world.time)<1e-8
        if i+1<n:
            expected=np.r_[packet['observation']['velocity'],packet['observation']['angular_velocity'],
                           packet['observation']['pain'],packet['observation']['touch']]
            assert np.array_equal(expected.astype(np.float32),corpus['body'][i+1]),i
        trajectory.append(env.world.position.copy());matched+=1
    assert np.array_equal(env.world.position,final.environment.world.position)
    assert np.array_equal(env.world.velocity,final.environment.world.velocity)
    assert env.world.heading==final.environment.world.heading
    assert env.world.angular_velocity==final.environment.world.angular_velocity
    assert abs(env.world.time-start_time-.1*n)<1e-7
    assert final.control_steps-initial.control_steps==n
    assert final.brain.runtime.clock.时间激活.sum()==1
    deltas=np.asarray([float(np.linalg.norm(trajectory[i+1]-trajectory[i])) for i in range(n)])
    # CPython's float sum uses compensated summation; summing np.float64
    # scalars instead differs in the last bits despite identical per-step data.
    independently=dict(path_length=sum(map(float,deltas)),net_displacement=float(np.linalg.norm(trajectory[-1]-start)),
        moving_steps=int(np.sum(deltas>1e-6)),contact_steps=sum(max(row['touch'])>.02 for row in rows),
        sources=dict(Counter(row['source'] for row in rows)))
    assert independently==report['movement']
    frame_delta=final.brain.runtime.frames_recorded-initial.brain.runtime.frames_recorded
    assert frame_delta in (n,n+1)
    assert int(final.brain.runtime.clock.当前时间)==int(initial.brain.runtime.clock.当前时间)+2*frame_delta
    assert np.array_equal(corpus['index_time'],corpus['index_time'][0]+2*np.arange(n))
    assert abs((env._path_length-start_path)-
               (final.environment._path_length-start_path))<1e-8
    longest_stationary=current=0
    for delta in deltas:
        current=current+1 if delta<=1e-6 else 0
        longest_stationary=max(longest_stationary,current)
    result=dict(status='complete',source_report=str(report_path),unchanged_input_sha256=hashes,
        physical_frames_replayed_exact=matched,position_heading_speed_touch_and_applied_muscles_exact=True,
        next_frame_raw_body_data_exact_float32=True,final_world_matches_full_checkpoint=True,
        movement=independently,integrated_substep_path_length=float(env._path_length-start_path),
        physics_contact_steps=int(env._contact_steps-start_contacts),longest_stationary_run_steps=longest_stationary,
        fraction_stationary=float(np.mean(deltas<=1e-6)),start_position=start.tolist(),
        final_position=env.world.position.tolist(),elapsed_physical_seconds=float(env.world.time-start_time),
        start_clock=report['starts']['clock'],final_clock=int(final.brain.runtime.clock.当前时间),
        sensory_frame_delta=frame_delta,one_shared_timeline=True,
        body_translation_observed=bool(independently['path_length']>.01),
        no_brain_training_or_teacher=True,no_position_velocity_or_heading_injection=True,
        limit='Exact physics replay demonstrates actual body motion, not goal-directed or learned route correctness.')
    assert hashes=={path:_sha(path) for path in hashes}
    output=report_path.with_name(report_path.stem+'_physical_audit.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'frames':matched,'movement':independently,
                      'longest_stationary_run_steps':longest_stationary,'sha256':_sha(output)},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path)
    audit(parser.parse_args().report)
