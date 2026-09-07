"""Single-life, uninterrupted throughput measurement with observed GC events.

Exactly one saved life is loaded. No snapshot, clone, serialization, recall
assessment, stimulus rescheduling, GC tuning or parameter change occurs during
the measured 180 real online-learning control steps. All measured frames remain
in the report. The temporary experience is never saved over a formal life.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from viewer import LifeSession
from fast_pfc_currents import FastPFC
from fast_hippocampal_index import FastHippocampalIndex


HERE=Path(__file__).resolve().parent
STEPS=180


def summary(rows):
    values=np.array([row['latency_ms'] for row in rows],float)
    return dict(steps=len(rows),median_ms=float(np.median(values)),mean_ms=float(values.mean()),
        p95_ms=float(np.percentile(values,95)),max_ms=float(values.max()),
        total_control_ms=float(values.sum()),steps_at_or_above_100ms=int((values>=100).sum()),
        steps_with_gc=sum(row['gc_count']>0 for row in rows),
        gc_ms_inside_steps=sum(row['gc_ms'] for row in rows))


def run(checkpoint):
    paths=[checkpoint,HERE/'results'/'living_original.npz']
    paths += [HERE/name for name in ('brain.py','original_runtime.py','viewer.py','sensory_adapter.py',
        'experience_environment.py','contact_reflex.py','fast_pfc_currents.py',
        'fast_hippocampal_index.py','weight_dependent_plasticity.py')]
    protected={str(path):hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None for path in paths}
    load_started=time.perf_counter()
    session=LifeSession.load(checkpoint)
    load_seconds=time.perf_counter()-load_started
    assert type(session.brain.runtime.pfc) is FastPFC
    assert isinstance(session.brain.runtime.index,FastHippocampalIndex)
    runtime=session.brain.runtime
    clock=runtime.clock
    before=dict(control_steps=session.control_steps,clock=int(clock.当前时间),
        sensory_frames=runtime.frames_recorded,internal_steps=runtime.internal_steps_recorded,
        physical_time=float(session.environment.world.time),position=session.environment.world.position.tolist(),
        one_active_time_cell=int(clock.时间激活.sum()),
        thought_count=int(session.brain.thought.sum()),E_edges=runtime.pfc.联想.条数())
    assert before['one_active_time_cell']==1
    rows=[]
    events=[]
    open_events={}
    current_step=None
    gc_configuration=dict(enabled=gc.isenabled(),thresholds=list(gc.get_threshold()),
                          initial_counts=list(gc.get_count()),initial_stats=gc.get_stats())

    def gc_event(phase,info):
        now=time.perf_counter_ns()
        generation=info.get('generation',-1)
        if phase=='start':
            open_events[generation]=(now,current_step)
        elif phase=='stop':
            beginning=open_events.pop(generation,None)
            if beginning is not None:
                tick,start_step=beginning
                events.append(dict(step_at_start=start_step,step_at_end=current_step,
                    generation=generation,duration_ms=(now-tick)/1e6,
                    collected=info.get('collected'),uncollectable=info.get('uncollectable')))

    gc.callbacks.append(gc_event)
    measured_start=time.perf_counter_ns()
    previous_position=session.environment.world.position.copy()
    expected_clock=int(clock.当前时间)
    try:
        for frame in range(STEPS):
            current_step=frame+1
            first_event=len(events)
            start=time.perf_counter_ns()
            session.tick()
            end=time.perf_counter_ns()
            current_step=None
            # Only lightweight physical/counter recording between controls.
            # This work and any between-step GC remain in total loop wall time.
            frame_events=[event for event in events[first_event:] if event['step_at_start']==frame+1]
            expected_clock=(expected_clock+clock.每帧步数)%clock.时间神经元数量
            assert int(clock.当前时间)==expected_clock
            assert clock.时间激活[expected_clock]
            position=session.environment.world.position.copy()
            packet=session.packet
            rows.append(dict(step=frame+1,latency_ms=(end-start)/1e6,
                gc_count=len(frame_events),gc_ms=sum(event['duration_ms'] for event in frame_events),
                gc_generations=[event['generation'] for event in frame_events],
                clock=int(clock.当前时间),physical_time=float(session.environment.world.time),
                position=position.tolist(),heading=float(session.environment.world.heading),
                path_increment=float(np.linalg.norm(position-previous_position)),
                muscles=session.executed.tolist(),source=session.decision.get('source'),
                pain=float(packet['observation']['pain']),touch=np.asarray(packet['observation']['touch']).tolist(),
                thought_count=int(session.brain.thought.sum()),E_edges=runtime.pfc.联想.条数()))
            previous_position=position
    finally:
        measured_end=time.perf_counter_ns()
        gc.callbacks.remove(gc_event)
    assert len(rows)==STEPS
    assert session.control_steps-before['control_steps']==STEPS
    assert runtime.frames_recorded-before['sensory_frames']==STEPS
    assert runtime.internal_steps_recorded-before['internal_steps']==STEPS*clock.每帧步数
    assert int(clock.时间激活.sum())==1
    assert np.isclose(session.environment.world.time-before['physical_time'],STEPS*.1)
    all_stats=summary(rows)
    no_gc=[row for row in rows if not row['gc_count']]
    with_gc=[row for row in rows if row['gc_count']]
    elapsed_ms=(measured_end-measured_start)/1e6
    final=dict(control_steps=session.control_steps,clock=int(clock.当前时间),
        sensory_frames=runtime.frames_recorded,internal_steps=runtime.internal_steps_recorded,
        physical_time=float(session.environment.world.time),position=session.environment.world.position.tolist(),
        one_active_time_cell=int(clock.时间激活.sum()),thought_count=int(session.brain.thought.sum()),
        E_edges=runtime.pfc.联想.条数())
    after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None for path in paths}
    assert protected==after,'Frozen source or formal life changed concurrently'
    return dict(passed=True,checkpoint=str(checkpoint),loaded_life_instances=1,actual_learning_steps=STEPS,
        no_snapshots_clones_serialization_or_recall_assessment_during_measurement=True,
        no_parameter_or_gc_configuration_changes=True,no_warmup_or_slow_frames_removed=True,
        temporary_experience_not_saved=True,teacher_actions=0,rewards=0,
        load_seconds=load_seconds,starts=before,ends=final,all_180_steps=all_stats,
        first_30_steps=summary(rows[:30]),last_150_steps=summary(rows[30:]),
        with_gc_steps=None if not with_gc else summary(with_gc),
        without_gc_steps=None if not no_gc else summary(no_gc),
        gc_configuration=gc_configuration,gc_events=events,
        gc_ms_between_steps=sum(event['duration_ms'] for event in events if event['step_at_start'] is None),
        total_measured_loop_wall_ms=elapsed_ms,loop_wall_control_hz=STEPS/(elapsed_ms/1000),
        physical_seconds=STEPS*.1,physical_path_length=sum(row['path_increment'] for row in rows),
        net_displacement=float(np.linalg.norm(np.array(final['position'])-np.array(before['position']))),
        contact_steps=sum(any(row['touch']) for row in rows),
        p95_below_100ms=all_stats['p95_ms']<100,max_below_100ms=all_stats['max_ms']<100,
        step_records=rows,protected_source_and_life_hashes=protected,
        limitations=['The overall180-step result is primary; first/last and GC groups are additional descriptions, not excluded frames.',
            'The sample is one fixed continuation of this learned physical state, without a new stimulus curriculum.',
            'GC callbacks and lightweight between-step recording add measurement overhead; no full-state checks or serialization occur in the measured loop.',
            'A p95 below100ms would only describe this window; it would not guarantee a hard real-time deadline for every future step.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results'/'active_body_additive_600.npz')
    parser.add_argument('--output',type=Path,default=HERE/'results'/'single_life_throughput_180.json')
    args=parser.parse_args()
    report=run(args.checkpoint)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({key:value for key,value in report.items() if key not in
        ('step_records','protected_source_and_life_hashes','limitations','gc_events')},ensure_ascii=True))


if __name__=='__main__':main()
