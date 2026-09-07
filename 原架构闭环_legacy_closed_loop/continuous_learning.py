"""One uninterrupted physical life, increasingly diverse sensory experience.

Stimulus labels and reference neuron sets are experiment-side diagnostics. Brain
input remains actual rays/body/PCM; no teacher actions or task-labelled memories.
Every checkpoint contains the full single-timeline brain and physical state.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from brain import OriginalBrain
from checkpoint import load_tree, save_tree
from dynamic_calibration import frozen_digest
from experience_environment import ExperienceEnvironment
from run_closed_loop import plain
sys.path.insert(0,str(Path(__file__).resolve().parent))
from viewer import LifeSession
from audit_continuous_learning import initial_catalog

HERE=Path(__file__).resolve().parent
PAIRS=(('red',220.),('green',440.),('blue',660.),
       ('red',990.),('green',1320.),('blue',1750.))


def apply_parameters(brain, changes):
    """Only original public parameters; no connection content or clock changes."""
    for name,value in changes.items():
        if name not in asdict(brain.config): raise ValueError('Unknown BrainConfig '+name)
        setattr(brain.config,name,value)
    tables={'pfc':(brain.runtime.pfc.联想,brain.runtime.pfc.去抑),
            'memory':tuple(t for m in brain.runtime.memories.values()
                           for t in (m.特征到时间,m.时间到特征))}
    for prefix,items in tables.items():
        for suffix,original in (('rate','衰减率'),('interval','衰减间隔'),('warning','警戒线')):
            value=getattr(brain.config,prefix+'_decay_'+suffix)
            if value is not None:
                for table in items: setattr(table,original,value)
    for table in tables['pfc']:
        table.强化量=brain.config.pfc_hebb_increment
        table.消失下限=brain.config.pfc_decay_floor
    brain.runtime.set_pfc_plasticity(None if brain.config.pfc_weight_ceiling is None else {
        'ceiling': brain.config.pfc_weight_ceiling,
        'depression': brain.config.pfc_weight_depression})


def present(session, index, event):
    """Move an ordinary visible stimulus; never move or steer the body."""
    env=session.environment
    color,frequency=PAIRS[index]
    nominal_angle=(0.,-.35,.35,0.,-.65,.65)[event%6]
    distance=(1.4,1.,1.8,2.2)[event%4]
    attempts=[]
    for angle in (nominal_angle,0.,-.8,.8,-1.2,1.2):
        heading=env.world.heading+angle
        position=np.clip(env.world.position+distance*np.array([np.cos(heading),np.sin(heading)]),
                         (.31,.31),(env.scene.width-.31,env.scene.height-.31))
        try:
            packet=env.setup_pairing(color,frequency,duration=.8,
                                    presentation_position=position)
        except ValueError as exc:
            attempts.append(str(exc)); continue
        session.packet=packet
        session.presentation={'color':color,'frequency':frequency}
        return dict(pair=index,color=color,frequency=frequency,visible=True,
                    retries=len(attempts),position=position.tolist(),angle=angle)
    # Occluded presentation is not counted as co-experience. Body continues its
    # own actual control; no forced turn or relabelling repairs the trial.
    session.packet=env.stop_tone()
    return dict(pair=index,color=color,frequency=frequency,visible=False,
                retries=len(attempts),failures=attempts)


def record_catalog(brain,env,info,catalog):
    timestamp=str(info['index_time'])
    visible=env.visible_beacon_rays()
    colors=[color for color,rays in visible.items() if rays]
    tone=env._tone
    frequency=(tone['frequency_hz'] if tone is not None and info['sound_rms']>.02 else None)
    catalog[timestamp]=dict(colors=colors,frequency=frequency,
        visual=np.flatnonzero(brain.runtime.visual_memory.激活).tolist(),
        audio=np.flatnonzero(brain.runtime.audio_memory.激活).tolist(),
        motor=np.flatnonzero(brain.runtime.motor_memory.激活).tolist(),
        physical_time=float(env.world.time))


def content_score(info,catalog,color):
    result={}
    for stage,key,cells in (('initial','auditory_recall_time','initially_recalled_visual_cells'),
                            ('final','final_recall_time','final_visual_cells')):
        stored=catalog.get(str(info[key]))
        exact=bool(stored and stored['visual'] and
            set(stored['visual'])==set(map(int,info[cells])))
        result[stage]=dict(time=info[key],colors=[] if stored is None else stored['colors'],
            exact_visual=exact,correct=bool(exact and color in stored['colors']),
            cells=plain(info[cells]))
    return result


def weight_statistics(table):
    values=np.fromiter((w for row in table.表.values() for w in row.values()),float)
    if not len(values): return dict(count=0)
    mean=float(values.mean()); std=float(values.std())
    counts,bins=np.histogram(values,bins=24)
    return dict(count=len(values),mean=mean,std=std,
        skewness=float(np.mean(((values-mean)/std)**3)) if std else 0.,
        quantile_levels=[0,.1,.5,.9,.99,1.],
        quantiles=np.quantile(values,[0,.1,.5,.9,.99,1.]).tolist(),
        histogram_counts=counts.tolist(),histogram_edges=bins.tolist(),
        includes_only_existing_connections=True,
        ceiling=getattr(table,'ceiling',None),depression=getattr(table,'depression','multiplicative'))


def assess(session,catalog,anchors,stage,probe_steps=8,baseline_times=()):
    brain=session.brain
    snap=brain.snapshot()
    direct=[]
    references=[(frequency,stamp) for frequency,stamp in anchors.items()]
    references.extend((None,int(stamp)) for stamp in baseline_times)
    for frequency,stamp in references:
        reference=catalog[str(stamp)]
        recalled=brain.runtime.recall_at_time(stamp,brain.config.memory_threshold)
        expected=set(reference['visual']); actual=set(np.flatnonzero(recalled['visual']))
        modal={}
        for name in ('visual','audio','motor'):
            original=set(reference[name]); now=set(np.flatnonzero(recalled[name]))
            modal[name]=dict(stored_count=len(original),recalled_count=len(now),
                exact_match=now==original,exact_nonempty=bool(original and now==original),
                minimum_expected_weight=float(min((recalled[name+'_current'][i] for i in original),default=0.)))
        direct.append(dict(frequency=None if frequency is None else float(frequency),time=stamp,
            from_starting_life=frequency is None,
            reference_source=reference.get('reference_source','actual_new_sensation'),modalities=modal,
            stored_count=len(expected),
            recalled_count=len(actual),exact_nonempty=bool(expected and actual==expected),
            jaccard=len(actual&expected)/max(1,len(actual|expected)),
            minimum_expected_weight=float(min((recalled['visual_current'][i] for i in expected),default=0.))))
    cues=[]
    for color,frequency in PAIRS:
        clone=OriginalBrain.from_snapshot(snap)
        clone.pending=clone.prepared=None
        clone.flags.update(exploration=False,reflex=False)
        env=copy.deepcopy(session.environment)
        env.hide_beacons()
        packet=env.emit_tone(frequency,duration=.6)
        before=frozen_digest(clone)
        rows=[]
        for frame in range(probe_steps):
            muscles,info=clone.observe_then_act(packet,learning=False)
            rows.append(dict(frame=frame,rms=info['sound_rms'],
                score=content_score(info,catalog,color),muscles=muscles.tolist(),
                thought_cells=plain(info['thought_cells'])))
            packet=env.step(muscles)
            clone.observe_outcome(packet,muscles,terminal=frame==probe_steps-1)
        assert frozen_digest(clone)==before
        cues.append(dict(color=color,frequency=frequency,previously_experienced=str(frequency) in anchors,
                         trace=rows))
    pfc=brain.runtime.pfc
    currents=pfc.驱动(brain.thought)
    net=pfc.门场(brain.thought)
    # Read-only original internal dynamics from the same full state, with both
    # outside and replay input removed. This is not a learned-logic score.
    endogenous=OriginalBrain.from_snapshot(snap)
    endogenous_rows=[]
    for _ in range(3):
        detail=endogenous._microstep(np.zeros(brain.width),trace=False)
        endogenous_rows.append(detail['active_count'])
    rates=brain.runtime.connection_counts()
    older=[row for row in direct if row['from_starting_life']]
    old_scores={name:dict(correct=sum(row['modalities'][name]['exact_nonempty'] for row in older),
        total=sum(row['modalities'][name]['stored_count']>0 for row in older)) for name in ('visual','audio','motor')}
    return dict(stage=stage,clock=int(brain.runtime.clock.当前时间),
        old_saved_content_retention=old_scores,
        cue_score_scope='Same-event visual-content diagnostic, not a navigation/action-correctness criterion; valid learned shortcuts may change intermediate content.',
        memory_anchors=direct,cues=cues,
        pfc_active=int(brain.thought.sum()),pfc_inhibitory_strength=float(pfc.强度),
        pfc_excitatory_current_max=float(currents.max(initial=0.)),
        pfc_net_inhibition_max=float(net.max(initial=0.)),
        endogenous_no_input_active_counts=endogenous_rows,
        pfc_weight_statistics={name:weight_statistics(getattr(pfc,field))
                               for name,field in (('excitatory','联想'),('disinhibitory','去抑'))},
        connections=rates,clock_active_count=int(brain.runtime.clock.时间激活.sum()),
        direct_anchor_correct=sum(row['exact_nonempty'] for row in direct),
        direct_anchor_total=len(direct),
        known_cue_initial_correct=sum(r['trace'][0]['score']['initial']['correct'] for r in cues if r['previously_experienced']),
        known_cue_first_final_correct=sum(r['trace'][0]['score']['final']['correct'] for r in cues if r['previously_experienced']),
        known_cue_count=sum(r['previously_experienced'] for r in cues))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-life',type=Path,default=HERE/'results'/'living_original.npz')
    parser.add_argument('--parameters',type=Path)
    parser.add_argument('--previous-report',type=Path)
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--novel',action='store_true',help='Add six further actual tones to test new experience')
    parser.add_argument('--familiarize-steps',type=int,default=240)
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--check-every',type=int,default=300)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new output report')
    global PAIRS
    if args.extended:
        PAIRS=PAIRS+(('red',307.),('green',777.),('blue',2047.),
                     ('red',2503.),('green',2999.),('blue',3331.))
    if args.novel:
        PAIRS=PAIRS+(('red',1127.),('green',1513.),('blue',1877.),
                     ('red',2253.),('green',2751.),('blue',3751.))
    session=LifeSession.load(args.base_life)
    modifications=json.loads(args.parameters.read_text(encoding='utf8')) if args.parameters else {}
    apply_parameters(session.brain,modifications)
    start_clock=int(session.brain.runtime.clock.当前时间)
    start_actions=session.control_steps
    source={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ('brain.py','original_runtime.py','sensory_adapter.py','experience_environment.py','continuous_learning.py',
             'weight_dependent_plasticity.py','fast_hippocampal_index.py','fast_pfc_currents.py','contact_reflex.py')}
    catalog,anchors=initial_catalog(session),{}
    if args.previous_report:
        previous=json.loads(args.previous_report.read_text(encoding='utf8'))
        if previous['status']!='complete': raise ValueError('Previous run is incomplete')
        if int(previous['checks'][-1]['clock'])!=start_clock:
            raise ValueError('Previous report and complete life clock differ')
        if session.control_steps!=previous['starts']['control_steps']+previous['current_steps']:
            raise ValueError('Previous report and actual action counts differ')
        if previous.get('complete_life_sha256') and previous['complete_life_sha256']!=hashlib.sha256(args.base_life.read_bytes()).hexdigest():
            raise ValueError('Previous report and life content differ')
        catalog.update(copy.deepcopy(previous['catalog']))
        anchors.update(previous['anchors'])
    baseline_times=tuple(catalog)
    corpus={name:[] for name in ('rgb','distance','body','pcm','muscles','time','index_time')}
    records,timings,stimuli,checks=[],[],[],[]
    started=time.perf_counter()
    report=dict(status='running',base_life=str(args.base_life),
        base_sha256=hashlib.sha256(args.base_life.read_bytes()).hexdigest(),
        parameters=modifications,brain_config=asdict(session.brain.config),source_sha256=source,
        no_brain_or_body_reset=True,no_teacher_actions=True,reward=0,
        schedule=f'One ordinary visible circle+PCM every12 physical steps; expand after step{args.familiarize_steps}; varied bearings/distances;4 silent steps per presentation.',
        pairs=PAIRS,shuffled_order=bool(args.shuffle),schedule_rng_seed=560031,
        previous_report=None if args.previous_report is None else str(args.previous_report),
        knowledge_scope='Physical visual/audio/motor experiences; no text/document interface or language knowledge is claimed.',
        starts=dict(clock=start_clock,control_steps=start_actions,sensory_frames=session.brain.runtime.frames_recorded,
                    position=session.environment.world.position.tolist()),
        checks=checks,stimulus_events=stimuli,records=records)
    event=0
    order_rng=np.random.default_rng(560031)
    order=[]
    stimulus=present(session,0,event); stimuli.append(stimulus)
    for frame in range(args.steps):
        begin=time.perf_counter()
        position_before=session.environment.world.position.copy()
        packet=session.packet
        muscles,decision=session.brain.observe_then_act(packet,learning=True)
        session._catalog(decision)
        if str(decision['index_time']) not in catalog:
            record_catalog(session.brain,session.environment,decision,catalog)
        observation=packet['observation']
        corpus['rgb'].append(np.asarray(observation['ray_colors'],np.float32))
        corpus['distance'].append(np.asarray(observation['ray_distances'],np.float32))
        corpus['body'].append(np.r_[observation['velocity'],observation['angular_velocity'],
                                    observation['pain'],observation['touch']].astype(np.float32))
        corpus['pcm'].append(np.asarray(packet['waveform'],np.float32))
        corpus['muscles'].append(muscles.copy())
        corpus['time'].append(session.environment.world.time)
        corpus['index_time'].append(decision['index_time'])
        packet=session.environment.step(muscles)
        if (frame+1)%12==0:
            event+=1
            available=(6 if args.extended else 3) if frame+1<args.familiarize_steps else len(PAIRS)
            if args.shuffle:
                if not order or frame+1==args.familiarize_steps:
                    order=list(map(int,order_rng.permutation(available)))
                index=order.pop(0)
            else:
                index=event%available
            stimulus=present(session,index,event); stimuli.append(stimulus)
            packet=session.packet
        info=session.brain.observe_outcome(packet,muscles)
        session.packet=packet
        session.executed=muscles.copy(); session.info=info; session.decision=decision
        session._catalog(info)
        session.control_steps+=1
        session.trail.append(session.environment.world.position.copy())
        record_catalog(session.brain,session.environment,info,catalog)
        for stamp in (decision['index_time'],info['index_time']):
            sample=catalog[str(stamp)]
            if sample['frequency'] is not None and sample['colors'] and sample['visual']:
                anchors.setdefault(str(sample['frequency']),stamp)
        elapsed=(time.perf_counter()-begin)*1000
        timings.append(elapsed)
        records.append(dict(step=frame+1,time=session.environment.world.time,
            clock=int(session.brain.runtime.clock.当前时间),pfc_active=int(session.brain.thought.sum()),
            pfc_strength=float(session.brain.runtime.pfc.强度),
            pfc_edges=session.brain.runtime.pfc.联想.条数(),source=decision['source'],
            visible=bool(catalog[str(info['index_time'])]['colors']),ms=elapsed,
            position=session.environment.world.position.tolist(),heading=float(session.environment.world.heading),
            path_increment=float(np.linalg.norm(session.environment.world.position-position_before)),
            speed=float(np.linalg.norm(session.environment.world.velocity)),
            touch=plain(session.packet['observation']['touch'])))
        if (frame+1)%60==0:
            print(f"LIFE {frame+1}/{args.steps} active={records[-1]['pfc_active']} "
                  f"E={records[-1]['pfc_edges']} I={records[-1]['pfc_strength']:.2f} "
                  f"p95={np.percentile(timings[-60:],95):.1f}ms",flush=True)
        if (frame+1)%args.check_every==0 or frame==args.steps-1:
            checks.append(assess(session,catalog,anchors,frame+1,baseline_times=baseline_times))
            last=checks[-1]
            print('CHECK '+json.dumps({k:last[k] for k in ('stage','direct_anchor_correct',
                'direct_anchor_total','known_cue_initial_correct','known_cue_first_final_correct',
                'known_cue_count','endogenous_no_input_active_counts')}),flush=True)
            report.update(elapsed_seconds=time.perf_counter()-started,catalog=catalog,anchors=anchors,
                current_steps=frame+1,physical_control_p95_ms=float(np.percentile(timings,95)))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf8')
            session.save(args.output.with_suffix('.npz'))
    # A new first stimulus normally occupies one additional observation. If it
    # exactly equals a prepared observation, reuse is legitimate and adds none.
    frame_delta=session.brain.runtime.frames_recorded-report['starts']['sensory_frames']
    assert frame_delta in (args.steps,args.steps+1)
    assert int(session.brain.runtime.clock.当前时间)==start_clock+2*frame_delta
    assert int(session.brain.runtime.clock.时间激活.sum())==1
    assert session.control_steps-start_actions==args.steps
    assert source=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in source}
    corpus_path=args.output.with_name(args.output.stem+'_actual_sensors.npz')
    np.savez_compressed(corpus_path,**{name:np.asarray(values) for name,values in corpus.items()})
    report.update(status='complete',elapsed_seconds=time.perf_counter()-started,
        physical_seconds=args.steps*.1,actual_sensor_corpus=str(corpus_path),
        complete_life_checkpoint=str(args.output.with_suffix('.npz')),
        complete_life_sha256=hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest(),
        movement=dict(path_length=sum(row['path_increment'] for row in records),
            net_displacement=float(np.linalg.norm(session.environment.world.position-np.asarray(report['starts']['position']))),
            moving_steps=sum(row['path_increment']>1e-6 for row in records),
            contact_steps=sum(max(row['touch'])>.02 for row in records),
            sources=dict(Counter(row['source'] for row in records))),
        limit='Sensory association and continual-learning stability; no evidence of text knowledge, abstract logic or AGI.')
    args.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf8')
    print('COMPLETE '+str(args.output),flush=True)


if __name__=='__main__':
    main()
