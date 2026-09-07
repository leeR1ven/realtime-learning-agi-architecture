"""One bounded, disclosed plasticity candidate on an existing physical life.

No production model/source/checkpoint is modified. Guards stop execution after
a real outcome; they never suppress neurons, remove edges, or rewrite actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import continuous_learning as course
from viewer import LifeSession
from weight_dependent_plasticity import _table_digest


HERE=Path(__file__).resolve().parent
CEILING=.15
ETA=.005
RATE=.9
INTERVAL=25
MAX_STEPS=120
ACTIVE_LIMIT=800
EDGE_LIMIT=1500000
PAIRS=(('red',220.),('green',440.),('blue',660.),('red',990.),('green',1320.),('blue',1750.),
       ('red',307.),('green',777.),('blue',2047.),('red',2503.),('green',2999.),('blue',3331.),
       ('red',1127.),('green',1513.),('blue',1877.),('red',2253.),('green',2751.),('blue',3751.))


def theory():
    q=1.-ETA/CEILING
    rows=[]
    for repetitions in (1,5,12,25):
        gamma=RATE*q**repetitions
        peak=CEILING*(1.-q**repetitions)/(1.-gamma)
        numeric=0.
        for _ in range(10000):
            numeric*=RATE
            for _ in range(repetitions):numeric+=ETA*(1.-numeric/CEILING)
        assert math.isclose(numeric,peak,rel_tol=1e-12)
        e=24*peak
        inhibition=max(0.,1.-.2*e)
        rows.append(dict(reinforcements_per_decay_cycle=repetitions,
            gamma_cycle_residual=gamma,equilibrium_peak_weight=peak,
            equilibrium_after_decay_weight=RATE*peak,
            N24_peak_E=e,N24_target_net_I_assuming_strength1_and_no_local_feedback=inhibition,
            N24_target_threshold=2.+inhibition,N24_margin=e-2.-inhibition,
            N24_can_cross_under_stated_assumptions=e>=2.+inhibition))
    return dict(ceiling=CEILING,eta=ETA,decay_rate=RATE,decay_interval=INTERVAL,
        potentiation_fraction_eta_over_C=ETA/CEILING,residual_per_reinforcement_q=q,
        equivalent_continuous_decay_lambda=-math.log(RATE)/INTERVAL,
        cycle_definition='One decay followed by m positive reinforcements; gamma = rate * q**m.',
        equilibrium_formula='C*(1-q**m)/(1-rate*q**m)',cycles=rows,
        N24_absolute_saturated_E=24*CEILING,N24_absolute_zero_inhibition_threshold=2.,
        N24_optimistic_rest_target_threshold=2.+max(0.,1.-.2*24*CEILING),
        limitations='Independent small-cluster bound only: assumes all24 relevant excitation and disinhibition edges exist with the stated weight, global strength1 and no target-site feedback. Real connectivity, phase, inhibition and usage frequency can prevent propagation. Legacy weights aboveC remain aboveC until forgotten.')


def atomic_json(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    os.replace(temporary,path)


def memory_references(session,previous_report):
    known=list(session.brain.runtime.index.时间到输出)
    selected=[int(known[i]) for i in np.linspace(0,len(known)-1,12,dtype=int)]
    if previous_report.exists():
        data=json.loads(previous_report.read_text(encoding='utf8'))
        selected=list(dict.fromkeys(list(map(int,data.get('anchors',{}).values()))+selected))[:24]
    references={}
    for stamp in selected:
        recalled=session.brain.runtime.recall_at_time(stamp,session.brain.config.memory_threshold)
        references[stamp]={name:np.flatnonzero(recalled[name]).tolist() for name in ('visual','audio','motor')}
    return references


def memory_retention(session,references):
    rows=[]
    for stamp,reference in references.items():
        recalled=session.brain.runtime.recall_at_time(stamp,session.brain.config.memory_threshold)
        rows.append(dict(time=stamp,modalities={name:dict(baseline_cells=len(reference[name]),
            current_cells=int(recalled[name].sum()),
            exact_match=np.flatnonzero(recalled[name]).tolist()==reference[name])
            for name in ('visual','audio','motor')}))
    return dict(scope='Preservation of sampled content already readable in the starting checkpoint; not proof that all original sensations had previously survived.',
        rows=rows,summary={name:dict(correct=sum(r['modalities'][name]['exact_match'] for r in rows if r['modalities'][name]['baseline_cells']),
            total=sum(bool(r['modalities'][name]['baseline_cells']) for r in rows)) for name in ('visual','audio','motor')})


def raw_dynamics(decision,outcome):
    steps=[]
    for stage,info in (('decision',decision),('outcome',outcome)):
        for i,row in enumerate(info.get('microsteps',[])):
            item=dict(stage=stage,microstep=i,active_count=int(row['active_count']),
                inhibition_strength_before=float(row['inhibition_strength_before']),
                inhibition_strength_after=float(row['inhibition_strength_after']))
            for field in ('excitatory_current','inhibitory_site_current','net_inhibitory_site_current',
                          'disinhibitory_site_current','external_current'):
                item[field+'_max']=float(np.asarray(row.get(field,[])).max(initial=0.))
            steps.append(item)
    return steps


def run(base,output):
    if output.exists() or output.with_suffix('.npz').exists():
        raise FileExistsError('Use a new output; existing experiment is retained')
    protected=[base,HERE/'brain.py',HERE/'original_runtime.py',HERE/'contact_reflex.py',
               HERE/'continuous_learning.py',HERE/'weight_dependent_plasticity.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    session=LifeSession.load(base)
    pfc=session.brain.runtime.pfc
    before={name:_table_digest(getattr(pfc,name)) for name in ('联想','去抑')}
    changes=dict(pfc_hebb_increment=ETA,pfc_weight_ceiling=CEILING,
        pfc_weight_depression='multiplicative',pfc_decay_rate=RATE,pfc_decay_interval=INTERVAL,pfc_decay_warning=1)
    course.apply_parameters(session.brain,changes)
    assert before=={name:_table_digest(getattr(pfc,name)) for name in ('联想','去抑')}
    course.PAIRS=PAIRS
    references=memory_references(session,base.with_suffix('.json'))
    rng=np.random.default_rng(560031)
    order=list(map(int,rng.permutation(len(PAIRS))))
    start_position=session.environment.world.position.copy()
    records=[]
    report=dict(status='running',candidate=changes,theory=theory(),base_life=str(base),
        protected_hashes=hashes,brain_config=asdict(session.brain.config),
        initial_weight_conversion_unchanged=True,no_brain_body_or_time_reset=True,
        teacher_actions=0,reward=0,neural_clamping=False,
        resource_guard=dict(check='immediately after each actual outcome',thought_count_greater_than=ACTIVE_LIMIT,
                            E_edges_greater_than=EDGE_LIMIT,action='save completed real state and stop before another control step'),
        maximum_actual_steps=MAX_STEPS,
        starts=dict(clock=int(session.brain.runtime.clock.当前时间),control_steps=session.control_steps,
                    position=start_position.tolist(),pfc_active=int(session.brain.thought.sum()),E_edges=pfc.联想.条数()),
        schedule=dict(pairs=PAIRS,order_seed=560031,order=order,presentation_frames=12,
                      sounding_frames=8,silent_frames=4),stimuli=[],records=records,
        old_memory_sample_before=memory_retention(session,references))
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,report)
    started=time.perf_counter()
    if report['starts']['pfc_active']>ACTIVE_LIMIT or report['starts']['E_edges']>EDGE_LIMIT:
        report['status']='stopped_initial_resource_guard'
        report['stopped_before_any_new_action']=True
    else:
        report['stimuli'].append(course.present(session,order[0],0))
        for frame in range(MAX_STEPS):
            begin=time.perf_counter()
            position_before=session.environment.world.position.copy()
            muscles,decision=session.brain.observe_then_act(session.packet,learning=True,trace=True)
            session._catalog(decision)
            packet=session.environment.step(muscles)
            if (frame+1)%12==0 and frame+1<MAX_STEPS:
                phase=(frame+1)//12
                stimulus=course.present(session,order[phase],phase)
                report['stimuli'].append(stimulus)
                packet=session.packet
            outcome=session.brain.observe_outcome(packet,muscles,trace=True)
            # Guard readings are the first inspection after the actual outcome.
            active=int(session.brain.thought.sum())
            edges=pfc.联想.条数()
            guard=active>ACTIVE_LIMIT or edges>EDGE_LIMIT
            session.packet=packet
            session.executed=muscles.copy();session.info=outcome;session.decision=decision
            session._catalog(outcome);session.control_steps+=1
            session.trail.append(session.environment.world.position.copy())
            session.last_frame_ms=(time.perf_counter()-begin)*1000
            dynamics=raw_dynamics(decision,outcome)
            records.append(dict(step=frame+1,clock=int(session.brain.runtime.clock.当前时间),
                physical_time=float(session.environment.world.time),thought_count=active,E_edges=edges,
                DI_edges=pfc.去抑.条数(),I_strength=float(pfc.强度),source=decision['source'],
                muscles=muscles.tolist(),position=session.environment.world.position.tolist(),
                heading=float(session.environment.world.heading),
                path_increment=float(np.linalg.norm(session.environment.world.position-position_before)),
                speed=float(np.linalg.norm(session.environment.world.velocity)),
                pain=float(packet['observation']['pain']),touch=np.asarray(packet['observation']['touch']).tolist(),
                visible_beacon_rays=session.environment.visible_beacon_rays(),
                raw_dynamics=dynamics,control_ms=session.last_frame_ms,guard_triggered=guard))
            if guard:
                report['status']='stopped_resource_guard'
                report['stop_reason']=dict(step=frame+1,thought_count=active,E_edges=edges)
                # Save the completed real state immediately; no next action,
                # recall probe, or extra neural learning occurs before saving.
                report['current_steps']=len(records)
                session.save(output.with_suffix('.npz'))
                atomic_json(output,report)
                print(json.dumps(dict(status=report['status'],**report['stop_reason'])),flush=True)
                break
            if (frame+1)%12==0:
                report['current_steps']=len(records)
                atomic_json(output,report)
                print(f"PRETRIAL {frame+1}/{MAX_STEPS} active={active} Eedges={edges} I={pfc.强度:.3f}",flush=True)
        else:
            report['status']='completed_bounded_pretrial'
    if not output.with_suffix('.npz').exists():session.save(output.with_suffix('.npz'))
    report['current_steps']=len(records)
    report['old_memory_sample_after']=memory_retention(session,references)
    dynamics=[item for row in records for item in row['raw_dynamics']]
    report['summary']=dict(guard_triggered=report['status'].startswith('stopped'),
        final_thought_max=max((r['thought_count'] for r in records),default=report['starts']['pfc_active']),
        all_recorded_microstep_active_max=max((r['active_count'] for r in dynamics),default=0),
        E_edges_max=max((r['E_edges'] for r in records),default=report['starts']['E_edges']),
        raw_I_current_max=max((r['inhibitory_site_current_max'] for r in dynamics),default=0),
        net_I_current_max=max((r['net_inhibitory_site_current_max'] for r in dynamics),default=0),
        raw_E_current_max=max((r['excitatory_current_max'] for r in dynamics),default=0),
        global_I_strength_max=max((r['I_strength'] for r in records),default=float(pfc.强度)),
        actual_path_length=sum(r['path_increment'] for r in records),
        net_displacement=float(np.linalg.norm(session.environment.world.position-start_position)),
        action_sources=dict(Counter(r['source'] for r in records)),
        contact_steps=sum(any(r['touch']) for r in records),
        p95_control_ms=float(np.percentile([r['control_ms'] for r in records],95)) if records else None)
    report['runtime_seconds']=time.perf_counter()-started
    report['complete_life_checkpoint']=str(output.with_suffix('.npz'))
    report['complete_life_sha256']=hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()
    report['protected_inputs_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in protected)
    assert report['protected_inputs_unchanged']
    report['interpretation']='One short resource-bounded candidate, without an additive control arm here. Passing this window is not long-run stability, independent small-cluster cognition, generalization, navigation or AGI validation.'
    atomic_json(output,report)
    print(json.dumps(dict(status=report['status'],current_steps=report['current_steps'],
        summary=report['summary'],retention=report['old_memory_sample_after']['summary']),ensure_ascii=True))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-life',type=Path,default=HERE/'results'/'continuous_balanced_extended_3600.npz')
    parser.add_argument('--output',type=Path,default=HERE/'results'/'pretrial_soft_cap015_120.json')
    parser.add_argument('--theory-only',action='store_true')
    args=parser.parse_args()
    if args.theory_only:print(json.dumps(theory(),ensure_ascii=True));return
    run(args.base_life,args.output)


if __name__=='__main__':
    main()
