"""Real-life equivalence/timing isolating only the PFC current optimization.

Both independent lives retain the exact fast hippocampal index. No learning,
plasticity parameter, physics, memory, stimulus or saved-state format differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from audit_fast_index_closed_loop import same_tree
from fast_hippocampal_index import FastHippocampalIndex
from fast_pfc_currents import FastPFC,OriginalPFC
from viewer import LifeSession
from brain import packet_digest


HERE=Path(__file__).resolve().parent


def run(checkpoint,steps):
    started=time.perf_counter()
    protected=[checkpoint,HERE/'brain.py',HERE/'original_runtime.py',HERE/'viewer.py',
               HERE/'fast_pfc_currents.py',HERE/'contact_reflex.py']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    t=time.perf_counter()
    original=LifeSession.load(checkpoint)
    fast=LifeSession.load(checkpoint)
    load_seconds=time.perf_counter()-t
    assert type(original.brain.runtime.pfc) is FastPFC
    assert type(fast.brain.runtime.pfc) is FastPFC
    original.brain.runtime.pfc.__class__=OriginalPFC
    assert type(original.brain.runtime.pfc) is OriginalPFC
    for life in (original,fast):
        assert isinstance(life.brain.runtime.index,FastHippocampalIndex)
    assert original.brain.runtime.pfc is not fast.brain.runtime.pfc
    for name in ('联想','去抑'):
        a,b=getattr(original.brain.runtime.pfc,name),getattr(fast.brain.runtime.pfc,name)
        assert a.表 is not b.表
        for source,row in a.表.items():assert row is not b.表[source]
    for timestamp,row in original.brain.runtime.index.时间到输出.items():
        assert row is not fast.brain.runtime.index.时间到输出[timestamp]
    assert not np.shares_memory(original.brain.thought,fast.brain.thought)
    assert not np.shares_memory(original.environment.world.position,fast.environment.world.position)
    same_tree(original.snapshot(),fast.snapshot())
    start_steps=original.control_steps
    start_clock=int(original.brain.runtime.clock.当前时间)
    timings={'original_pfc':[],'fast_pfc':[]}
    records=[]
    for step in range(steps):
        order=(('original_pfc',original),('fast_pfc',fast))
        if step%2:order=order[::-1]
        for name,life in order:
            t=time.perf_counter_ns();life.tick()
            timings[name].append((time.perf_counter_ns()-t)/1e6)
        same_tree(original.snapshot(),fast.snapshot())
        assert packet_digest(original.packet)==packet_digest(fast.packet)
        records.append(dict(step=step+1,clock=int(original.brain.runtime.clock.当前时间),
            physical_time=float(original.environment.world.time),muscles=original.executed.tolist(),
            position=original.environment.world.position.tolist(),
            pfc_active=int(original.brain.thought.sum()),
            E_edges=original.brain.runtime.pfc.联想.条数(),
            I_strength=float(original.brain.runtime.pfc.强度),
            source=original.decision.get('source'),
            recalled_time=original.info.get('final_recall_time')))
    with tempfile.TemporaryDirectory(prefix='exact-pfc-life-') as directory:
        path=Path(directory)/'complete_life.npz'
        fast.save(path)
        restored=LifeSession.load(path)
        assert type(restored.brain.runtime.pfc) is FastPFC
        assert isinstance(restored.brain.runtime.index,FastHippocampalIndex)
        same_tree(original.snapshot(),restored.snapshot())
        original.tick();restored.tick()
        same_tree(original.snapshot(),restored.snapshot())
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    stats={name:dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
        max_ms=float(max(values)),total_ms=float(sum(values)),
        steps_at_or_above_100ms=sum(value>=100 for value in values)) for name,values in timings.items()}
    return dict(passed=True,checkpoint=str(checkpoint),isolated_change='OriginalPFC vs FastPFC current/gate only',
        both_indices_fast=True,independent_full_lives=True,actual_control_steps_per_arm=steps,
        full_canonical_state_equal_after_every_step=True,
        full_state_equal_after_safe_save_restore_and_one_further_step=True,
        no_parameter_learning_or_physics_rule_difference=True,teacher_actions=0,rewards=0,
        initial_steps=start_steps,initial_clock=start_clock,complete_control_timing=stats,
        total_control_speedup=stats['original_pfc']['total_ms']/stats['fast_pfc']['total_ms'],
        fast_p95_below_100ms=stats['fast_pfc']['p95_ms']<100,
        fast_max_below_100ms=stats['fast_pfc']['max_ms']<100,
        two_life_load_seconds=load_seconds,seconds=time.perf_counter()-started,
        records=records,protected_hashes=before,
        limits=['Timing includes act, real physics, outcome, and online learning; excludes verification and checkpoint IO.',
                'This is a fixed short continuation of the saved life, without rescheduling stimuli. It is evidence for this window, not a perpetual 10Hz guarantee.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results'/'active_body_additive_600.npz')
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--output',type=Path,default=HERE/'results'/'fast_pfc_closed_loop_audit.json')
    args=parser.parse_args()
    if not 1<=args.steps<=30:parser.error('--steps must be 1..30')
    report=run(args.checkpoint,args.steps)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('records','protected_hashes','limits')},ensure_ascii=True))


if __name__=='__main__':main()
