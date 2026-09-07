"""Independent real-life original-index/fast-index equivalence and timing.

Load two independent copies of one complete physical life; replace only the
reference copy's index implementation with the unchanged original class. The
real sensory, action, outcome, shared-time learning and save paths are retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from fast_hippocampal_index import FastHippocampalIndex, OriginalHippocampalIndex
from viewer import LifeSession
from brain import packet_digest


HERE=Path(__file__).resolve().parent


def same_tree(a,b,path='life'):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:same_tree(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same_tree(x,y,path+'/'+str(i))
    else:
        assert a==b,(path,a,b)


def run(checkpoint,steps):
    started=time.perf_counter()
    protected=[checkpoint,HERE/'brain.py',HERE/'original_runtime.py',HERE/'viewer.py',
               HERE/'fast_hippocampal_index.py']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    t=time.perf_counter()
    reference=LifeSession.load(checkpoint)
    candidate=LifeSession.load(checkpoint)
    load_seconds=time.perf_counter()-t
    old=OriginalHippocampalIndex()
    old.时间到输出=reference.brain.runtime.index.时间到输出
    reference.brain.runtime.index=old
    assert type(reference.brain.runtime.index) is OriginalHippocampalIndex
    assert isinstance(candidate.brain.runtime.index,FastHippocampalIndex)
    assert reference.brain.runtime.index.时间到输出 is not candidate.brain.runtime.index.时间到输出
    for timestamp in old.时间到输出:
        assert old.时间到输出[timestamp] is not candidate.brain.runtime.index.时间到输出[timestamp]
    assert not np.shares_memory(reference.brain.thought,candidate.brain.thought)
    assert not np.shares_memory(reference.environment.world.position,candidate.environment.world.position)
    same_tree(reference.snapshot(),candidate.snapshot())
    initial_steps=reference.control_steps
    initial_clock=int(reference.brain.runtime.clock.当前时间)
    initial_edges=old.连接数()
    timings={'original':[],'fast':[]}
    traces=[]
    for i in range(steps):
        order=(('original',reference),('fast',candidate))
        if i%2:order=order[::-1]
        for name,session in order:
            t=time.perf_counter_ns();session.tick()
            timings[name].append((time.perf_counter_ns()-t)/1e6)
        # Comparison includes all canonical learned tables, live neural states,
        # RNG, pending/prepared action, clock, world, stimuli and actual packet.
        same_tree(reference.snapshot(),candidate.snapshot())
        assert np.array_equal(reference.executed,candidate.executed)
        assert packet_digest(reference.packet)==packet_digest(candidate.packet)
        traces.append(dict(step=i+1,physical_time=float(reference.environment.world.time),
            clock=int(reference.brain.runtime.clock.当前时间),muscles=reference.executed.tolist(),
            position=reference.environment.world.position.tolist(),
            source=reference.decision.get('source'),
            recalled_time=reference.info.get('final_recall_time'),
            index_edges=old.连接数()))
    with tempfile.TemporaryDirectory(prefix='exact-index-life-') as directory:
        path=Path(directory)/'complete_life.npz'
        candidate.save(path)
        with np.load(path,allow_pickle=False) as archive:
            manifest=json.loads(archive['manifest'].tobytes().decode('utf8'))
            assert set(manifest['brain']['runtime']['index'])=={'times','offsets','features'}
        restored=LifeSession.load(path)
        assert isinstance(restored.brain.runtime.index,FastHippocampalIndex)
        same_tree(reference.snapshot(),restored.snapshot())
        reference.tick();restored.tick()
        same_tree(reference.snapshot(),restored.snapshot())
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before==after,'Protected source/checkpoint changed concurrently'
    distributions={name:dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
        total_ms=float(sum(values))) for name,values in timings.items()}
    return dict(passed=True,checkpoint=str(checkpoint),independent_full_life_copies=True,
        original_vs_fast_real_steps=steps,full_canonical_life_comparison_after_every_step=True,
        continued_step_after_safe_save_load=True,cache_absent_from_saved_index=True,
        teacher_actions=0,rewards=0,initial_control_steps=initial_steps,initial_clock=initial_clock,
        initial_index_edges=initial_edges,final_index_edges=old.连接数(),
        complete_step_timings=distributions,
        complete_step_total_speedup=distributions['original']['total_ms']/distributions['fast']['total_ms'],
        initial_two_life_load_seconds=load_seconds,seconds=time.perf_counter()-started,
        frame_trace=traces,protected_hashes=before,
        limits=['Timing includes real act, physics, outcome and online learning; excludes verification and archive IO.',
                'One 1200-step life continued for a short fixed window, without stimulus rescheduling; not a long-run throughput guarantee.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results'/'continuous_balanced_1200.npz')
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--output',type=Path,default=HERE/'results'/'fast_index_closed_loop_audit.json')
    args=parser.parse_args()
    if not 1 <= args.steps <= 30:parser.error('--steps must be 1..30')
    report=run(args.checkpoint,args.steps)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('frame_trace','protected_hashes','limits')},ensure_ascii=True))


if __name__=='__main__':
    main()
