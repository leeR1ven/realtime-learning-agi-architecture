"""Exact, cache-free accumulation for the original recurrent PFC equations.

This is an execution optimization, not a changed neural model. Source cells
retain np.nonzero order; each dictionary retains insertion order. np.add.at
performs unbuffered repeated-index additions. No weights or cues are discarded,
no sums are grouped by target, and no connectivity/content cache is maintained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from 前额叶区_prefrontal import 前额叶联想区 as OriginalPFC, 前额叶神经网络


def _ordered_current(table,active,width):
    output=np.zeros(width)
    rows=[table.查(int(source)) for source in active]
    count=sum(map(len,rows))
    if count:
        targets=np.fromiter(itertools.chain.from_iterable(row.keys() for row in rows),
                            dtype=np.intp,count=count)
        weights=np.fromiter(itertools.chain.from_iterable(row.values() for row in rows),
                            dtype=np.float64,count=count)
        np.add.at(output,targets,weights)
    return output


class FastPFC(OriginalPFC):
    """Same constructor/state/API; only current and gate accumulation differ."""

    def 驱动(self,上念头):
        return _ordered_current(self.联想,np.nonzero(上念头)[0],self.宽度)

    def 门场(self,上念头):
        sites=(self.宽度+self.每段-1)//self.每段
        active=np.nonzero(上念头)[0]
        feedback=np.bincount(active//self.每段,minlength=sites).astype(float)
        inhibition=self.强度*(self.静息抑制+self.反馈抑制*feedback)
        disinhibition=_ordered_current(self.去抑,active,sites)
        return np.maximum(0.0,inhibition-self.去抑抵消*disinhibition)


def _same_float(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label


def _unpack(value,archive):
    if isinstance(value,dict):
        if set(value)=={'__array__'}:return archive[value['__array__']].copy()
        return {key:_unpack(item,archive) for key,item in value.items()}
    if isinstance(value,list):return [_unpack(item,archive) for item in value]
    return value


def _restore_table(table,saved):
    for name,value in saved['parameters'].items():setattr(table,name,value)
    for name,value in saved['counters'].items():setattr(table,name,value)
    table.表={}
    for row,source in enumerate(saved['sources']):
        a,b=map(int,saved['offsets'][row:row+2])
        table.表[int(source)]={int(t):float(w) for t,w in zip(saved['targets'][a:b],saved['weights'][a:b])}


def _clone_as_fast(original):
    fast=FastPFC.__new__(FastPFC)
    fast.__dict__=copy.deepcopy(original.__dict__)
    return fast


def _equal_tables(original,fast):
    for name in ('联想','去抑'):
        a,b=getattr(original,name),getattr(fast,name)
        assert set(vars(a))==set(vars(b))
        assert list(a.表)==list(b.表)
        for source,row in a.表.items():
            assert list(row)==list(b.表[source])
            _same_float(list(row.values()),list(b.表[source].values()),'table/'+name)
        for key in vars(a):
            if key!='表':assert getattr(a,key)==getattr(b,key)


def _compare(original,fast,cue,external=None):
    _same_float(original.驱动(cue),fast.驱动(cue),'excitatory_current')
    _same_float(original.门场(cue),fast.门场(cue),'inhibitory_gate')
    a=original.微步(cue,外部信号=external)
    b=fast.微步(cue,外部信号=external)
    _same_float(a,b,'microstep_activity')
    _same_float(np.float64(original.强度),np.float64(fast.强度),'dynamic_inhibitory_strength')


def _ordering_fixture():
    original=OriginalPFC(80,目标上限=79,强度上限=1000.)
    weights=(float(2**53),1.,1.,1e-200,3.,1e16,2.,.25)
    # Vary dictionary target order; repeated destinations across source rows
    # expose any regrouping or pairwise sum of large and tiny positive weights.
    for source,weight in enumerate(weights):
        targets=(41,3,70) if source%2 else (70,41,3)
        original.联想.表[source]={target:weight*(1.+target/100.) for target in targets}
        original.去抑.表[source]={site:weight*(1.+site/10.) for site in (1,0)}
    original.联想.连接条数=24;original.去抑.连接条数=16
    fast=_clone_as_fast(original)
    cue=np.zeros(80,bool);cue[:8]=True
    _compare(original,fast,cue)
    for name in ('联想','去抑'):
        a,b=getattr(original,name),getattr(fast,name)
        saved_a,saved_b=a.表,b.表
        a.表={};b.表={}
        _compare(original,fast,cue,np.zeros(80))
        a.表=saved_a;b.表=saved_b
    original.联想.表[0][3]=.123456789012345
    fast.联想.表[0][3]=.123456789012345
    original.去抑.表[1][0]=.314159265358979
    fast.去抑.表[1][0]=.314159265358979
    _compare(original,fast,cue)
    _compare(original,fast,np.zeros(80,bool))
    # Duplicate first-axis indices retain the original nonzero semantics too.
    _compare(original,fast,np.array([[1,1],[1,0]],int))
    assert FastPFC.微步 is OriginalPFC.微步
    assert FastPFC.学习 is OriginalPFC.学习
    assert FastPFC.__init__ is OriginalPFC.__init__
    assert set(vars(original))==set(vars(fast))
    return dict(large_tiny_positive_weight_sum_order_bitwise_equal=True,
                direct_dictionary_mutations_immediately_visible=True,
                temporary_excitation_and_disinhibition_lesions_exact=True,
                empty_and_multidimensional_cues_match=True,
                original_learning_microstep_and_constructor_inherited=True,no_new_instance_state=True)


def audit(checkpoint,sensors):
    from sensory_adapter import SensoryAdapter
    started=time.perf_counter()
    protected=[checkpoint,sensors,ROOT/'前额叶区_prefrontal.py',HERE/'brain.py',HERE/'original_runtime.py',
               HERE/'weight_dependent_plasticity.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    with np.load(checkpoint,allow_pickle=False) as archive:
        manifest=json.loads(archive['manifest'].tobytes().decode('utf8'))
        brain=manifest['brain'];config=brain['config']
        params=brain['runtime']['parameters']
        pfc_saved=_unpack(brain['runtime']['pfc'],archive)
        adapter=SensoryAdapter.from_snapshot(_unpack(brain['adapter'],archive))
        born=_unpack(brain['forward'],archive)
    width=sum(adapter.widths.values())
    original=OriginalPFC(width,**params['pfc_parameters'])
    original.念头=pfc_saved['activity'];original.强度=pfc_saved['inhibition_strength']
    _restore_table(original.联想,pfc_saved['excitatory'])
    _restore_table(original.去抑,pfc_saved['disinhibitory'])
    fast=_clone_as_fast(original)
    initial_I=float(original.强度)
    rng_state=np.random.get_state()
    try:
        np.random.seed(config['seed']+10001)
        forward=前额叶神经网络(width,隐藏层阈值=config['pfc_hidden_threshold'],
            输出层阈值=config['pfc_output_threshold'],扩散宽度=config['pfc_spread'])
    finally:np.random.set_state(rng_state)
    forward.来源=born['sources'];forward.权重=born['weights'];forward.阈值=born['thresholds']
    cues=[]
    with np.load(sensors,allow_pickle=False) as archive:
        for row in np.linspace(0,len(archive['time'])-1,64,dtype=int):
            body=archive['body'][row]
            packet={'observation':dict(ray_colors=archive['rgb'][row].astype(float),
                ray_distances=archive['distance'][row].astype(float),
                velocity=body[:2].astype(float),angular_velocity=float(body[2]),pain=float(body[3]),
                touch=body[4:].astype(float)),'waveform':archive['pcm'][row].astype(float)}
            encoded=adapter.encode(packet,archive['muscles'][row])
            v,a,m=[encoded[name] for name in ('visual','audio','motor')]
            full=forward.前向传播(v,a,m).astype(bool).copy()
            audio=forward.前向传播(np.zeros_like(v),a,np.zeros_like(m)).astype(bool).copy()
            cues.extend([('actual_recorded_multimodal',full),('actual_recorded_audio',audio)])
    rng=np.random.default_rng(604211)
    for i in range(120):
        n=(1,24,31,64,120,250,350,800,1500)[i%9]
        cue=np.zeros(width,bool);cue[rng.choice(width,n,replace=False)]=True
        cues.append(('random',cue))
    cues.extend([('empty',np.zeros(width,bool)),('dense',np.ones(width,bool)),
                 ('saved_actual_thought',original.念头.copy())])
    _ordering_fixture()
    microstep_rows=[]
    for i,(category,cue) in enumerate(cues):
        external=None if i%3==0 else (3.5*cue if i%3==1 else rng.uniform(0.,12.,width))
        _compare(original,fast,cue,external)
        microstep_rows.append(dict(category=category,cue_size=int(cue.sum()),
            actual_active_count=int(original.念头.sum()),inhibition_strength=float(original.强度)))
    _equal_tables(original,fast)
    # A continuing internal activity sequence, without resets, must evolve
    # bitwise equally in activity and dynamic inhibition at every microstep.
    for step in range(24):
        cue=original.念头.copy()
        _compare(original,fast,cue,2.5*cues[(step*7)%len(cues)][1])
    # Reuse the original learning method and then verify current reads see the
    # changed original dictionaries immediately, with no cache rebuild hook.
    for i in range(12):
        before=np.zeros(width,bool);after=np.zeros(width,bool)
        before[rng.choice(width,24,replace=False)]=True
        after[rng.choice(width,31,replace=False)]=True
        original.学习(before,after);fast.学习(before,after)
        _compare(original,fast,before)
    _equal_tables(original,fast)
    for name in ('联想','去抑'):
        a,b=getattr(original,name),getattr(fast,name)
        saved_a,saved_b=a.表,b.表
        a.表={};b.表={}
        _compare(original,fast,cues[0][1])
        a.表=saved_a;b.表=saved_b
    first_source=next(iter(original.联想.表))
    first_target=next(iter(original.联想.表[first_source]))
    original.联想.表[first_source][first_target]+=.123456789
    fast.联想.表[first_source][first_target]+=.123456789
    _compare(original,fast,np.ones(width,bool))

    # Interleave paired timings; no comparisons/hash/learning in timed regions.
    original.强度=fast.强度=initial_I
    timings={name:{operation:[] for operation in ('drive','gate','microstep')} for name in ('original','fast')}
    for repeat in range(3):
        for i,(_,cue) in enumerate(cues):
            order=(('original',original),('fast',fast))
            if (i+repeat)%2:order=order[::-1]
            for name,pfc in order:
                for operation,function in (('drive',pfc.驱动),('gate',pfc.门场),('microstep',pfc.微步)):
                    t=time.perf_counter_ns();function(cue)
                    timings[name][operation].append((time.perf_counter_ns()-t)/1e6)
            _same_float(np.float64(original.强度),np.float64(fast.强度),'timed_microstep_I')
            _same_float(original.念头,fast.念头,'timed_microstep_activity')
    stats={name:{operation:dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
        total_ms=float(sum(values))) for operation,values in rows.items()} for name,rows in timings.items()}
    speedups={operation:stats['original'][operation]['total_ms']/stats['fast'][operation]['total_ms']
              for operation in ('drive','gate','microstep')}
    category_counts={category:sum(category==name for name,_ in cues) for category in sorted({c for c,_ in cues})}
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert hashes==after,'Protected source or checkpoint changed concurrently'
    return dict(passed=True,checkpoint=str(checkpoint),cue_cases=len(cues),
        distinct_cue_bit_patterns=len({(len(q),q.tobytes()) for _,q in cues}),categories=category_counts,
        exact_current_gate_microstep_cases=len(cues),continuing_microsteps=24,original_learning_updates=12,
        timed_additional_exact_microstep_cases=3*len(cues),fixtures=_ordering_fixture(),
        original_pfc_E_edges=int(pfc_saved['excitatory']['counters']['连接条数']),
        original_pfc_DI_edges=int(pfc_saved['disinhibitory']['counters']['连接条数']),
        benchmark_after_identical_disclosed_local_updates=True,
        current_gate_and_activity_bitwise_equal=True,dynamic_inhibition_bitwise_equal=True,
        stats=stats,total_time_speedup=speedups,
        persistent_cache_bytes=0,maximum_dense_excitation_temporary_payload_bytes=
            original.联想.条数()*(np.dtype(np.intp).itemsize+8),
        original_source_and_dict_order_preserved=True,protected_hashes=hashes,
        microstep_diagnostics=microstep_rows,seconds=time.perf_counter()-started,
        limitations=['Exactness is empirically bitwise on the current NumPy/CPU and tested valid original table data.',
            'Temporary target/weight arrays are rebuilt every call; no stale cache or new persistent model state exists.',
            'Only recurrent current/gate execution is optimized; original learning, decay, thresholds, inhibition updates and saved model remain unchanged.',
            'This is a component benchmark, not an end-to-end 10Hz or long-run throughput guarantee.',
            'Recorded sensory probes retain the float32 precision of the actual sensor archive.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results'/'active_body_additive_600.npz')
    parser.add_argument('--sensors',type=Path,default=HERE/'results'/'active_body_additive_600_actual_sensors.npz')
    parser.add_argument('--output',type=Path,default=HERE/'results'/'fast_pfc_currents_audit.json')
    args=parser.parse_args()
    report=audit(args.checkpoint,args.sensors)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('protected_hashes','microstep_diagnostics','limitations')},ensure_ascii=True))


if __name__=='__main__':main()
