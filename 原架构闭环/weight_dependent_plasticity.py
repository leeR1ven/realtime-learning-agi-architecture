"""Disclosed optional weight-dependent plasticity for original connection tables.

No birth connections, Gaussian distribution, noise, eligibility trace or new
per-edge state is introduced. The only additional state is ceiling/depression.
The original dictionary and its public API/counters/maintenance schedule remain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from 权重连接管理_weight_manager import 连接表 as OriginalConnectionTable


ORIGINAL_PARAMETERS=('强化量','衰减率','消失下限','警戒线','衰减间隔')
ORIGINAL_COUNTERS=('连接条数','维护次数','上次衰减维护')


class SoftBoundConnectionTable(OriginalConnectionTable):
    """Δw = eta*max(0,1-w/ceiling), with original or optional soft depression.

    Existing above-ceiling weights are retained unchanged by potentiation.
    The original instance must cease acting as an owner after from_original:
    the dictionary is shared, whereas the scalar maintenance counters are copied.
    """

    def __init__(self,*args,ceiling,depression='multiplicative',**kwargs):
        super().__init__(*args,**kwargs)
        self.ceiling=float(ceiling)
        self.depression=depression
        self._validate_configuration()

    def _validate_configuration(self):
        if not math.isfinite(self.ceiling) or self.ceiling<=0:
            raise ValueError('ceiling must be finite and positive')
        if self.depression not in ('multiplicative','weight_dependent'):
            raise ValueError('depression must be multiplicative or weight_dependent')
        if not math.isfinite(self.强化量) or not 0 < self.强化量 <= self.ceiling:
            raise ValueError('强化量 must satisfy 0 < eta <= ceiling')
        if not math.isfinite(self.衰减率) or not 0 < self.衰减率 <= 1:
            raise ValueError('衰减率 must be finite and in (0,1]')
        if not math.isfinite(self.消失下限) or self.消失下限 < 0:
            raise ValueError('消失下限 must be finite and nonnegative')

    @classmethod
    def from_original(cls,table,ceiling,depression='multiplicative'):
        result=cls(**{key:getattr(table,key) for key in ORIGINAL_PARAMETERS},
                   ceiling=ceiling,depression=depression)
        result.表=table.表
        for key in ORIGINAL_COUNTERS:
            setattr(result,key,getattr(table,key))
        return result

    def 学习(self,起点,终点,增量=None):
        eta=float(self.强化量 if 增量 is None else 增量)
        if not math.isfinite(self.ceiling) or not 0 < self.ceiling:
            raise ValueError('ceiling must be finite and positive')
        if not math.isfinite(eta) or not 0 < eta <= self.ceiling:
            raise ValueError('Learning increment must satisfy 0 < eta <= ceiling')
        source,target=int(起点),int(终点)
        row=self.表.setdefault(source,{})
        old=row.get(target,0.0)
        if old >= self.ceiling:
            new=old
        else:
            new=old+eta*max(0.,1.-old/self.ceiling)
            # The real-valued update is bounded for eta<=ceiling. This only
            # protects against a floating-point overshoot of that endpoint;
            # the above-ceiling branch never clips a legacy weight.
            new=min(new,self.ceiling)
        row[target]=new
        if old==0.0:
            self.连接条数+=1  # Preserve the original positive-edge convention.

    def _depressed_weight(self,weight):
        rate=self.衰减率
        if rate==1. or weight==0.:
            return weight
        # Algebraically w/[1+(1/r-1)*(w/C)]. These two equivalent branches
        # avoid overflowing w/C or 1/r for finite extreme positive inputs.
        if weight <= self.ceiling:
            return weight*(rate/(rate+(1.-rate)*(weight/self.ceiling)))
        return self.ceiling*(rate/(rate*(self.ceiling/weight)+(1.-rate)))

    def 整体衰减(self):
        if self.depression=='multiplicative':
            return super().整体衰减()
        self._validate_configuration()
        for row in self.表.values():
            for target in list(row):
                weight=self._depressed_weight(row[target])
                if weight < self.消失下限:
                    del row[target]
                    self.连接条数-=1
                else:
                    row[target]=weight
        for source in [source for source,row in self.表.items() if not row]:
            del self.表[source]


def _table_digest(table):
    h=hashlib.sha256()
    for source,row in table.表.items():
        h.update(str(source).encode());h.update(b':')
        for target,weight in row.items():
            h.update(str(target).encode());h.update(np.float64(weight).tobytes())
    return h.hexdigest()


def _unpack(value,archive):
    if isinstance(value,dict):
        if set(value)=={'__array__'}:return archive[value['__array__']].copy()
        return {key:_unpack(item,archive) for key,item in value.items()}
    if isinstance(value,list):return [_unpack(item,archive) for item in value]
    return value


def _restore_original(saved):
    table=OriginalConnectionTable(**saved['parameters'])
    for i,source in enumerate(saved['sources']):
        a,b=map(int,saved['offsets'][i:i+2])
        table.表[int(source)]={int(target):float(weight) for target,weight in
            zip(saved['targets'][a:b],saved['weights'][a:b])}
    for key,value in saved['counters'].items():setattr(table,key,value)
    return table


def _edge(table,value,source=0,target=0):
    table.表={source:{target:float(value)}}
    table.连接条数=1


def audit(checkpoint):
    started=time.perf_counter()
    protected=[checkpoint,ROOT/'权重连接管理_weight_manager.py',HERE/'brain.py',HERE/'original_runtime.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    failures_rejected=0
    for kwargs in (dict(ceiling=0),dict(ceiling=float('inf')),dict(ceiling=float('nan')),
                   dict(ceiling=1,强化量=0),dict(ceiling=1,强化量=2),dict(ceiling=1,depression='normal'),
                   dict(ceiling=1,衰减率=0),dict(ceiling=1,衰减率=1.1)):
        try:SoftBoundConnectionTable(**kwargs)
        except ValueError:failures_rejected+=1
        else:raise AssertionError(('invalid parameters accepted',kwargs))
    table=SoftBoundConnectionTable(ceiling=1.,强化量=.1)
    for eta in (0.,-1.,1.001,float('inf'),float('nan')):
        try:table.学习(1,2,eta)
        except ValueError:failures_rejected+=1
        else:raise AssertionError('invalid explicit eta accepted')
        assert not table.表 and table.连接条数==0

    amplification=[]
    for initial in (0.,.1,.5,.9,1.,1.5,10.):
        table=SoftBoundConnectionTable(ceiling=1.,强化量=.1)
        if initial:_edge(table,initial)
        table.学习(0,0)
        expected=initial+.1*max(0.,1.-initial)
        assert math.isclose(table.表[0][0],expected,rel_tol=1e-15,abs_tol=1e-15)
        amplification.append(dict(initial=initial,increment=table.表[0][0]-initial,after=table.表[0][0]))
    recurrence=[]
    for ceiling in (.001,1.,100.):
        for fraction in (.0001,.1,1.):
            eta=ceiling*fraction
            for initial_ratio in (0.,.4,1.,1.5):
                table=SoftBoundConnectionTable(ceiling=ceiling,强化量=eta)
                initial=ceiling*initial_ratio
                if initial:_edge(table,initial)
                previous=initial
                for step in range(10000):
                    table.学习(0,0)
                    current=table.表[0][0]
                    assert previous <= current <= max(initial,ceiling)
                    previous=current
                expected=(initial if initial>=ceiling else
                          ceiling-(ceiling-initial)*(1.-fraction)**10000)
                assert math.isclose(current,expected,rel_tol=2e-12,abs_tol=ceiling*2e-12)
                assert table.连接条数==1 and len(table)==1
                recurrence.append(dict(ceiling=ceiling,eta=eta,initial=initial,after_10000=current,analytic=expected))

    depression_rows=[]
    for rate in (.1,.5,.9,1.):
        table=SoftBoundConnectionTable(ceiling=2.,强化量=.1,衰减率=rate,depression='weight_dependent',消失下限=0.)
        retention=[]
        for weight in (0.,.02,.2,1.,2.,4.,200.):
            actual=table._depressed_weight(weight)
            expected=weight/(1.+(1./rate-1.)*(weight/2.))
            assert 0 <= actual <= weight
            assert math.isclose(actual,expected,rel_tol=2e-15,abs_tol=1e-15)
            if weight:retention.append(actual/weight)
            depression_rows.append(dict(rate=rate,before=weight,after=actual))
        assert all(a>=b for a,b in zip(retention,retention[1:]))
        assert table._depressed_weight(2.)==2.*rate
        if rate<1:
            _edge(table,4.)
            for _ in range(100):table.整体衰减()
            expected=4./(1.+100*(1./rate-1.)*(4./2.))
            assert math.isclose(table.表[0][0],expected,rel_tol=1e-13)

    extreme=[]
    for ceiling,weight,rate,expected in ((1e-300,1e300,.5,1e-300),
            (1.,1e-300,1e-300,5e-301),(1.,1e300,1e-300,1e-300),
            (1.,1e300,1.,1e300)):
        table=SoftBoundConnectionTable(ceiling=ceiling,强化量=ceiling,衰减率=rate,depression='weight_dependent')
        actual=table._depressed_weight(weight)
        assert math.isfinite(actual) and 0 <= actual <= weight
        assert math.isclose(actual,expected,rel_tol=1e-14,abs_tol=0.)
        extreme.append(dict(ceiling=ceiling,weight=weight,rate=rate,result=actual))

    # Unchanged multiplicative branch must match original values and counters
    # exactly under identical old weights and identical maintenance calls.
    original=OriginalConnectionTable(强化量=.1,衰减率=.5,消失下限=.04,警戒线=3,衰减间隔=4)
    for i in range(3):original.学习(i,i)
    soft=SoftBoundConnectionTable.from_original(copy.deepcopy(original),ceiling=1.)
    original_initial=dict(vars(original))
    flags=[]
    for step in range(25):
        if step==12:
            for i in range(3,6):original.学习(i,i);soft.学习(i,i)
        expected=original.维护();actual=soft.维护()
        assert expected==actual
        assert original.表==soft.表
        for key in ORIGINAL_COUNTERS:assert getattr(original,key)==getattr(soft,key)
        flags.append(actual)
    assert not soft.表 and soft.连接条数==0
    # Weight-dependent maintenance inherits that exact scheduling function.
    assert SoftBoundConnectionTable.维护 is OriginalConnectionTable.维护
    schedule_reference=OriginalConnectionTable(强化量=.1,衰减率=.5,消失下限=0.,警戒线=1,衰减间隔=4)
    schedule_reference.学习(0,0)
    schedule_soft=SoftBoundConnectionTable.from_original(copy.deepcopy(schedule_reference),ceiling=1.,depression='weight_dependent')
    for _ in range(25):
        assert schedule_reference.维护()==schedule_soft.维护()
        for key in ORIGINAL_COUNTERS:assert getattr(schedule_reference,key)==getattr(schedule_soft,key)
    for mode in ('multiplicative','weight_dependent'):
        floor=SoftBoundConnectionTable(ceiling=1.,强化量=.1,衰减率=.5,消失下限=.5,depression=mode)
        _edge(floor,1.)
        floor.整体衰减()
        assert floor.表[0][0]==.5 and floor.连接条数==1
        floor.整体衰减()
        assert not floor.表 and floor.连接条数==0
        assert floor.维护次数==0

    real_tables=[]
    with np.load(checkpoint,allow_pickle=False) as archive:
        manifest=json.loads(archive['manifest'].tobytes().decode('utf8'))
        runtime=manifest['brain']['runtime']
        selections={'pfc_excitatory':runtime['pfc']['excitatory'],
                    'visual_time_to_feature':runtime['memories']['visual']['time_to_feature']}
        for name,packed in selections.items():
            old=_restore_original(_unpack(packed,archive))
            weights=[w for row in old.表.values() for w in row.values()]
            ceiling=max(old.强化量,float(np.quantile(weights,.25)))
            old_digest=_table_digest(old)
            for mode in ('multiplicative','weight_dependent'):
                new=SoftBoundConnectionTable.from_original(old,ceiling,mode)
                assert new.表 is old.表 and _table_digest(new)==old_digest
                for source in old.表:assert new.表[source] is old.表[source]
                for key in ORIGINAL_PARAMETERS+ORIGINAL_COUNTERS:
                    assert getattr(new,key)==getattr(old,key)
                assert set(vars(new))-set(vars(old))=={'ceiling','depression'}
            real_tables.append(dict(name=name,edges=old.连接条数,conversion_weight_sha256=old_digest,
                audit_ceiling_only=ceiling,above_ceiling_existing_edges=sum(w>ceiling for w in weights),
                both_modes_share_original_weights_unchanged=True))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in protected)
    return dict(passed=True,parameters_rejected_before_mutation=failures_rejected,
        potentiation_formula='delta_w = eta * max(0, 1 - w/ceiling), 0 < eta <= ceiling',
        depression_formula='w_next = w / (1 + (1/decay_rate - 1) * w/ceiling)',
        amplification=amplification,repeated_training=recurrence,depression=depression_rows,
        extreme_positive_float_cases=extreme,multiplicative_values_and_counters_match_original=True,
        inherited_maintenance_schedule=True,maintenance_trigger_flags=flags,
        strict_less_than_floor_deletion_and_empty_rows=True,
        repeated_learning_one_edge_counts_once=True,legacy_above_ceiling_not_clipped=True,
        real_old_table_conversions=real_tables,no_per_edge_state=True,
        no_distribution_constraint_or_noise=True,protected_hashes=hashes,seconds=time.perf_counter()-started,
        limits=['This changes the learning rule, not merely execution speed; no behavioural success is implied.',
                'For normal positive edge weights the original counter convention is retained, including old==0 identifying a new connection.',
                'A bounded single-edge weight does not itself bound the number of edges or recurrent network stability.',
                'Weak-edge-dependent depression retains weak edges longer; edge count and actual retention need whole-life validation.',
                'Only rounding overshoots for weights previously at/below the ceiling are limited to that endpoint; legacy higher weights are unchanged by potentiation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results'/'continuous_balanced_1200.npz')
    parser.add_argument('--output',type=Path,default=HERE/'results'/'weight_dependent_plasticity_audit.json')
    args=parser.parse_args()
    report=audit(args.checkpoint)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({key:value for key,value in report.items() if key not in
        ('amplification','repeated_training','depression','protected_hashes','limits')},ensure_ascii=True))


if __name__=='__main__':
    main()
