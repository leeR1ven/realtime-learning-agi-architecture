"""Exact implementation optimization of the original hippocampal index.

The original 时间到输出 dictionary of sets remains the sole authoritative
memory. All other fields are disposable, reconstructible search caches. They
must not be serialized as a second memory. No weights, thresholds, candidates,
features, tie rules, or chronological links are changed.

Use 学习 for updates. After intentionally editing the public original mapping
directly (including changing a set in place), call rebuild_from_original().
This explicit cache invalidation is necessary for arbitrary external writes.
"""
from __future__ import annotations

from array import array
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from 前额叶区_prefrontal import 海马索引 as OriginalHippocampalIndex, 前额叶神经网络


class FastHippocampalIndex(OriginalHippocampalIndex):
    """Same original API, with exact inverted integer posting lists."""

    def __init__(self):
        super().__init__()
        self.rebuild_from_original()

    @classmethod
    def from_time_mapping(cls, mapping, *, copy_mapping=False):
        obj = cls()
        obj.时间到输出 = ({t:set(cells) for t,cells in mapping.items()}
                         if copy_mapping else mapping)
        return obj.rebuild_from_original()

    def rebuild_from_original(self, original=None):
        if original is not None:
            self.时间到输出 = original.时间到输出
        if array('i').itemsize != 4:
            raise RuntimeError("This implementation requires 32-bit C int posting arrays")
        self._times = list(self.时间到输出)
        if len(self._times) >= 2**31:
            raise OverflowError("Posting row identifiers exceed int32 capacity")
        self._row_by_time = {t:i for i,t in enumerate(self._times)}
        self._sizes = array('i')
        self._postings = {}
        self._edge_count = 0
        for row, timestamp in enumerate(self._times):
            cells = self.时间到输出[timestamp]
            self._sizes.append(len(cells))
            self._edge_count += len(cells)
            for cell in cells:
                self._postings.setdefault(cell,array('i')).append(row)
        return self

    def 学习(self, 时间, 输出模式):
        # np.nonzero(...)[0] deliberately matches the original, including
        # duplicate first-axis indices should callers pass a multidimensional
        # pattern. A zero pattern must not insert even an empty time record.
        active = np.nonzero(输出模式)[0]
        if not len(active):
            return
        timestamp = int(时间)
        if timestamp not in self.时间到输出:
            if len(self._times) >= 2**31:
                raise OverflowError("Posting row identifiers exceed int32 capacity")
            self.时间到输出[timestamp] = set()
            self._row_by_time[timestamp] = len(self._times)
            self._times.append(timestamp)
            self._sizes.append(0)
        row = self._row_by_time[timestamp]
        stored = self.时间到输出[timestamp]
        for value in active:
            cell = int(value)
            if cell not in stored:
                stored.add(cell)
                self._postings.setdefault(cell,array('i')).append(row)
                self._sizes[row] += 1
                self._edge_count += 1

    def 连接数(self):
        return self._edge_count

    def 锁定时间(self, 输出模式, 相对门槛=0.7):
        cue = np.unique(np.nonzero(输出模式)[0])
        if not len(cue):
            return None, 0.0
        parts = []
        for cell in cue:
            posting = self._postings.get(int(cell))
            if posting:
                parts.append(np.frombuffer(posting,dtype=np.int32))
        if not parts:
            return None, 0.0
        # Every cue feature has one vote for each stored set containing it.
        # Row ordinals preserve dictionary insertion order, not numeric time.
        votes = np.bincount(np.concatenate(parts),minlength=len(self._times))
        best_votes = int(votes.max())
        tied_rows = np.flatnonzero(votes == best_votes)
        sizes = np.frombuffer(self._sizes,dtype=np.int32)
        gaps = np.abs(sizes[tied_rows].astype(np.int64)-len(cue))
        winner = int(tied_rows[int(np.argmin(gaps))])
        ratio = best_votes / len(cue)
        return (self._times[winner] if ratio >= 相对门槛 else None), ratio

    def cache_memory(self):
        """Allocated cache containers, excluding the retained original memory.

        Key objects already referenced by the original mapping are not charged
        again; newly allocated row integers are counted. Query temporaries are
        reported separately, because they are released after each lookup.
        """
        total = sys.getsizeof(self._times)+sys.getsizeof(self._row_by_time)
        total += sys.getsizeof(self._sizes)+sys.getsizeof(self._postings)
        total += sum(sys.getsizeof(v) for v in self._postings.values())
        total += sum(sys.getsizeof(v) for v in self._row_by_time.values())
        return dict(allocated_cache_bytes_estimate=total,
                    posting_payload_bytes=sum(len(v)*v.itemsize for v in self._postings.values()),
                    time_rows=len(self._times),feature_postings=len(self._postings),edges=self._edge_count)


def _unpack(value, archive):
    if isinstance(value,dict):
        if set(value)=={'__array__'}:
            return archive[value['__array__']].copy()
        return {k:_unpack(v,archive) for k,v in value.items()}
    if isinstance(value,list):
        return [_unpack(v,archive) for v in value]
    return value


def _pattern(cells,width):
    result=np.zeros(width,bool)
    result[list(cells)]=True
    return result


def _assert_equal(original,fast,cue,threshold):
    expected=original.锁定时间(cue,相对门槛=threshold)
    result=fast.锁定时间(cue,相对门槛=threshold)
    assert expected==result,(expected,result,threshold,np.flatnonzero(cue).tolist())
    return result


def _fixtures():
    original=OriginalHippocampalIndex()
    original.时间到输出={90:{1,2,3},7:{1,2},3:{1,2},101:set()}
    fast=FastHippocampalIndex().rebuild_from_original(original)
    assert fast.时间到输出 is original.时间到输出
    cue=_pattern([1,2],16)
    assert _assert_equal(original,fast,cue,.7)==(7,1.)
    # Query denominator includes unknown cells. Equal vote+gap chooses the
    # first inserted record even when another timestamp is numerically lower.
    assert _assert_equal(original,fast,_pattern([1,2,14],16),.8)==(None,2/3)
    assert _assert_equal(original,fast,_pattern([1,2,14],16),2/3)==(90,2/3)
    queries=[np.zeros(16,bool),_pattern([15],16),cue,np.ones(16,bool),
             np.array([[0,1],[1,1]],int),np.array([0.,np.nan,1.])]
    count=0
    for q in queries:
        for threshold in (-1.,0.,.7,1.,1.01,float('nan')):
            _assert_equal(original,fast,q,threshold);count+=1
    # Independently update two copies through the ordinary API, including
    # repeatedly revisiting times, empty learning, and growing existing sets.
    reference=OriginalHippocampalIndex()
    reference.时间到输出=copy.deepcopy(original.时间到输出)
    fast=FastHippocampalIndex.from_time_mapping(original.时间到输出,copy_mapping=True)
    for timestamp,active in ((700,[]),(7,[1,2]),(7,[4,5]),(2,[14]),(90,[1,2,3]),(7,[6])):
        q=_pattern(active,16)
        reference.学习(timestamp,q);fast.学习(timestamp,q)
        assert reference.时间到输出==fast.时间到输出
        assert list(reference.时间到输出)==list(fast.时间到输出)
        assert reference.连接数()==fast.连接数()
        for probe in queries:
            _assert_equal(reference,fast,probe,.7);count+=1
    assert 700 not in fast.时间到输出
    # External original-class learning is deliberately followed by cache
    # rebuild; there is no hidden second copy of the canonical memory.
    reference.学习(7,_pattern([8,9],16))
    fast.rebuild_from_original(reference)
    assert fast.时间到输出 is reference.时间到输出
    for q in queries:
        _assert_equal(reference,fast,q,.7);count+=1
    return count


def audit(checkpoint_path,sensors_path):
    from sensory_adapter import SensoryAdapter
    started=time.perf_counter()
    protected=[checkpoint_path,sensors_path,ROOT/'前额叶区_prefrontal.py',HERE/'brain.py',HERE/'original_runtime.py']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    with np.load(checkpoint_path,allow_pickle=False) as archive:
        manifest=json.loads(archive['manifest'].tobytes().decode('utf8'))
        brain=manifest['brain']
        saved=_unpack(brain['runtime']['index'],archive)
        adapter=SensoryAdapter.from_snapshot(_unpack(brain['adapter'],archive))
        born=_unpack(brain['forward'],archive)
        config=brain['config']
    original=OriginalHippocampalIndex()
    for row,timestamp in enumerate(saved['times']):
        a,b=map(int,saved['offsets'][row:row+2])
        original.时间到输出[int(timestamp)]=set(map(int,saved['features'][a:b]))
    construct_started=time.perf_counter()
    fast=FastHippocampalIndex().rebuild_from_original(original)
    rebuild_seconds=time.perf_counter()-construct_started
    assert fast.时间到输出 is original.时间到输出
    width=sum(adapter.widths.values())
    state=np.random.get_state()
    try:
        np.random.seed(config['seed']+10001)
        forward=前额叶神经网络(width,隐藏层阈值=config['pfc_hidden_threshold'],
                            输出层阈值=config['pfc_output_threshold'],扩散宽度=config['pfc_spread'])
    finally:
        np.random.set_state(state)
    forward.来源=born['sources'];forward.权重=born['weights'];forward.阈值=born['thresholds']
    rng=np.random.default_rng(260909)
    queries=[]
    with np.load(sensors_path,allow_pickle=False) as archive:
        for row in np.linspace(0,len(archive['time'])-1,120,dtype=int):
            body=archive['body'][row]
            packet={'observation':dict(ray_colors=archive['rgb'][row].astype(float),
                ray_distances=archive['distance'][row].astype(float),
                velocity=body[:2].astype(float),angular_velocity=float(body[2]),pain=float(body[3]),
                touch=body[4:].astype(float)),'waveform':archive['pcm'][row].astype(float)}
            encoded=adapter.encode(packet,archive['muscles'][row])
            v,a,m=[encoded[k] for k in ('visual','audio','motor')]
            sound=forward.前向传播(np.zeros_like(v),a,np.zeros_like(m)).astype(bool).copy()
            full=forward.前向传播(v,a,m).astype(bool).copy()
            queries.extend([('actual_recorded_audio',sound),('actual_recorded_multimodal',full)])
    timestamps=list(original.时间到输出)
    for _ in range(90):
        remembered=sorted(original.时间到输出[timestamps[int(rng.integers(len(timestamps)))]] )
        keep=int(rng.integers(0,len(remembered)+1))
        subset=rng.choice(remembered,keep,replace=False).tolist() if keep else []
        extra=rng.choice(width+64,int(rng.integers(0,45)),replace=False).tolist()
        queries.append(('partial_stored_plus_noise',_pattern(set(subset+extra),width+64)))
    for _ in range(90):
        n=int(rng.integers(0,600))
        queries.append(('random',_pattern(rng.choice(width+64,n,replace=False),width+64)))
    queries.extend([('empty',np.zeros(width,bool)),('unknown',_pattern([width+2],width+64)),
                    ('dense',np.ones(width+64,bool))])
    thresholds=(-.1,0.,.7,.8,.9,1.,1.01)
    comparisons=_fixtures()
    results=[]
    for category,q in queries:
        result=None
        for threshold in thresholds:
            result=_assert_equal(original,fast,q,threshold);comparisons+=1
        results.append(dict(category=category,cue_size=int(np.unique(np.nonzero(q)[0]).size),
                            time_at_threshold_08=fast.锁定时间(q,.8)[0],ratio=result[1]))
    assert fast.连接数()==original.连接数()

    # Same-time ring revisits, fresh timestamps, repeated/empty updates and
    # ordinary ongoing lookups are checked against separate original objects.
    update_reference=OriginalHippocampalIndex()
    update_reference.时间到输出=copy.deepcopy(original.时间到输出)
    update_fast=FastHippocampalIndex.from_time_mapping(original.时间到输出,copy_mapping=True)
    updates=[]
    for i in range(160):
        timestamp=timestamps[i%len(timestamps)] if i%3 else max(timestamps)+1+i
        q=queries[int(rng.integers(len(queries)))][1]
        updates.append((timestamp,q))
        update_reference.学习(timestamp,q);update_fast.学习(timestamp,q)
        if i%5==0:
            probe=queries[int(rng.integers(len(queries)))][1]
            for threshold in (.0,.7,.8,1.):
                _assert_equal(update_reference,update_fast,probe,threshold);comparisons+=1
    assert update_reference.时间到输出==update_fast.时间到输出
    assert list(update_reference.时间到输出)==list(update_fast.时间到输出)
    assert update_reference.连接数()==update_fast.连接数()
    update_fast.rebuild_from_original(update_reference)
    for _,q in queries:
        _assert_equal(update_reference,update_fast,q,.8);comparisons+=1

    # Per-query timing is interleaved to reduce order/thermal bias. Equality
    # checks stay outside the timed region; benchmark includes dense queries.
    timings={'original':[],'fast':[]}
    for repeat in range(3):
        for i,(_,q) in enumerate(queries):
            order=(('original',original),('fast',fast))
            if (i+repeat)%2:order=order[::-1]
            for name,index in order:
                t=time.perf_counter_ns();index.锁定时间(q,.8)
                timings[name].append((time.perf_counter_ns()-t)/1e6)
    learning_times={}
    combined_times={}
    for name in ('original','fast'):
        index=OriginalHippocampalIndex() if name=='original' else FastHippocampalIndex()
        index.时间到输出=copy.deepcopy(original.时间到输出)
        if name=='fast':index.rebuild_from_original()
        t=time.perf_counter()
        for timestamp,q in updates:index.学习(timestamp,q)
        learning_times[name]=(time.perf_counter()-t)*1000
        index=OriginalHippocampalIndex() if name=='original' else FastHippocampalIndex()
        index.时间到输出=copy.deepcopy(original.时间到输出)
        if name=='fast':index.rebuild_from_original()
        t=time.perf_counter()
        for i,(timestamp,q) in enumerate(updates):
            index.学习(timestamp,q)
            for offset in (0,1,2):index.锁定时间(queries[(i*3+offset)%len(queries)][1],.8)
        combined_times[name]=(time.perf_counter()-t)*1000
    distributions={name:dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                            total_ms=float(np.sum(values))) for name,values in timings.items()}
    categories={category:dict(count=sum(row['category']==category for row in results),
        nonempty=sum(row['category']==category and row['cue_size']>0 for row in results),
        mean_cue_size=float(np.mean([row['cue_size'] for row in results if row['category']==category])))
        for category in sorted({row['category'] for row in results})}
    original_bytes=sys.getsizeof(original.时间到输出)
    original_bytes+=sum(sys.getsizeof(t)+sys.getsizeof(cells)+sum(sys.getsizeof(c) for c in cells)
                        for t,cells in original.时间到输出.items())
    max_votes_payload=max(sum(len(fast._postings.get(int(cell),())) for cell in np.unique(np.nonzero(q)[0]))
                          for _,q in queries)*4
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before==after,'Protected input changed concurrently'
    return dict(passed=True,exact_comparisons=comparisons,cue_cases=len(queries),
        distinct_cue_bit_patterns=len({(len(q),q.tobytes()) for _,q in queries}),
        thresholds=list(thresholds),learning_update_calls=len(updates),
        existing_time_update_calls=sum(t in original.时间到输出 for t,_ in updates),
        new_time_update_calls=sum(t not in original.时间到输出 for t,_ in updates),
        actual_added_time_rows=len(update_reference.时间到输出)-len(timestamps),
        original_time_rows=len(timestamps),
        original_index_edges=original.连接数(),original_mapping_shared_not_copied=True,
        categories=categories,query_results=results,lookup=distributions,
        lookup_total_speedup=distributions['original']['total_ms']/distributions['fast']['total_ms'],
        rebuild_ms=rebuild_seconds*1000,learning_160_updates_ms=learning_times,
        mixed_160_learning_plus_480_query_ms=combined_times,
        mixed_speedup=combined_times['original']/combined_times['fast'],
        cache_memory=fast.cache_memory(),original_mapping_memory_bytes_estimate=original_bytes,
        largest_query_posting_payload_bytes=max_votes_payload,
        source_checkpoint_hashes=before,seconds=time.perf_counter()-started,
        limitations=[
            'External direct dict/set mutation requires explicit rebuild_from_original; normal learning updates caches exactly.',
            'Cache memory is additional implementation overhead and is not a second saved knowledge store.',
            'Recorded sensor arrays use their saved float32 precision; probes are re-encoded from that actual recorded data.',
            'Microbenchmark measures this real 1200-step index, not end-to-end frame speed or an unbounded future index.',
            'Query concatenation temporarily allocates one int32 row entry per matched feature-time posting.'
        ])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=HERE/'results'/'continuous_balanced_1200.npz')
    parser.add_argument('--sensors',type=Path,default=HERE/'results'/'continuous_balanced_1200_actual_sensors.npz')
    parser.add_argument('--output',type=Path,default=HERE/'results'/'fast_hippocampal_index_audit.json')
    args=parser.parse_args()
    report=audit(args.checkpoint,args.sensors)
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    brief={k:v for k,v in report.items() if k not in ('query_results','source_checkpoint_hashes','limitations','categories')}
    print(json.dumps(brief,ensure_ascii=True))


if __name__=='__main__':
    main()
