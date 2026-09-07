"""Original-class equivalence and safe-state acceptance; no navigation training."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from original_runtime import OriginalRuntime, _original_modules, _table_snapshot


HERE = Path(__file__).resolve().parent


def equal_tree(left, right, path='root'):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            equal_tree(left[key], right[key], path+'/'+str(key))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, path+'/'+str(index))
    else:
        assert left == right, (path, left, right)


def edge_digest(runtime):
    digest = hashlib.sha256()
    for name, memory in runtime.memories.items():
        for direction in ('特征到时间', '时间到特征'):
            item = _table_snapshot(getattr(memory, direction))
            digest.update((name+direction).encode())
            for key in ('sources', 'offsets', 'targets', 'weights'):
                digest.update(item[key].tobytes())
            digest.update(json.dumps(item['parameters'], sort_keys=True).encode())
            digest.update(json.dumps(item['counters'], sort_keys=True).encode())
    for table in (runtime.pfc.联想, runtime.pfc.去抑):
        item = _table_snapshot(table)
        for key in ('sources', 'offsets', 'targets', 'weights'):
            digest.update(item[key].tobytes())
        digest.update(json.dumps(item['parameters'], sort_keys=True).encode())
        digest.update(json.dumps(item['counters'], sort_keys=True).encode())
    digest.update(json.dumps({key: sorted(value) for key, value in runtime.index.时间到输出.items()}).encode())
    return digest.hexdigest()


def safe_roundtrip(runtime):
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            key = 'array_'+str(len(arrays))
            assert not value.dtype.hasobject
            arrays[key] = value
            return {'__array__': key}
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    metadata = pack(runtime.snapshot())
    arrays['metadata'] = np.frombuffer(json.dumps(metadata).encode(), np.uint8)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    size = stream.tell()
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {'__array__'}:
                    return archive[value['__array__']].copy()
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item) for item in value]
            return value
        state = unpack(json.loads(archive['metadata'].tobytes()))
    return OriginalRuntime.from_snapshot(state), size


def random_frames(count=12):
    rng = np.random.default_rng(2026090617)
    frames = []
    for _ in range(count):
        parts = []
        for width, active in ((16, 3), (16, 3), (8, 2), (32, 3)):
            vector = np.zeros(width, bool)
            vector[rng.choice(width, active, replace=False)] = True
            parts.append(vector)
        frames.append(parts)
    return frames


def assert_matches_reference(runtime, reference):
    np.testing.assert_array_equal(runtime.clock.时间激活, reference['clock'].时间激活)
    assert runtime.clock.当前时间 == reference['clock'].当前时间
    for name, memory in runtime.memories.items():
        np.testing.assert_array_equal(memory.激活, reference[name].激活)
        for direction in ('特征到时间', '时间到特征'):
            equal_tree(_table_snapshot(getattr(memory, direction)), _table_snapshot(getattr(reference[name], direction)))
    assert runtime.index.时间到输出 == reference['index'].时间到输出
    for name in ('联想', '去抑'):
        equal_tree(_table_snapshot(getattr(runtime.pfc, name)), _table_snapshot(getattr(reference['pfc'], name)))
    assert runtime.pfc.强度 == reference['pfc'].强度
    np.testing.assert_array_equal(runtime.pfc.念头, reference['pfc'].念头)


def equivalence_checks():
    pfc_parameters = {'每段': 4, '门槛': .9, '目标下限': 1, '目标上限': 16}
    connections = {'强化量': .8, '警戒线': 10, '衰减间隔': 3, '衰减率': .9, '消失下限': .0005}
    memory_connections = {'警戒线': 20, '衰减间隔': 3, '衰减率': .9}
    runtime = OriginalRuntime(16, 16, 8, 32, time_neurons=64, pfc_parameters=pfc_parameters,
        pfc_connection_parameters=connections, memory_connection_parameters=memory_connections)
    modules = _original_modules()
    reference = {'clock': modules['海马体时间区'].时间环(64),
                 'visual': modules['视觉记忆区'].记忆区(16),
                 'audio': modules['听觉记忆区'].记忆区(16),
                 'motor': modules['运动记忆区'].记忆区(8),
                 'index': modules['前额叶区'].海马索引(),
                 'pfc': modules['前额叶区'].前额叶联想区(32, **pfc_parameters)}
    assert type(runtime.clock) is type(reference['clock'])
    assert isinstance(runtime.pfc, type(reference['pfc']))
    # The exact-summation subclass may override only current accumulation.
    # Preserve the original constructor, firing and plasticity implementations.
    for method in ('__init__', '微步', '学习'):
        assert getattr(type(runtime.pfc), method) is getattr(type(reference['pfc']), method)
    for name in ('visual', 'audio', 'motor'):
        assert type(runtime.memories[name]) is type(reference[name])
        for direction in ('特征到时间', '时间到特征'):
            for key, value in memory_connections.items():
                setattr(getattr(reference[name], direction), key, value)
    for table in (reference['pfc'].联想, reference['pfc'].去抑):
        for key, value in connections.items():
            setattr(table, key, value)
    frames = random_frames()
    previous = None
    frozen_checks = 0
    for index, frame in enumerate(frames):
        visual, audio, motor, pfc = frame
        learning = index not in (4, 5)
        before_digest = edge_digest(runtime)
        info = runtime.record(*frame, learning=learning)
        if learning and previous is not None:
            reference['pfc'].学习(previous, pfc)
        previous = pfc.copy()
        for microstep in range(reference['clock'].每帧步数):
            old, new = reference['clock'].走一步()
            if learning and microstep == 0:
                reference['index'].学习(old, pfc)
            for name, values in zip(('visual', 'audio', 'motor'), (visual, audio, motor)):
                if learning:
                    reference[name].学习(old, new, values)
                else:
                    reference[name].激活[:] = values
        if not learning:
            assert edge_digest(runtime) == before_digest
            assert not info['pfc_transition_learned']
            frozen_checks += 1
        assert_matches_reference(runtime, reference)
        external = pfc.astype(float)*2.
        expected_excitation = reference['pfc'].驱动(pfc)
        expected_gate = reference['pfc'].门场(pfc)
        expected_activity = reference['pfc'].微步(pfc, 外部信号=external)
        weights_before_thought = edge_digest(runtime)
        activity, detail = runtime.pfc_microstep(pfc, external)
        np.testing.assert_array_equal(activity, expected_activity)
        np.testing.assert_array_equal(detail['excitatory_current'], expected_excitation)
        np.testing.assert_array_equal(detail['net_inhibitory_site_current'], expected_gate)
        assert edge_digest(runtime) == weights_before_thought
        assert_matches_reference(runtime, reference)
    assert runtime.frames_recorded == len(frames) and runtime.internal_steps_recorded == 2*len(frames)
    assert all(memory.时间到特征.上次衰减维护 >= 0 for memory in runtime.memories.values())

    # Compare wrapper audio playback with the original class's native method.
    # Temporary binding is test-only and restored; production never rebinds it.
    old_global = modules['听觉记忆区'].时钟
    try:
        modules['听觉记忆区'].时钟 = runtime.clock
        for frame in frames:
            expected = runtime.audio_memory.回忆(frame[1], 步数=6, 阈值=1)
            actual = runtime.recall_from_audio(frame[1], steps=6, threshold=1)['playback']
            assert len(expected) == len(actual)
            for (timestamp, audio), item in zip(expected, actual):
                assert timestamp == item['time']
                np.testing.assert_array_equal(audio, item['audio'])
    finally:
        modules['听觉记忆区'].时钟 = old_global
    return runtime, {'frames_equal_to_direct_original_calls': len(frames),
        'frozen_frames_advance_clock_without_connections_or_maintenance': frozen_checks,
        'microsteps_equal_to_original_activity_excitation_gate': len(frames),
        'native_original_audio_playback_equal_queries': len(frames),
        'decay_maintenance_exercised': True,
        'runtime_objects_are_original_class_instances': True,
        'microstep_does_not_autowrite_connections': True}


def temporal_checks():
    runtime = OriginalRuntime(16, 16, 8, 32, time_neurons=32)
    frames = []
    for number in range(4):
        v, a, m, p = (np.zeros(width, bool) for width in (16, 16, 8, 32))
        v[2*number:2*number+2] = True
        a[2*number:2*number+2] = True
        m[number] = True
        p[2*number:2*number+2] = True
        frames.append((v, a, m, p))
        runtime.record(v, a, m, p)
    for number, (v, a, m, p) in enumerate(frames):
        recalled = runtime.recall_from_pfc(p)
        assert recalled['time'] == number*2 and recalled['ratio'] == 1.
        for name, expected in (('visual', v), ('audio', a), ('motor', m)):
            np.testing.assert_array_equal(recalled[name], expected)
    native = runtime.recall_from_audio(frames[0][1], steps=2)
    assert native['time'] == 2
    np.testing.assert_array_equal(native['playback'][0]['visual'], frames[1][0])
    current = runtime.clock.当前时间
    before = edge_digest(runtime)
    for _ in range(3):
        runtime.recall_from_pfc(frames[0][3])
        runtime.recall_from_audio(frames[0][1], steps=4)
        runtime.recall_at_time(0)
    assert runtime.clock.当前时间 == current and edge_digest(runtime) == before
    unknown = np.zeros(32, bool)
    unknown[31] = True
    assert runtime.recall_from_pfc(unknown)['time'] is None

    # A small ring wraps exactly like the original, merging reused-time
    # connections; there is no slot-retirement policy hidden in this wrapper.
    wrapped = OriginalRuntime(16, 16, 8, 32, time_neurons=4)
    for frame in frames[:3]:
        wrapped.record(*frame)
    assert wrapped.clock.当前时间 == 2 and wrapped.clock.时间激活.sum() == 1
    np.testing.assert_array_equal(wrapped.recall_at_time(0)['visual'], frames[0][0] | frames[2][0])
    assert wrapped.index.时间到输出[0] == set(np.flatnonzero(frames[0][3] | frames[2][3]))
    return {'same_time_three_modality_completion': 4,
            'native_audio_forward_offset_preserved_internal_steps': 2,
            'recall_does_not_advance_clock_or_learn': True,
            'unknown_index_cue_rejected': True,
            'wrap_uses_original_connection_union_semantics': True}


def index_input_binding_checks():
    """Distinct forward/thought fixture; no claim of real acoustic encoding."""
    runtime = OriginalRuntime(16, 16, 8, 32, time_neurons=32)
    default = OriginalRuntime(16, 16, 8, 32, time_neurons=32)
    frames = []
    for number in range(2):
        v, a, m, thought, forward = (np.zeros(width, bool) for width in (16, 16, 8, 32, 32))
        v[number*2:number*2+2] = True
        a[number*2:number*2+2] = True
        m[number] = True
        thought[28+number*2:30+number*2] = True
        forward[number*8:number*8+4] = True
        cue = np.zeros(32, bool)
        cue[number*8:number*8+2] = True
        frames.append((v, a, m, thought, forward, cue))
        info = runtime.record(v, a, m, thought, index_pfc_bool=forward)
        default.record(v, a, m, thought)
        assert info['index_pfc_active_count'] == 4 and info['explicit_index_pfc_input']
        assert runtime.index.时间到输出[number*2] == set(np.flatnonzero(forward))
        recalled = runtime.recall_from_pfc(cue)
        assert recalled['time'] == number*2 and recalled['ratio'] == 1.
        for name, value in (('visual', v), ('audio', a), ('motor', m)):
            np.testing.assert_array_equal(recalled[name], value)
        assert default.recall_from_pfc(cue)['time'] is None
        assert default.recall_from_pfc(thought)['time'] == number*2
    # Index binding cannot silently substitute forward activity into the
    # original thought->thought transition learner or create a second index.
    expected_sources = set(np.flatnonzero(frames[0][3]))
    expected_targets = set(np.flatnonzero(frames[1][3]))
    assert set(runtime.pfc.联想.表) == expected_sources
    assert all(set(row) == expected_targets for row in runtime.pfc.联想.表.values())
    equal_tree(_table_snapshot(runtime.pfc.联想), _table_snapshot(default.pfc.联想))
    equal_tree(_table_snapshot(runtime.pfc.去抑), _table_snapshot(default.pfc.去抑))
    np.testing.assert_array_equal(runtime.previous_actual_pfc, frames[1][3])
    before = edge_digest(runtime)
    runtime.record(*frames[0][:4], learning=False, index_pfc_bool=frames[0][4])
    assert edge_digest(runtime) == before and runtime.clock.当前时间 == 6
    restored, _ = safe_roundtrip(runtime)
    equal_tree(runtime.snapshot(), restored.snapshot())
    np.testing.assert_array_equal(restored.recall_from_pfc(frames[1][5])['visual'], frames[1][0])
    return {'explicit_same_frame_forward_index_partial_cue_completion': 2,
            'default_thought_index_behavior_unchanged': True,
            'thought_to_thought_hebb_connections_unchanged': True,
            'same_single_index_and_same_frame_time': True,
            'explicit_index_frozen_learning_and_npz_restore_checked': True,
            'scope': 'Distinct discrete forward/thought activity binding fixture, not real PCM recognition.'}


def state_checks(runtime):
    # Deliberate public parameter intervention must survive state export.
    runtime.pfc.门槛 = .91
    runtime.pfc.去抑抵消 = .21
    runtime.pfc.联想.强化量 = .01
    restored, npz_bytes = safe_roundtrip(runtime)
    equal_tree(runtime.snapshot(), restored.snapshot())
    assert restored.clock is not runtime.clock
    for index, frame in enumerate(random_frames(10)):
        learning = index % 3 != 1
        before_other_time = restored.clock.当前时间
        runtime.record(*frame, learning=learning)
        assert restored.clock.当前时间 == before_other_time
        restored.record(*frame, learning=learning)
        first = runtime.pfc_microstep(frame[3], frame[3].astype(float)*1.7)
        second = restored.pfc_microstep(frame[3], frame[3].astype(float)*1.7)
        equal_tree(first, second)
        equal_tree(runtime.recall_from_pfc(frame[3]), restored.recall_from_pfc(frame[3]))
        equal_tree(runtime.recall_from_audio(frame[1], steps=4), restored.recall_from_audio(frame[1], steps=4))
        equal_tree(runtime.snapshot(), restored.snapshot())
    assert not np.shares_memory(runtime.clock.时间激活, restored.clock.时间激活)
    return {'npz_allow_pickle_false_bytes': npz_bytes, 'exact_continuation_frames': 10,
            'independent_clone_shared_clock_state': True, 'public_pfc_parameter_intervention_restored': True}


def adapter_width_check():
    started = time.perf_counter()
    runtime = OriginalRuntime(2640, 2560, 80, 5280, time_neurons=1728000,
                              pfc_connection_parameters={'强化量': .01, '消失下限': .00005})
    construction = time.perf_counter()-started
    rng = np.random.default_rng(2026090618)
    timings = []
    for _ in range(5):
        frame = []
        for width, count in ((2640, 64), (2560, 64), (80, 8), (5280, 24)):
            activity = np.zeros(width, bool)
            activity[rng.choice(width, count, replace=False)] = True
            frame.append(activity)
        start = time.perf_counter()
        runtime.record(*frame)
        runtime.pfc_microstep(frame[3], frame[3].astype(float)*3., diagnostics=False)
        timings.append(1000*(time.perf_counter()-start))
    assert runtime.clock.时间激活.sum() == 1 and runtime.frames_recorded == 5
    return {'widths': runtime.widths, 'time_cells': runtime.clock.时间神经元数量,
            'pfc_hebb_increment': runtime.pfc.联想.强化量,
            'pfc_disappearance_threshold': runtime.pfc.联想.消失下限,
            'construction_seconds_after_original_imports': construction,
            'synthetic_active_counts': {'visual': 64, 'audio': 64, 'motor': 8, 'pfc': 24},
            'record_and_microstep_median_ms': float(np.median(timings)),
            'record_and_microstep_max_ms': float(np.max(timings)),
            'note': 'Fixed dimensions smoke only. Not a dense-activity performance guarantee or embodied learning result.'}


def main():
    start = time.perf_counter()
    runtime, equivalence = equivalence_checks()
    source_hashes_before = dict(runtime.source_hashes)
    result = {'status': 'passed', 'scope': 'Unchanged original-class orchestration and snapshot acceptance; no navigation training.',
        'runtime_sha256': hashlib.sha256((HERE/'original_runtime.py').read_bytes()).hexdigest(),
        'original_source_sha256': source_hashes_before,
        'equivalence': equivalence, 'temporal': temporal_checks(),
        'index_input_binding': index_input_binding_checks(),
        'snapshot': state_checks(runtime), 'adapter_width_smoke': adapter_width_check(),
        'limits': [
            'The original audio feature->time offset is retained; it must not be presented as guaranteed same-frame sound-to-vision alignment.',
            'The original hippocampal index retains its single winning-time/tie behavior. No new disambiguator is added.',
            'A full turn of the original ring reuses time identifiers and can merge old/new content. This wrapper does not introduce retirement or extra epoch labels.',
            'Original additive Hebb, raw current sums and original global inhibition are preserved; large activity can still saturate or be expensive.',
            'PFC transition learning is once per actual outer frame; three modality memories learn on each of the two original internal clock steps.',
            'Safe snapshots contain only original connection/state data and explicit parameters, without goal, route, reward, fact/value or action-planner fields.'
        ]}
    from original_runtime import ROOT
    assert source_hashes_before == {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_hashes_before}
    result['original_sources_unchanged'] = True
    result['elapsed_seconds'] = time.perf_counter()-start
    output = HERE/'results'
    output.mkdir(exist_ok=True)
    (output/'original_runtime_acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'original_runtime_acceptance.md').write_text(
        '# 原架构运行层验收\n\n'
        '直接正常导入并实例化原时间环、三个原记忆类、海马索引与前额叶联想区。原文件哈希未变。\n\n'
        '- 12 帧包装层调用与逐步直接调用原类的时钟、活动、全部连接权重和维护状态逐值一致。\n'
        '- 12 次前额叶微步的实际放电、兴奋电流、净抑制与原类逐值一致，思考本身不自动写入经历连接。\n'
        '- 冻结学习时真实时间/活动继续推进，连接、海马索引和遗忘维护完全不变。\n'
        '- 12 个声音线索的播放与原听觉记忆方法逐值一致；同刻三模态索引补全通过，原声音入口偏后两格的语义保持。\n'
        '- 可选 index_pfc_bool 把同刻实际前额叶正向输出绑定唯一海马索引，念头前后 Hebb 不变；省略时保持旧默认。离散夹具中部分正向线索补全 2/2，冻结与存档恢复通过。\n'
        '- 安全 NPZ（allow_pickle=False）恢复后连续 10 帧，包括学习/冻结、读出、PFC 参数干预和衰减状态全部逐值一致。克隆时钟互不影响。\n'
        '- 2640/2560/80/5280 宽度和 1728000 时间细胞构造/写入/微步通过；PFC 强化 .01、消失阈值 .00005 均保存。\n\n'
        '这是运行层验收，尚不代表真实导航或交流成功。原时间环绕回后会把新旧内容连接到复用的时间编号，原音记忆入口也保留时间偏移；没有在包装层另加替代算法。\n',
        encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'runtime_sha256': result['runtime_sha256'], 'equivalence': equivalence,
        'snapshot': result['snapshot'], 'adapter_width_smoke': result['adapter_width_smoke']}))


if __name__ == '__main__':
    main()
