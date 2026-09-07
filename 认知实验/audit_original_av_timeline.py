"""Original-class shared-time AV mechanism fixture, not PCM/RGB recognition.

The fixture supplies explicit discrete feature activations and a hypothetical
partial PFC cue. Only the original learning methods create connections. A
synthetic concatenated PFC code does NOT establish that the original fixed
PFC encoder can produce that partial code from real sound alone.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_FILES = ('海马体时间区.py', '视觉记忆区.py', '听觉记忆区.py',
                '前额叶区.py', '权重连接管理.py')
SEED = 202609061


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def original_class(filename, name, namespace, include_literal_assignments=False):
    """Compile the unchanged class AST, excluding module-level born networks."""
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    selected = []
    for node in tree.body:
        if include_literal_assignments and isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == name:
            selected.append(node)
    assert any(isinstance(node, ast.ClassDef) for node in selected), (filename, name)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def build_original_system():
    namespace = {'np': np, '__name__': 'isolated_original_av_fixture'}
    table = original_class('权重连接管理.py', '连接表', namespace, True)
    clock_type = original_class('海马体时间区.py', '时间环', namespace)
    clock = clock_type(时间神经元数量=512, 每步秒数=.05, 每帧步数=2)
    common = {'np': np, '连接表': table, '时钟': clock, '__name__': 'isolated_original_av_fixture'}
    visual_type = original_class('视觉记忆区.py', '记忆区', dict(common))
    auditory_type = original_class('听觉记忆区.py', '记忆区', dict(common))
    index_type = original_class('前额叶区.py', '海马索引', dict(common))
    visual, auditory, index = visual_type(16), auditory_type(16), index_type()
    assert visual.回忆.__globals__['时钟'] is auditory.回忆.__globals__['时钟'] is clock
    return {'clock': clock, 'visual': visual, 'auditory': auditory, 'index': index}


def codes():
    rng = np.random.default_rng(SEED)
    visual_order, auditory_order = rng.permutation(16), rng.permutation(16)
    visual, auditory = [], []
    for number in range(6):
        v, a = np.zeros(16, bool), np.zeros(16, bool)
        v[visual_order[number*2:number*2+2]] = True
        a[auditory_order[number*2:number*2+2]] = True
        visual.append(v)
        auditory.append(a)
    unknown = np.zeros(16, bool)
    unknown[auditory_order[12:14]] = True
    return visual, auditory, unknown


def train_experience(hold_frames=1, permutation=None, visual_present=True, repeated_sound=False):
    system = build_original_system()
    visual_codes, auditory_codes, unknown = codes()
    permutation = list(range(6)) if permutation is None else list(permutation)
    timeline, onsets = [], []
    for pair in range(6):
        vision = visual_codes[permutation[pair]] if visual_present else np.zeros(16, bool)
        sound = auditory_codes[0 if repeated_sound else pair]
        # Explicit sparse PFC fixture code, not the original fixed encoder.
        pfc = np.concatenate((vision, sound))
        onsets.append(int(system['clock'].当前时间))
        for frame in range(hold_frames):
            for microstep in range(system['clock'].每帧步数):
                old, new = system['clock'].走一步()
                if microstep == 0:
                    system['index'].学习(old, pfc)
                system['visual'].学习(old, new, vision)
                system['auditory'].学习(old, new, sound)
                assert system['clock'].时间激活.sum() == 1
                timeline.append({'old_time': int(old), 'new_time': int(new),
                    'pair': pair, 'frame_within_pair': frame, 'microstep': microstep,
                    'visual_features': np.flatnonzero(vision).tolist(),
                    'auditory_features': np.flatnonzero(sound).tolist(),
                    'index_written': microstep == 0})
    system.update(visual_codes=visual_codes, auditory_codes=auditory_codes,
                  unknown=unknown, timeline=timeline, onsets=onsets, permutation=permutation)
    return system


def content_at(memory, timestamp, threshold=1.):
    """Read only actual time->feature edges, as 前额叶区.py:420-425 does."""
    result = np.zeros(memory.总数, bool)
    if timestamp is not None:
        for feature, weight in memory.时间到特征.查(int(timestamp)).items():
            if weight >= threshold:
                result[feature] = True
    return result


def system_digest(system):
    tables = {}
    for name in ('visual', 'auditory'):
        memory = system[name]
        for direction in ('特征到时间', '时间到特征'):
            table = getattr(memory, direction)
            tables[name+direction] = {'weights': table.表, 'maintenance': table.维护次数,
                                      'last_decay': table.上次衰减维护, 'edges': table.连接条数}
    tables['index'] = {int(key): sorted(value) for key, value in system['index'].时间到输出.items()}
    digest = hashlib.sha256(json.dumps(tables, sort_keys=True).encode())
    digest.update(system['clock'].时间激活.tobytes())
    digest.update(str(system['clock'].当前时间).encode())
    return digest.hexdigest()


def query_pairs(system):
    before = system_digest(system)
    rows = []
    for pair, sound in enumerate(system['auditory_codes']):
        expected = system['visual_codes'][system['permutation'][pair]]
        audio_playback = system['auditory'].回忆(sound, 步数=4, 阈值=1)
        native_time = audio_playback[0][0] if audio_playback else None
        native_visual = content_at(system['visual'], native_time)
        partial_pfc = np.concatenate((np.zeros(16, bool), sound))
        index_time, ratio = system['index'].锁定时间(partial_pfc)
        index_visual = content_at(system['visual'], index_time)
        rows.append({'pair': pair, 'experience_onset': system['onsets'][pair],
            'expected_visual_features': np.flatnonzero(expected).tolist(),
            'sound_features': np.flatnonzero(sound).tolist(),
            'native_auditory_start_time': native_time,
            'native_offset_internal_steps': None if native_time is None else native_time-system['onsets'][pair],
            'native_start_visual': np.flatnonzero(native_visual).tolist(),
            'native_same_pair_visual': bool(np.array_equal(native_visual, expected)),
            'native_crossmodal_playback': [{'time': int(t),
                'visual_features': np.flatnonzero(content_at(system['visual'], t)).tolist()}
                for t, _ in audio_playback],
            'index_time_from_partial_pfc': index_time, 'index_vote_ratio': ratio,
            'index_visual': np.flatnonzero(index_visual).tolist(),
            'index_same_pair_visual': bool(np.array_equal(index_visual, expected))})
    assert system_digest(system) == before, 'Readout must not learn or move the shared clock'
    return rows


def run():
    begin = time.perf_counter()
    hashes = source_hashes()
    rapid = train_experience(hold_frames=1)
    rapid_rows = query_pairs(rapid)
    assert all(row['native_offset_internal_steps'] == 2 for row in rapid_rows)
    assert not any(row['native_same_pair_visual'] for row in rapid_rows)
    assert all(row['index_same_pair_visual'] and row['index_vote_ratio'] == 1. for row in rapid_rows)

    held = train_experience(hold_frames=3)
    held_rows = query_pairs(held)
    assert all(row['native_same_pair_visual'] and row['index_same_pair_visual'] for row in held_rows)
    assert all(row['native_offset_internal_steps'] == 2 for row in held_rows)

    first_sound = rapid['auditory_codes'][0]
    one_feature_cue = np.concatenate((np.zeros(16, bool), first_sound.copy()))
    one_feature_cue[np.flatnonzero(one_feature_cue)[1]] = False
    one_time, one_ratio = rapid['index'].锁定时间(one_feature_cue)
    assert one_time == 0 and one_ratio == 1.
    mixed_cue = np.concatenate((np.zeros(16, bool), rapid['auditory_codes'][0] | rapid['auditory_codes'][1]))
    mixed_time, mixed_ratio = rapid['index'].锁定时间(mixed_cue)
    assert mixed_time is None and mixed_ratio == .5
    unknown_cue = np.concatenate((np.zeros(16, bool), rapid['unknown']))
    unknown_time, unknown_ratio = rapid['index'].锁定时间(unknown_cue)
    assert unknown_time is None and unknown_ratio == 0.
    assert rapid['auditory'].回忆(rapid['unknown']) == []
    assert content_at(rapid['visual'], 0, threshold=1.).sum() == 2
    assert content_at(rapid['visual'], 0, threshold=1.000001).sum() == 0

    # Only the ordering of experienced pairs changes. No weights are assigned.
    permutation = [3, 5, 1, 4, 0, 2]
    swapped = train_experience(hold_frames=3, permutation=permutation)
    swapped_rows = query_pairs(swapped)
    assert all(row['index_same_pair_visual'] for row in swapped_rows)
    assert all(not np.array_equal(content_at(swapped['visual'], row['index_time_from_partial_pfc']),
                                  swapped['visual_codes'][row['pair']]) for row in swapped_rows)

    sound_only = train_experience(hold_frames=3, visual_present=False)
    for sound in sound_only['auditory_codes']:
        timestamp, ratio = sound_only['index'].锁定时间(np.concatenate((np.zeros(16, bool), sound)))
        assert timestamp is not None and ratio == 1.
        assert not content_at(sound_only['visual'], timestamp).any()
    ambiguous = train_experience(hold_frames=1, repeated_sound=True)
    timestamp, ratio = ambiguous['index'].锁定时间(np.concatenate((np.zeros(16, bool), first_sound)))
    assert timestamp == 0 and ratio == 1.
    compatible_times = [int(t) for t, feature_ids in ambiguous['index'].时间到输出.items()
                        if set(np.flatnonzero(np.concatenate((np.zeros(16, bool), first_sound)))).issubset(feature_ids)]
    assert len(compatible_times) == 6

    fresh = build_original_system()['clock']
    one_frame_return = fresh.一帧()
    assert one_frame_return == (1, 2)
    assert hashes == source_hashes(), 'Original source files changed during audit'
    return {'status': 'completed_expected_alignment_findings', 'seed': SEED,
        'scope': 'Discrete time-index mechanism unit fixture; not real PCM/RGB or fixed-PFC encoder learning.',
        'source_sha256_before_and_after': hashes,
        'original_classes_extracted_without_changes': ['时间环', '连接表', '视觉记忆区.记忆区', '听觉记忆区.记忆区', '海马索引'],
        'shared_clock': {'instances': 1, 'cells': 512, 'active_each_step': 1,
                         'seconds_per_internal_step': .05, 'internal_steps_per_frame': 2,
                         'one_frame_method_returns_final_internal_transition': list(one_frame_return)},
        'rapid_changes_1_frame_per_pair': rapid_rows,
        'held_3_frames_per_pair': held_rows,
        'rapid_timeline': rapid['timeline'],
        'first_sound_feature_to_time_edges': {
            str(int(feature)): rapid['auditory'].特征到时间.查(int(feature))
            for feature in np.flatnonzero(first_sound)},
        'controls': {
            'partial_one_feature_index_time': one_time, 'partial_one_feature_vote_ratio': one_ratio,
            'two_disjoint_known_sounds_index_time': mixed_time, 'two_disjoint_known_sounds_vote_ratio': mixed_ratio,
            'unknown_sound_rejected': True, 'sound_without_visual_experience_does_not_recall_color': True,
            'changed_cooccurrence_permutation': permutation,
            'changed_cooccurrence_recalled_actual_pairing': True,
            'readout_does_not_change_weights_or_clock': True,
            'single_exposure_time_visual_weight': 1.,
            'visual_threshold_1_active': 2, 'visual_threshold_above_1_active': 0,
            'same_sound_multiple_colors_matching_times': compatible_times,
            'same_sound_multiple_colors_selected_time': timestamp,
            'same_sound_multiple_colors_selected_ratio': ratio,
        },
        'interpretation': [
            'Under the original two-internal-step experience loop, native sound-feature entry starts two internal steps later than the original same-frame color. Rapid pair changes therefore recall the following color or unwritten time.',
            'Holding each pairing for three frames yields correct color through native sound entry but retains the two-step offset; long presentations can conceal the alignment issue.',
            'An appropriate partial PFC cue locks the stored frame-onset time and completes the same-frame color through actual time-to-visual edges.',
            'The fixture supplies the partial PFC cue by concatenating explicit feature lanes. It does not establish that real auditory input alone yields this cue through the original fixed PFC network.',
            'Index vote ratio divides by query size: one matching cue cell can yield 1.0. Repeated identical sound paired with different colors remains ambiguous and the original tie rule chooses the earliest inserted match.',
            'No connection was set by a sound-to-color rule; all learned edges arose through the original learning calls on the shared ordered experience.',
        ], 'elapsed_seconds': time.perf_counter()-begin}


def main():
    result = run()
    output = HERE / 'results'
    output.mkdir(exist_ok=True)
    (output / 'original_av_timeline_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    text = (
        '# 原类共同视听时间索引单测\n\n'
        '只使用原时间环、视觉/听觉记忆、海马索引与连接表类的未改动 AST；避开模块全局大网。'
        '输入为显式离散神经元活动，部分前额叶线索也由夹具提供，因此不是实际 PCM/RGB 端到端学会。\n\n'
        '| 经历条件 | 声音特征原生时间入口补全同对视觉 | 部分 PFC 线索经海马索引补全同刻视觉 |\n'
        '|---|---:|---:|\n'
        '| 每对保持 1 帧，随后立即换对 | 0/6 | 6/6 |\n'
        '| 每对保持 3 帧 | 6/6 | 6/6 |\n\n'
        '原生声音入口两组条件均偏后 2 个内部时间格（0.1 秒）。原记忆写的是“旧特征 → 新时间”和“旧时间 → 新特征”；'
        '声音回忆从特征指向的时间开始，所以立即换画面会读到下一对。长时间保持共现会掩盖这个偏移。'
        '原海马索引在帧首旧时间写入，因此获得合适的部分 PFC 线索时，同刻内容能够正确补全。没有在审计里修补偏移。\n\n'
        '控制检查：更换共现配对后回忆跟随实际经历；未见声音被拒绝；只有声音而没有视觉经历不会产生颜色；'
        '回忆不改变权重或共享时钟；一次共现的时间→视觉连接权重为 1，读出阈值大于 1 时没有视觉细胞亮起。\n\n'
        '限制：海马票数比例按查询大小归一，只有一个匹配线索细胞也可得到 100%；'
        '同一声音与六个不同颜色共同出现时有六个同票时刻，原类选择最先写入者。'
        '这不是消歧能力，也不能保证真实声音通过出生前额叶网络产生可用索引线索。\n\n'
        '所有记忆使用同一个时钟实例，每步恰有一个时间细胞亮起。时间环一帧方法返回最后一次内部跃迁 (1, 2)，'
        '不是整帧跨度 (0, 2)；本检查依照原前额叶示例逐内部步写入。原文件哈希前后一致。\n'
    )
    (output / 'original_av_timeline_audit.md').write_text(text, encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'rapid_native_same_pair': sum(row['native_same_pair_visual'] for row in result['rapid_changes_1_frame_per_pair']),
        'rapid_index_same_pair': sum(row['index_same_pair_visual'] for row in result['rapid_changes_1_frame_per_pair']),
        'held_native_same_pair': sum(row['native_same_pair_visual'] for row in result['held_3_frames_per_pair']),
        'native_offset_internal_steps': 2, 'source_hashes_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
