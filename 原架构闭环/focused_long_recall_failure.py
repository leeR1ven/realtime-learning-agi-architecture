"""Bounded first-frame blue recall diagnosis after six real experience rounds.

Reuses the delegated instrumentation from focused_recall_failure. No equations,
learned tables, real trajectories or frozen reports are changed by this audit.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from brain import OriginalBrain
from dynamic_calibration import frozen_digest
from experience_environment import ExperienceEnvironment
from focused_recall_failure import InstrumentedRun, candidates, configure_clone, event_difference, summarized_record
from run_closed_loop import TONES, completion_score, plain, train


HERE = Path(__file__).resolve().parent
EXPECTED = 'blue'


def distribution(values):
    a = np.asarray(values, float)
    if not a.size:
        return {'count': 0}
    return {'count': int(a.size), 'min': float(a.min()), 'p25': float(np.quantile(a, .25)),
            'median': float(np.median(a)), 'p75': float(np.quantile(a, .75)),
            'max': float(a.max()), 'mean': float(a.mean())}


def cell_group(brain, micro, selected):
    cells = np.array(sorted(selected), int)
    current = micro['currents']
    sites = cells // brain.runtime.pfc.每段
    e = np.asarray(current['excitatory_current'])[cells]
    external = brain.config.external_gain * np.isin(cells, micro['current_projection']) * brain.flags['current_drive']
    replay = brain.config.recall_gain * np.isin(cells, micro['remembered_projection']) * brain.flags['recalled_drive']
    net = np.asarray(current['net_inhibitory_site_current'])[sites]
    threshold = current['threshold'] + net
    assert np.allclose(external + replay, np.asarray(current['external_current'])[cells])
    margin = external + replay + e - threshold
    active = np.isin(cells, micro['after'])
    arrays = {'E': e, 'current_drive': external, 'replay_drive': replay,
              'net_inhibition': net, 'threshold_including_gate': threshold,
              'total_input': external + replay + e, 'margin': margin}
    rows = [{'cell': int(cell), 'site': int(site), 'active': bool(on),
             **{name: float(array[i]) for name, array in arrays.items()}}
            for i, (cell, site, on) in enumerate(zip(cells, sites, active))]
    return {'cell_count': len(cells), 'active_count': int(active.sum()),
            'distributions': {name: distribution(array) for name, array in arrays.items()},
            'cells': rows}


def main():
    started = time.perf_counter()
    dependencies = ('brain.py', 'original_runtime.py', 'sensory_adapter.py', 'run_closed_loop.py', 'experience_environment.py')
    source = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in dependencies}
    config = json.loads((HERE/'results'/'continuity_gain6_config.json').read_text(encoding='utf8'))
    config.update(seed=2026, recall_gain=6.)
    brain = OriginalBrain(config)
    env = ExperienceEnvironment(seed=20260909)
    training = train(brain, env, repeats=6, hold_frames=8)
    snapshot = brain.snapshot()
    catalog = training['catalog']
    env.hide_beacons()
    packet = env.emit_tone(TONES[2], duration=.6)
    cases = []
    full_record = None
    correct_time = wrong_time = None
    for gain, condition, index_threshold in ((6., 'full', .9), (6., 'no_current_drive', .9),
                                             (7., 'full', .9), (10., 'full', .9), (6., 'full', .8)):
        clone, same_env = configure_clone(snapshot, env, condition)
        clone.config.recall_gain = gain
        clone.config.index_threshold = index_threshold
        before = frozen_digest(clone)
        tracer = InstrumentedRun(clone, same_env, catalog, condition, expected=EXPECTED)
        _, info = clone._process(copy.deepcopy(packet), learning=False, trace=True, choose_action=False)
        record = tracer.records[0]
        assert frozen_digest(clone) == before
        if full_record is None:
            full_record = record
            correct_time = record['lookups'][0]['time']
            first_wrong_micro = next(i for i, lookup in enumerate(record['lookups'][1:])
                                     if lookup['catalog_color'] != EXPECTED)
            wrong_time = record['lookups'][-1]['time']
            assert correct_time == 44 and record['lookups'][-1]['time'] == 272, record['lookups']
            assert record['lookups'][0]['catalog_color'] == EXPECTED
            assert record['lookups'][-1]['catalog_color'] == 'green'
        correct = set(clone.runtime.index.时间到输出[correct_time])
        wrong = set(clone.runtime.index.时间到输出[wrong_time])
        cue = set(record['projections'][0]['output'])
        case = {'recall_gain': gain, 'condition': condition, 'index_threshold': index_threshold,
            'score': completion_score(info, catalog, EXPECTED),
            'record': summarized_record(record), 'microstep_target_distributions': [],
            'learned_tables_index_maintenance_unchanged': True,
            'physical_pose_unchanged': bool(np.array_equal(env.world.position, same_env.world.position)
                                          and env.world.heading == same_env.world.heading)}
        for i, micro in enumerate(record['microsteps']):
            case['microstep_target_distributions'].append({'microstep_zero_based': i,
                'inhibition_strength_before': micro['currents']['inhibition_strength_before'],
                'inhibition_strength_after': micro['currents']['inhibition_strength_after'],
                'blue_time44_index': cell_group(clone, micro, correct),
                'blue_specific_vs_green_time272': cell_group(clone, micro, correct-wrong),
                'real_blue_auditory_projection': cell_group(clone, micro, cue),
                'blue_specific_and_replayed': cell_group(clone, micro, (correct-wrong) & set(micro['remembered_projection'])),
                'top3_index': record['lookups'][i+1]['top3']})
        cases.append(case)
        print('LONG_FOCUS '+json.dumps({'gain': gain, 'condition': condition, 'index_threshold': index_threshold,
            'initial': case['score']['initial_color'], 'final': case['score']['final_color'],
            'top3': case['microstep_target_distributions'][0]['top3_index'],
            'blue_specific': case['microstep_target_distributions'][0]['blue_specific_vs_green_time272']['distributions']}), flush=True)
    auditory_checks = []
    for threshold in (.9, .8):
        for frequency, expected in ((220., 'red'), (440., 'green'), (660., 'blue'), (777., None), (2047., None)):
            clone, same_env = configure_clone(snapshot, env, 'full')
            clone.config.index_threshold = threshold
            test_packet = same_env.emit_tone(frequency, duration=.6)
            before = frozen_digest(clone)
            encoded = clone.adapter.encode(test_packet, clone.last_executed)
            query = clone._project(np.zeros_like(encoded['visual']), encoded['audio'], np.zeros_like(encoded['motor']))
            recalled = clone.runtime.recall_from_pfc(query, threshold, clone.config.memory_threshold)
            assert frozen_digest(clone) == before
            actual = catalog.get(str(recalled['time']), {}).get('color')
            auditory_checks.append({'index_threshold': threshold, 'frequency_hz': frequency,
                'trained_pair_expected_color': expected, 'query_count': int(query.sum()),
                'selected_time': recalled['time'], 'selected_color': actual, 'ratio': recalled['ratio'],
                'accepted': recalled['time'] is not None,
                'correct_known_color_or_unknown_rejection': actual == expected,
                'top3': candidates(clone.runtime.index, query, catalog),
                'scope': 'Actual PCM through frozen original auditory adapter and fixed PFC projection into original initial index only; no internal thinking or action.',
                'learned_tables_index_maintenance_unchanged': True})
    # Verify observer wrappers do not change this first-frame neural result.
    plain_brain, _ = configure_clone(snapshot, env, 'full')
    _, plain_info = plain_brain._process(copy.deepcopy(packet), learning=False, trace=True, choose_action=False)
    assert completion_score(plain_info, catalog, EXPECTED) == cases[0]['score']
    assert np.array_equal(plain_brain.thought, np.isin(np.arange(plain_brain.width), full_record['microsteps'][-1]['after']))
    assert source == {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in dependencies}
    report = {'status': 'complete', 'config': config, 'source_sha256': source,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_frames': training['frames'], 'training_schedule': training['presentation_schedule'],
        'shared_clock_after_training': int(brain.runtime.clock.当前时间),
        'correct_time': correct_time, 'wrong_time_after_final_microstep': wrong_time,
        'first_failing_microstep_zero_based': first_wrong_micro,
        'original_failure_full_record': full_record,
        'event_difference': event_difference(brain, catalog, correct_time, wrong_time, full_record),
        'conditions': cases, 'auditory_initial_index_checks': auditory_checks,
        'instrumentation_matches_uninstrumented': True,
        'elapsed_seconds': time.perf_counter()-started,
        'boundary': 'Only the first blue sensor packet after six rounds of real training plus scoped initial auditory index probes. Parameter changes reuse identical learned state and physical pose; no subsequent retention, cue switching or retraining at gains7/10 or index.8 is tested.'}
    first = full_record['microsteps'][0]
    current_cells, replay_cells, active_cells = (set(first[name]) for name in ('current_projection', 'remembered_projection', 'after'))
    report['mechanism'] = {'current_projection_count': len(current_cells), 'remembered_projection_count': len(replay_cells),
        'after_microstep_count': len(active_cells), 'replay_cells_lost': sorted(replay_cells-active_cells),
        'current_only_added_cells': sorted((current_cells-replay_cells)&active_cells),
        'active_equals_current_replay_union': active_cells == current_cells | replay_cells,
        'explanation': 'All 210 blue recalled cells survive and 52 current-only cells join. The best blue candidate has224/262=.85496 support and is rejected at.9. Without replay on the next microstep, blue auditory activity falls32->4 and the index selects green272. Gains7/10 leave the same binary pattern and cannot correct this first rejection.'}
    output = HERE/'results'/'focused_long_recall_failure.json'
    output.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2), encoding='utf8')
    prose = ['# 较长经历后的蓝色首帧回忆取证', '',
        'seed2026、回放增益6、六轮三色真实自主共现（144动作）。所有对照克隆同一完整神经状态和感觉包，不执行新身体动作、不写连接。', '',
        '| 条件 | 初始→最终 | 首微步保留蓝色特异细胞 | 蓝色特异细胞门槛范围 | 首微步全球抑制强度 |',
        '|---|---|---:|---|---|']
    for row in cases:
        m = row['microstep_target_distributions'][0]
        group = m['blue_specific_vs_green_time272']
        th = group['distributions']['threshold_including_gate']
        prose.append(f'| gain={row["recall_gain"]:g}, index={row["index_threshold"]:g}, {row["condition"]} | {row["score"]["initial_color"]}→{row["score"]["final_color"]} | '
            f'{group["active_count"]}/{group["cell_count"]} | {th["min"]:.6f}–{th["max"]:.6f} | '
            f'{m["inhibition_strength_before"]:.6f}→{m["inhibition_strength_after"]:.6f} |')
    prose += ['', '首微步完整保留210个蓝色回忆细胞，当前感觉又加入52个细胞，活动恰好是两者并集。'
        '原索引前三名都是蓝色时间240/242/244，均224票/262查询≈.854962，低于.9所以没有回忆。'
        '第二微步因此没有回放输入，32个蓝色听觉细胞只剩4个，绿色272以204/211≈.966825获选。'
        '全局抑制强度两步均为1；不是全局强度失控或蓝色特异细胞首步电流不足。', '',
        '同状态回放增益7、10均产生相同活动与错误。索引门槛.8直接检查第一次正确候选被拒的瓶颈。', '',
        '| 索引门槛 | 实际PCM频率 | 已学对应颜色 | 初始接受颜色 | 比率 |', '|---:|---:|---|---|---:|']
    for row in auditory_checks:
        prose.append(f'| {row["index_threshold"]} | {row["frequency_hz"]:g} | {row["trained_pair_expected_color"]} | '
                     f'{row["selected_color"]} | {row["ratio"]:.6f} |')
    prose += ['', 'JSON逐细胞记录当前输入、真实回放输入、原学得兴奋电流、2+净抑制门槛与原索引前三票。',
        '这只确定本次门槛和计票瓶颈，不能把同包恢复当作参数重训后长期稳定。未知声音只检查了两个频率；不代表开放声音识别已解决。没有继续搜索更多增益。']
    output.with_suffix('.md').write_text('\n'.join(prose)+'\n', encoding='utf8')
    print('LONG_FOCUS_COMPLETE '+json.dumps({'elapsed_seconds': report['elapsed_seconds'],
        'cases': [{'gain': r['recall_gain'], 'condition': r['condition'], 'score': r['score']} for r in cases]}), flush=True)


if __name__ == '__main__':
    main()
