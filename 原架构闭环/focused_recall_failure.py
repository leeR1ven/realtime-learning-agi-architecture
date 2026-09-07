"""Trace the original runtime's seed-2027 green->red silent recall drift.

Instrumentation delegates every neural operation unchanged and records actual
arguments/results. Labels, time-catalog comparisons and vote-only interventions
are offline diagnostics, never input to the embodied brain.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from brain import OriginalBrain, packet_digest
from experience_environment import ExperienceEnvironment
from run_closed_loop import COLORS, TONES, completion_score, plain, train
from dynamic_calibration import frozen_digest


HERE = Path(__file__).resolve().parent
EXPECTED = 'green'
CONDITIONS = ('full', 'no_pfc_recurrence', 'no_disinhibition', 'no_current_drive', 'no_motor_recall')


def ids(value):
    return np.flatnonzero(value).tolist()


def overlap(a, b):
    a, b = set(a), set(b)
    return {'a_count': len(a), 'b_count': len(b), 'intersection': len(a & b),
            'jaccard': len(a & b)/max(1, len(a | b)),
            'a_only': sorted(a-b), 'b_only': sorted(b-a)}


def candidates(index, cue, catalog, limit=3):
    query = set(np.flatnonzero(cue))
    ranked = []
    for insertion, (timestamp, stored) in enumerate(index.时间到输出.items()):
        votes = len(query & stored)
        if votes:
            ranked.append({'time': int(timestamp), 'votes': votes,
                'stored_count': len(stored), 'query_count': len(query),
                'ratio': votes/len(query), 'size_gap': abs(len(stored)-len(query)),
                'insertion_order': insertion, 'catalog_color': catalog.get(str(timestamp), {}).get('color')})
    # The original key is (votes, -size_gap), with first-insertion ties retained.
    ranked.sort(key=lambda row: (row['votes'], -row['size_gap']), reverse=True)
    return ranked[:limit]


class InstrumentedRun:
    def __init__(self, brain, env, catalog, condition, expected=EXPECTED):
        self.brain, self.env, self.catalog, self.condition = brain, env, catalog, condition
        self.expected = expected
        self.records = []
        self.current = None
        self.failure_state = self.failure_packet = self.failure_record = None
        self.failure_environment = None
        self._process = brain._process
        self._project = brain._project
        self._microstep = brain._microstep
        self._recall = brain.runtime.recall_from_pfc
        brain._process = self.process
        brain._project = self.project
        brain._microstep = self.microstep
        brain.runtime.recall_from_pfc = self.recall

    def project(self, visual, audio, motor):
        output = self._project(visual, audio, motor)
        if self.current is not None:
            self.current['projections'].append({'visual': ids(visual), 'audio': ids(audio),
                                                'motor': ids(motor), 'output': ids(output)})
        return output

    def recall(self, cue, *args, **kwargs):
        result = self._recall(cue, *args, **kwargs)
        if self.current is not None:
            self.current['lookups'].append({'query': ids(cue), 'time': result['time'],
                'ratio': result['ratio'], 'catalog_color': self.catalog.get(str(result['time']), {}).get('color'),
                'top3': candidates(self.brain.runtime.index, cue, self.catalog)})
        return result

    def microstep(self, external, *, trace):
        previous = self.brain.thought.copy()
        detail = self._microstep(external, trace=True)
        self.current['microsteps'].append({'previous': ids(previous), 'after': ids(self.brain.thought),
            'current_projection': self.current['projections'][1]['output'],
            'remembered_projection': self.current['projections'][-1]['output'],
            'currents': copy.deepcopy(detail)})
        return detail

    def process(self, packet, **kwargs):
        before = self.brain.snapshot() if self.failure_state is None else None
        record = {'observation_number': len(self.records), 'packet_digest': packet_digest(packet),
            'physical_time': float(self.env.world.time), 'position': self.env.world.position.tolist(),
            'heading': float(self.env.world.heading), 'previous_thought': ids(self.brain.thought),
            'last_executed_muscles': self.brain.last_executed.tolist(),
            'projections': [], 'lookups': [], 'microsteps': []}
        self.current = record
        arguments = dict(kwargs, trace=True)
        muscles, info = self._process(packet, **arguments)
        self.current = None
        record.update(sound_rms=float(info['sound_rms']), score=completion_score(info, self.catalog, self.expected),
                      next_muscles=muscles.tolist(), source=info['source'])
        self.records.append(record)
        if self.failure_state is None and not record['score']['correct_final_color']:
            self.failure_state, self.failure_packet, self.failure_record = before, copy.deepcopy(packet), record
            self.failure_environment = copy.deepcopy(self.env)
        return muscles, info


def configure_clone(snapshot, env, condition):
    brain = OriginalBrain.from_snapshot(snapshot)
    brain.prepared = None
    brain.pending = None
    brain.flags['exploration'] = False
    brain.flags['reflex'] = False
    if condition != 'full':
        brain.flags[condition.removeprefix('no_')] = False
    return brain, copy.deepcopy(env)


def summarized_record(record):
    micros = []
    for micro in record['microsteps']:
        detail = micro['currents']
        item = {'previous_count': len(micro['previous']), 'after_count': len(micro['after']),
                'inhibition_strength_before': detail['inhibition_strength_before'],
                'inhibition_strength_after': detail['inhibition_strength_after']}
        for name in ('excitatory_current', 'external_current', 'inhibitory_site_current',
                     'disinhibitory_site_current', 'net_inhibitory_site_current'):
            a = np.asarray(detail[name])
            item[name] = {'max': float(a.max(initial=0.)), 'mean': float(a.mean())}
        micros.append(item)
    return {key: value for key, value in record.items() if key not in ('projections', 'microsteps')} | {'microsteps': micros}


def event_difference(brain, catalog, correct, wrong, record):
    entries = {}
    for name, timestamp in (('correct', correct), ('wrong', wrong)):
        entry = catalog.get(str(timestamp), {})
        entries[name] = {'time': timestamp, **copy.deepcopy(entry),
                         'index_cells': sorted(brain.runtime.index.时间到输出.get(timestamp, ())) }
    current = record['projections'][1]
    comparison = {'events': entries,
        'stored_visual_overlap': overlap(entries['correct'].get('actual_visual_cells', ()),
                                         entries['wrong'].get('actual_visual_cells', ())),
        'stored_motor_overlap': overlap(entries['correct'].get('actual_motor_cells', ()),
                                        entries['wrong'].get('actual_motor_cells', ())),
        'stored_index_overlap': overlap(entries['correct']['index_cells'], entries['wrong']['index_cells'])}
    for name in ('correct', 'wrong'):
        entry = entries[name]
        comparison['current_visual_vs_'+name] = overlap(current['visual'], entry.get('actual_visual_cells', ()))
        comparison['current_motor_vs_'+name] = overlap(current['motor'], entry.get('actual_motor_cells', ()))
        comparison['current_projection_vs_'+name] = overlap(current['output'], entry['index_cells'])
    return comparison


def channel_projection_support(snapshot, packet):
    brain = OriginalBrain.from_snapshot(snapshot)
    encoded = brain.adapter.encode(packet, brain.last_executed)
    v, a, m = (encoded[name] for name in ('visual', 'audio', 'motor'))
    zeros = (np.zeros_like(v), np.zeros_like(a), np.zeros_like(m))
    outputs = {'visual': ids(brain._project(v, zeros[1], zeros[2])),
               'audio': ids(brain._project(zeros[0], a, zeros[2])),
               'motor': ids(brain._project(zeros[0], zeros[1], m))}
    for name, (start, end) in brain.adapter.visual_segments.items():
        selected = np.zeros_like(v)
        selected[start:end] = v[start:end]
        outputs['visual_'+name] = ids(brain._project(selected, zeros[1], zeros[2]))
    return outputs


def decisive_cells(brain, record, micro_index, correct_time, wrong_time, catalog, channel_support):
    micro = record['microsteps'][micro_index]
    current = micro['currents']
    correct = brain.runtime.index.时间到输出[correct_time]
    wrong = brain.runtime.index.时间到输出[wrong_time]
    after = set(micro['after'])
    wrong_only_active = sorted((wrong-correct) & after)
    correct_only_lost = sorted((correct-wrong)-after)
    current_projection = set(micro['current_projection'])
    remembered_projection = set(micro['remembered_projection'])
    query = np.zeros(brain.width, bool)
    query[micro['after']] = True
    removals = []
    recovered = None
    for cell in wrong_only_active:
        query[cell] = False
        removals.append(cell)
        timestamp, ratio = brain.runtime.index.锁定时间(query, 相对门槛=brain.config.index_threshold)
        if catalog.get(str(timestamp), {}).get('color') == EXPECTED:
            recovered = {'time': timestamp, 'ratio': ratio, 'removed_cells': removals.copy()}
            break
    pool = list(dict.fromkeys((recovered['removed_cells'] if recovered else [])+wrong_only_active
                             +correct_only_lost[:4]+correct_only_lost[-4:]))
    cells = []
    pfc = brain.runtime.pfc
    for cell in pool[:12]:
        site = cell//pfc.每段
        e_sources = [(int(source), float(pfc.联想.查(source).get(cell, 0.))) for source in micro['previous']]
        e_sources = [(source, weight) for source, weight in e_sources if weight]
        e_sources.sort(key=lambda row: row[1], reverse=True)
        di_sources = [(int(source), float(pfc.去抑.查(source).get(site, 0.))) for source in micro['previous']]
        di_sources = [(source, weight) for source, weight in di_sources if weight]
        di_sources.sort(key=lambda row: row[1], reverse=True)
        e_total = sum(weight for _, weight in e_sources)
        assert np.isclose(e_total, current['excitatory_current'][cell], atol=1e-12)
        current_drive = brain.config.external_gain*float(cell in current_projection)
        replay_drive = brain.config.recall_gain*float(cell in remembered_projection)
        assert np.isclose(current_drive+replay_drive, current['external_current'][cell])
        threshold = current['threshold']+current['net_inhibitory_site_current'][site]
        cells.append({'cell': cell, 'site': site, 'new_since_previous': cell not in micro['previous'],
            'category': 'wrong_event_only_active' if cell in wrong_only_active else 'correct_event_only_lost',
            'in_current_projection': cell in current_projection, 'in_remembered_projection': cell in remembered_projection,
            'single_channel_current_support': [name for name, active in channel_support.items() if cell in active],
            'excitation': e_total, 'actual_current_drive': current_drive, 'actual_replay_drive': replay_drive,
            'net_inhibition': float(current['net_inhibitory_site_current'][site]),
            'firing_threshold_including_gate': float(threshold),
            'firing_margin': float(e_total+current_drive+replay_drive-threshold),
            'excitatory_source_count': len(e_sources), 'excitatory_top_sources': e_sources[:8],
            'disinhibitory_source_count': len(di_sources), 'disinhibitory_top_sources': di_sources[:8],
            'disinhibitory_actual_current': float(current['disinhibitory_site_current'][site])})
    return {'wrong_event_only_active_cells': wrong_only_active,
            'correct_event_only_lost_cells': correct_only_lost,
            'vote_only_removal_counterfactual': recovered,
            'counterfactual_scope': 'Offline original index query only; no neurons or weights in the actual brain were changed.',
            'cells': cells}


def recall_gain_checks(full, catalog, micro_index, lost_cells):
    results = []
    for gain in (3.5, 6., 7.):
        brain, env = configure_clone(full.failure_state, full.failure_environment, 'full')
        brain.config.recall_gain = gain
        before = frozen_digest(brain)
        instrument = InstrumentedRun(brain, env, catalog, f'recall_gain_{gain:g}')
        _, info = brain._process(copy.deepcopy(full.failure_packet), learning=False, trace=True, choose_action=False)
        failed_step = instrument.records[0]['microsteps'][micro_index]
        restored = sorted(set(lost_cells) & set(failed_step['after']))
        assert frozen_digest(brain) == before
        recovered_state = brain.snapshot()
        switches = []
        for color_index in (0, 2):
            new_brain, new_env = configure_clone(recovered_state, env, 'full')
            expected = COLORS[color_index]
            new_packet = new_env.emit_tone(TONES[color_index], duration=.6)
            tracer = InstrumentedRun(new_brain, new_env, catalog, f'new_{expected}_gain_{gain:g}', expected=expected)
            new_before = frozen_digest(new_brain)
            _, switched = new_brain._process(new_packet, learning=False, trace=True, choose_action=False)
            assert frozen_digest(new_brain) == new_before
            switches.append({'new_tone': TONES[color_index], 'expected_color': expected,
                'score': completion_score(switched, catalog, expected),
                'record': summarized_record(tracer.records[0]),
                'physical_pose_unchanged': bool(np.array_equal(new_env.world.position, env.world.position)
                                              and new_env.world.heading == env.world.heading)})
        cell_margins = []
        for cell in list(lost_cells[:3])+list(lost_cells[-3:]):
            detail = failed_step['currents']
            site = cell//brain.runtime.pfc.每段
            total = detail['excitatory_current'][cell]+detail['external_current'][cell]
            threshold = detail['threshold']+detail['net_inhibitory_site_current'][site]
            cell_margins.append({'cell': cell, 'excitation': float(detail['excitatory_current'][cell]),
                'current_plus_replay': float(detail['external_current'][cell]),
                'threshold_including_gate': float(threshold), 'margin': float(total-threshold),
                'active': cell in failed_step['after']})
        results.append({'recall_gain': gain, 'same_packet_score': completion_score(info, catalog, EXPECTED),
            'restored_correct_specific_cells': restored, 'restored_count': len(restored),
            'same_microstep_top3': instrument.records[0]['lookups'][micro_index+1]['top3'],
            'same_packet_record': summarized_record(instrument.records[0]),
            'specific_cell_margins': cell_margins, 'new_cue_switches': switches,
            'all_learned_weights_index_maintenance_unchanged': True,
            'scope': 'Same preceding state and actual sensor packet; scalar recall-current gain only. New tones are emitted at the same physical pose; no new physical action is executed.'})
    return results


def main():
    start = time.perf_counter()
    config_path = HERE/'results'/'dynamic_best_tested_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['seed'] = 2027
    dependencies = ('brain.py', 'original_runtime.py', 'sensory_adapter.py', 'run_closed_loop.py', 'experience_environment.py')
    source_hashes = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in dependencies}
    brain = OriginalBrain(config)
    env = ExperienceEnvironment(seed=20260909)
    training = train(brain, env, repeats=2, hold_frames=8)
    snapshot = brain.snapshot()
    catalog = training['catalog']
    runs = {}
    for condition in CONDITIONS:
        clone, environment = configure_clone(snapshot, env, condition)
        environment.hide_beacons()
        packet = environment.emit_tone(TONES[1], duration=.6)
        instrument = InstrumentedRun(clone, environment, catalog, condition)
        for frame in range(12):
            muscles, _ = clone.observe_then_act(packet, learning=False, trace=True)
            packet = environment.step(muscles)
            clone.observe_outcome(packet, muscles, terminal=frame==11, trace=True)
        runs[condition] = instrument
        print('FOCUS '+condition+' '+json.dumps([
            (row['observation_number'], row['score']['initial_color'], row['score']['final_color']) for row in instrument.records]), flush=True)
    full = runs['full']
    failed = full.failure_record
    assert failed is not None and failed['observation_number'] == 6
    assert failed['score']['correct_initial_color'] and not failed['score']['correct_final_color']
    micro_index = next(i for i, lookup in enumerate(failed['lookups'][1:]) if lookup['catalog_color'] != EXPECTED)
    correct_time = failed['lookups'][micro_index]['time']
    wrong_time = failed['lookups'][micro_index+1]['time']
    assert wrong_time is not None
    channel_support = channel_projection_support(full.failure_state, full.failure_packet)
    cell_evidence = decisive_cells(full.brain, failed, micro_index, correct_time, wrong_time, catalog, channel_support)
    same_packet = []
    for condition in CONDITIONS:
        clone, environment = configure_clone(full.failure_state, full.failure_environment, condition)
        instrument = InstrumentedRun(clone, environment, catalog, condition)
        # No physics step or new action is performed. Only the same already
        # observed packet is processed by an identical preceding neural state.
        clone._process(copy.deepcopy(full.failure_packet), learning=False, trace=True, choose_action=False)
        same_packet.append({'condition': condition, 'record': summarized_record(instrument.records[0])})
    gain_checks = recall_gain_checks(full, catalog, micro_index, cell_evidence['correct_event_only_lost_cells'])
    # Verify instrumentation itself does not alter the observed first six steps.
    plain_brain, plain_env = configure_clone(snapshot, env, 'full')
    plain_env.hide_beacons()
    packet = plain_env.emit_tone(TONES[1], duration=.6)
    for frame in range(7):
        muscles, info = plain_brain.observe_then_act(packet, learning=False, trace=True)
        reference = full.records[frame]
        assert np.array_equal(muscles, reference['next_muscles'])
        assert completion_score(info, catalog, EXPECTED) == reference['score']
        packet = plain_env.step(muscles)
        plain_brain.observe_outcome(packet, muscles, terminal=frame==6, trace=True)
    assert source_hashes == {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in dependencies}
    report = {'status': 'complete', 'config': config, 'expected_color': EXPECTED,
        'source_sha256': source_hashes, 'birth_hash': brain.birth_hash,
        'training': training, 'first_failure_observation': failed['observation_number'],
        'first_failure_microstep_zero_based': micro_index,
        'correct_time_before_microstep': correct_time, 'wrong_time_after_microstep': wrong_time,
        'full_failure': plain(failed),
        'correct_wrong_event_difference': event_difference(full.brain, catalog, correct_time, wrong_time, failed),
        'fixed_projection_channel_support': channel_support,
        'decisive_cells': cell_evidence,
        'recall_gain_parameter_counterfactuals': gain_checks,
        'same_packet_same_neural_state_counterfactuals': same_packet,
        'physical_branch_traces': {name: [summarized_record(row) for row in run.records] for name, run in runs.items()},
        'first_failure_by_condition': {name: None if run.failure_record is None else run.failure_record['observation_number'] for name, run in runs.items()},
        'instrumentation_matches_uninstrumented_actual_actions_and_scores_frames': 7,
        'elapsed_seconds': time.perf_counter()-start,
        'boundary': 'A single reproduced failure and scoped pathway interventions; no model equations, weights, rewards, task partition or planner added.'}
    output = HERE/'results'/'focused_recall_failure.json'
    output.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2), encoding='utf-8')
    summary = [{'condition': row['condition'], 'initial': row['record']['score']['initial_color'],
                'final': row['record']['score']['final_color']} for row in same_packet]
    prose = ['# 原前额叶静音回忆漂移取证', '',
        f'种子 2027、Hebb .0001、反馈抑制 .12、目标上限 350。两轮三色真实自主共现后，绿色音调结束的第 {failed["observation_number"]} 个观测、'
        f'第 {micro_index+1} 个内部微步，把绿色时间 {correct_time} 改选为红色时间 {wrong_time}。初始索引仍指向正确绿色。', '',
        '同一故障前脑状态和完全相同感觉包，仅切既有通路开关（不执行新肌肉动作）：', '',
        '| 条件 | 初始内容 | 微步后内容 |', '|---|---|---|']
    prose += [f'| {row["condition"]} | {row["initial"]} | {row["final"]} |' for row in summary]
    prose += ['', '增大回放电流的同包参数对照（全部原连接、身体状态和当前输入相同）：', '',
              '| recall_gain | 恢复绿色特异细胞 | 微步后内容 | 随后红音/蓝音能否切换 |', '|---:|---:|---|---|']
    for row in gain_checks:
        switched = all(item['score']['correct_final_color'] for item in row['new_cue_switches'])
        prose.append(f'| {row["recall_gain"]:g} | {row["restored_count"]}/{len(cell_evidence["correct_event_only_lost_cells"])} | '
                     f'{row["same_packet_score"]["final_color"]} | {switched} |')
    prose += ['', '具体原因：绿色特异细胞在静音时只收到回放 3.5，而局部总放电门槛为 4.25146 或 5.33146；'
              '兴奋回送约 .3271 仍补不够。当前深度投影的 1927/1928 则仍跨门槛，导致红色事件多两票。'
              '这次漂移不是仅靠抑低 E 就能消除；提高回放电流的对照直接检验了正确特异细胞的门槛缺口。']
    prose += ['', '完整 JSON 保存每次原索引的前三票、正误事件的真实视觉/运动细胞、当前与回忆投影、故障前念头、原兴奋/去抑来源连接和逐细胞放电门槛。',
              '计票反事实仅用于离线解释索引，不改实际脑活动；单通道投影是原固定网络的诊断读取，不将颜色标签输入脑。原方程和源文件未修改。',
              '该结论只针对这次可复现漂移；长期稳定性和其它种子的失败仍需分别检查。']
    output.with_suffix('.md').write_text('\n'.join(prose)+'\n', encoding='utf-8')
    print('FOCUS_COMPLETE '+json.dumps({'failure_microstep': micro_index, 'correct': correct_time, 'wrong': wrong_time,
        'same_packet': summary, 'first_failure_by_condition': report['first_failure_by_condition'],
        'gain_checks': [{'gain': row['recall_gain'], 'restored': row['restored_count'],
                         'final': row['same_packet_score']['final_color'],
                         'switches_correct': [item['score']['correct_final_color'] for item in row['new_cue_switches']]}
                        for row in gain_checks],
        'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
