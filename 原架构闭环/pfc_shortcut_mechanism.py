"""Original-PFC shortcut learning: explicit temporal-overlap fixture only.

All training activity below is artificially clamped and openly specified. The
original learning method creates every connection. A pure-A recall probe has no
external current and no plasticity. No production brain or earlier data changes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from 前额叶区 import 前额叶联想区


WIDTH, SIZE, INCREMENT = 160, 24, .12
REPETITIONS = (1, 3, 10)
SEQUENCES = {
    'strict_separation': ('A', 'B', 'C', 'zero'),
    'A_artificially_retained_into_previous_B': ('A', 'AB', 'C', 'zero'),
    'A_returns_only_when_C_arrives': ('A', 'B', 'AC', 'zero'),
}


def patterns():
    result = {}
    for i, name in enumerate(('A', 'B', 'C', 'U')):
        array = np.zeros(WIDTH, bool)
        array[i*40:i*40+SIZE] = True
        result[name] = array
    result['AB'] = result['A'] | result['B']
    result['AC'] = result['A'] | result['C']
    result['zero'] = np.zeros(WIDTH, bool)
    return result


def birth():
    pfc = 前额叶联想区(WIDTH, 目标上限=96)
    pfc.联想.强化量 = pfc.去抑.强化量 = INCREMENT
    assert pfc.联想.条数() == pfc.去抑.条数() == 0
    return pfc


def digest(pfc):
    return hashlib.sha256(json.dumps([pfc.联想.表, pfc.去抑.表], sort_keys=True).encode()).hexdigest()


def edge_group(pfc, features, source, target):
    sources = np.flatnonzero(features[source])
    targets = np.flatnonzero(features[target])
    weights = [pfc.联想.查(int(a)).get(int(b), 0.) for a in sources for b in targets]
    a, b = int(sources[0]), int(targets[0])
    return {'possible_edges': len(weights), 'present_edges': int(np.count_nonzero(weights)),
        'min_weight': float(min(weights)), 'max_weight': float(max(weights)),
        'representative': {'source_cell': a, 'target_cell': b,
            'E_weight': pfc.联想.查(a).get(b, 0.),
            'DI_target_site': b//pfc.每段, 'DI_weight': pfc.去抑.查(a).get(b//pfc.每段, 0.)}}


def recall(pfc, features, steps=6):
    clone = copy.deepcopy(pfc)
    before = digest(clone)
    previous = features['A'].copy()
    trajectory = []
    for microstep in range(steps):
        excitation = clone.驱动(previous)
        net = clone.门场(previous)[np.arange(WIDTH)//clone.每段]
        threshold = clone.门槛 + net
        strength = clone.强度
        activity = clone.微步(previous)
        assert np.array_equal(activity, excitation >= threshold)
        cells = []
        for name in ('A', 'B', 'C'):
            cell = int(np.flatnonzero(features[name])[0])
            incoming = [{'source_cell': int(a), 'weight': clone.联想.查(int(a)).get(cell, 0.)}
                        for a in np.flatnonzero(previous) if clone.联想.查(int(a)).get(cell, 0.)]
            assert np.isclose(sum(row['weight'] for row in incoming), excitation[cell])
            cells.append({'group': name, 'cell': cell, 'E': float(excitation[cell]),
                'net_I': float(net[cell]), 'threshold': float(threshold[cell]),
                'margin': float(excitation[cell]-threshold[cell]), 'active': bool(activity[cell]),
                'incoming_excitatory_sources': incoming})
        trajectory.append({'microstep_one_based': microstep+1,
            'previous_cells': np.flatnonzero(previous).tolist(),
            'active_cells': np.flatnonzero(activity).tolist(),
            'active_groups': [name for name in ('A', 'B', 'C', 'U') if np.any(activity & features[name])],
            'group_counts': {name: int((activity & features[name]).sum()) for name in ('A', 'B', 'C', 'U')},
            'strength_before': strength, 'strength_after': clone.强度,
            'representative_cells': cells,
            'all_cell_E': excitation.tolist(), 'all_cell_net_I': net.tolist(),
            'all_cell_threshold': threshold.tolist(), 'all_cell_margin': (excitation-threshold).tolist()})
        previous = activity.copy()
    assert digest(clone) == before
    return {'initial_clamp': 'A only', 'external_current': 'none for all microsteps',
        'learning_during_probe': False, 'learned_connections_unchanged': True,
        'first_C_microstep': next((row['microstep_one_based'] for row in trajectory if row['group_counts']['C']), None),
        'trajectory': trajectory}


def append_gradual_shortcut():
    """Add one same-parameter curriculum without regenerating old nine cases."""
    path = HERE/'results'/'pfc_shortcut_mechanism.json'
    report = json.loads(path.read_text(encoding='utf8'))
    old_cases_hash = hashlib.sha256(json.dumps(report['cases'], sort_keys=True).encode()).hexdigest()
    production_paths = (ROOT/'前额叶区.py', ROOT/'权重连接管理.py', HERE/'brain.py', HERE/'original_runtime.py')
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in production_paths}
    features = patterns()
    gamma = .02
    # With24 sources and this gamma, six exposures cross the original firing
    # condition: 24*.02*6=2.88 >= 2+1-.2*2.88=2.424. This is a calculation,
    # not a parameter sweep. Gamma stays unchanged across both training phases.
    initial_exposures = 6
    pfc = birth()
    pfc.联想.强化量 = pfc.去抑.强化量 = gamma

    def teach(model, sequence):
        previous = features['zero']
        for pattern in sequence:
            following = features[pattern]
            model.学习(previous, following)
            previous = following

    def measured(model, added):
        probe = recall(model, features)
        c = next(row for row in probe['trajectory'][0]['representative_cells'] if row['group']=='C')
        return {'strict_pretraining_exposures': initial_exposures,
            'additional_overlap_exposures': added,
            'learning_calls': model.联想.维护次数,
            'gamma_E': model.联想.强化量, 'gamma_DI': model.去抑.强化量,
            'edges': {a+'->'+b: edge_group(model, features, a, b)
                      for a,b in (('A','A'),('A','B'),('B','C'),('A','C'))},
            'first_microstep_C_cell': c, 'probe': probe}

    for _ in range(initial_exposures):
        teach(pfc, SEQUENCES['strict_separation'])
    strict_control = copy.deepcopy(pfc)
    milestones = [measured(pfc, 0)]
    assert milestones[0]['probe']['first_C_microstep'] == 2
    assert milestones[0]['edges']['A->C']['present_edges'] == 0
    for added in range(1, 11):
        teach(pfc, SEQUENCES['A_artificially_retained_into_previous_B'])
        teach(strict_control, SEQUENCES['strict_separation'])
        if added in (1, 3, 6, 10):
            milestones.append(measured(pfc, added))
    control = measured(strict_control, 10)
    control['additional_overlap_exposures'] = 0
    control['additional_strict_exposures'] = 10
    assert control['edges']['A->C']['present_edges'] == 0
    assert control['probe']['first_C_microstep'] == 2
    assert [row['probe']['first_C_microstep'] for row in milestones] == [2, 2, 2, 1, 1]
    assert all(row['gamma_E'] == gamma and row['gamma_DI'] == gamma for row in milestones)
    report['gradual_same_network_shortcut'] = {
        'scope': 'One artificial neural network with fixed gamma, first learning a strict chain then receiving openly supplied A-retention examples. Recall probes use clones without changing its learned continuation.',
        'gamma_E_and_DI_fixed': gamma,
        'parameter_selection': 'One analytical choice, no parameter sweep: N24,gamma.02,threshold2,restI1,DI.2 require at least six equal exposures for one source ensemble to drive its learned successor.',
        'strict_pretraining': {'sequence': SEQUENCES['strict_separation'], 'exposures': initial_exposures},
        'continued_teaching': {'sequence': SEQUENCES['A_artificially_retained_into_previous_B'],
            'exposures': 10, 'A_persistence': 'artificially clamped by the experimenter, not an emergent retention claim'},
        'milestones': milestones, 'continued_strict_control': control,
        'original_nine_cases_sha256_unchanged': old_cases_hash,
        'original_nine_cases_rerun': False,
        'production_source_sha256_at_run': source_hashes,
        'script_sha256_at_addition': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'conclusion': 'In the same parameter regime and continued learned network, direct A->C initially does not exist, then exists below firing threshold, and finally gives C a one-microstep response after six added overlap exposures. Continuing only strict examples keeps a two-step response.',
        'boundary': 'Shows how a shortcut can become effective gradually when the required temporal overlap is provided. Does not establish that the real sensory brain naturally retains A or learned sound->route automation.'}
    assert hashlib.sha256(json.dumps(report['cases'], sort_keys=True).encode()).hexdigest() == old_cases_hash
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    reread = json.loads(path.read_text(encoding='utf8'))
    assert hashlib.sha256(json.dumps(reread['cases'], sort_keys=True).encode()).hexdigest() == old_cases_hash
    marker = '\n## 同网络同学习率的逐渐缩短对照\n'
    original_text = path.with_suffix('.md').read_text(encoding='utf8').split(marker)[0]
    lines = [marker.rstrip(), '',
        '这项补充使用同一网络，强化量始终为.02。先人工提供6次严格A→B→C→空白，形成纯A启动的两步链；'
        '随后继续人工提供A→(A+B)→C→空白，明确把A持留到B时刻。检查点只克隆读取，原网络继续累积学习，未重置已学连接。', '',
        '参数由24×.02×6=2.88、门槛2+1−.2×2.88=2.424直接选定，只试这一组，没有扫描后选优。', '',
        '| 已学严格链后，新增重叠次数 | A→C每边权重 | 纯A首微步给C的E | C放电门槛 | 边际 | C首次激活微步 |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in milestones:
        c = row['first_microstep_C_cell']
        lines.append(f'| {row["additional_overlap_exposures"]} | {row["edges"]["A->C"]["representative"]["E_weight"]:.2f} | '
            f'{c["E"]:.3f} | {c["threshold"]:.3f} | {c["margin"]:+.3f} | {row["probe"]["first_C_microstep"]} |')
    lines += ['',
        '新增1次、3次时，A→C连接已经存在，但电流尚不足，所以仍须经B，C在第2步出现。'
        '新增6次时，A→C跨门槛，C提前到第1步；新增10次仍是第1步。'
        '同一个已学两步链分支若继续10次严格A→B→C，A→C仍为0，C仍在第2步出现。', '',
        '这展示了同一学习率下“没有快捷边→已有弱快捷边→快捷边足以直接唤起”的连续过程。'
        '额外重叠教学与A的人为持留都已明示；尚未验证真实网络会自主提供这一条件。原9组数据未重跑或改值，正式模型与原AND数据未改。']
    path.with_suffix('.md').write_text(original_text+'\n'+'\n'.join(lines)+'\n', encoding='utf8')
    print('GRADUAL_SHORTCUT '+json.dumps({'gamma': gamma, 'old_nine_cases_unchanged': True,
        'milestones': [{'overlap_exposures': row['additional_overlap_exposures'],
            'AC_weight': row['edges']['A->C']['representative']['E_weight'],
            'E': row['first_microstep_C_cell']['E'], 'threshold': row['first_microstep_C_cell']['threshold'],
            'first_C_microstep': row['probe']['first_C_microstep']} for row in milestones],
        'strict_control_first_C': control['probe']['first_C_microstep']}), flush=True)


def main():
    started = time.perf_counter()
    files = (ROOT/'前额叶区.py', ROOT/'权重连接管理.py', HERE/'brain.py', HERE/'original_runtime.py')
    inspected_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    features = patterns()
    cases = []
    for name, sequence in SEQUENCES.items():
        pfc = birth()
        for repetition in range(1, max(REPETITIONS)+1):
            previous = features['zero']
            for pattern in sequence:
                following = features[pattern]
                pfc.学习(previous, following)
                previous = following
            if repetition in REPETITIONS:
                edge_summary = {a+'->'+b: edge_group(pfc, features, a, b)
                                for a in ('A', 'B', 'C') for b in ('A', 'B', 'C')}
                shortcut_expected = name == 'A_artificially_retained_into_previous_B'
                assert bool(edge_summary['A->C']['present_edges']) == shortcut_expected
                assert np.isclose(edge_summary['A->C']['representative']['E_weight'],
                                  repetition*INCREMENT if shortcut_expected else 0.)
                case = {'sequence': name, 'clamped_pattern_order': sequence, 'repetitions': repetition,
                    'learning_calls': pfc.联想.维护次数, 'E_edge_count': pfc.联想.条数(),
                    'DI_edge_count': pfc.去抑.条数(), 'edges': edge_summary,
                    'probe': recall(pfc, features)}
                cases.append(case)
                print('SHORTCUT '+json.dumps({'sequence': name, 'repetitions': repetition,
                    'A_C_weight': edge_summary['A->C']['representative']['E_weight'],
                    'first_C_microstep': case['probe']['first_C_microstep'],
                    'pure_A_trajectory': [r['active_groups'] for r in case['probe']['trajectory']]}), flush=True)
    # Parent may independently edit production wrappers; preserve the exact
    # version inspected instead of asserting they were frozen by this subtask.
    inspected_hashes_after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for path in files[:2]:
        assert inspected_hashes[str(path)] == inspected_hashes_after[str(path)]
    report = {'status': 'complete', 'scope': 'Artificial-neuron temporal-overlap fixture. Retention ofA atB is explicitly supplied by the experimenter, not claimed to emerge naturally.',
        'parameters': {'width': WIDTH, 'group_size': SIZE, 'site_size': 40, 'increment_E_and_DI': INCREMENT,
            'firing_threshold': 2., 'initial_global_strength': 1., 'target_high': 96,
            'rest_inhibition': 1., 'feedback_inhibition': .12, 'DI_cancellation': .2,
            'other_parameters': 'original class and connection manager defaults'},
        'sequences': SEQUENCES, 'repetitions': REPETITIONS, 'cases': cases,
        'source_sha256_at_inspection': inspected_hashes, 'source_sha256_after_fixture': inspected_hashes_after,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'learning_rule_conclusion': 'A->C grows exactly when a cell inA is in the previous learned activity and a cell inC is in the following learned activity. Repeating disjoint A,B,C does not perform a transitive closure of A->B->C. Same-time A+C withoutA in the preceding frame does not by itself growA->C.',
        'current_production_review': {
            'original_rule': {'file': str(ROOT/'前额叶区.py'), 'learning_line': 203, 'E_write_line': 215, 'DI_write_line': 222},
            'actual_runtime_transition': {'file': str(HERE/'original_runtime.py'), 'record_line': 180,
                'learning_line': 190, 'copy_previous_line': 191,
                'meaning': 'Whenlearning=True, original PFC learns previous_actual_pfc->the current supplied final thought; then replaces previous_actual_pfc with a copy. No separate facts/rules/shortcut store.'},
            'brain_observation': {'file': str(HERE/'brain.py'), 'microstep_loop_line': 184,
                'current_plus_recalled_current_lines': [186,187], 'record_line': 217,
                'meaning': 'Current and recalled fixed projections jointly drive original microsteps. The final thought is recorded after the loop; internal microstep transitions are not each sent to PFC learning. index_pfc_bool=external is for the same hippocampal index, not the source of recurrent PFC Hebb updates.'},
            'possible_condition': 'A retained in one recorded final thought andC present in the next recorded final thought can growA->C, including overlap arising from actual current/recalled activity. Overlap confined to unrecorded intermediate microsteps is insufficient for this wrapper.',
            'verification_status': 'Possible from the source rule; not demonstrated by this fixture to arise spontaneously from the real sensory brain. The current real environment has no validated route-memory task, so no learned sound->route shortcut is established.',
            'automaticity_boundary': 'A direct learned shortcut and multistep recall are both legitimate automatic association. There is no requirement that every learned response permanently traverses every originally observed intermediate. Neither establishes AGI.'},
        'production_files_changed_by_this_script': False,
        'elapsed_seconds': time.perf_counter()-started}
    path = HERE/'results'/'pfc_shortcut_mechanism.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# 原前额叶快捷联想连接核查', '',
        '原规则允许形成A→C快捷连接，但需要A在某次已学习的前一时刻仍活跃、C在相邻后一时刻活跃。'
        '严格分开的A→B→C反复出现，只会加强A→B和B→C，不会自动把两段边合成为A→C。', '',
        '**这是人工神经活动夹具。下表中A的持续是人为给定，没有声称真实感觉网络自然产生了这种持续。所有边从空表开始，仅由原学习方法建立。**', '',
        '每簇24细胞，固定强化.12，原放电门槛2、初始全局抑制1。纯A提示后不再给任何外部电流，依次运行6个原微步，期间不学习。', '',
        '| 人工经历顺序 | 重复次数 | A→B权重 | B→C权重 | A→C权重 | C首次激活微步 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in cases:
        w = lambda edge: row['edges'][edge]['representative']['E_weight']
        lines.append(f'| {" → ".join(row["clamped_pattern_order"])} | {row["repetitions"]} | '
                     f'{w("A->B"):.2f} | {w("B->C"):.2f} | {w("A->C"):.2f} | {row["probe"]["first_C_microstep"]} |')
    lines += ['',
        '具体数值：一次A→(A+B)→C经历后，A→C的每条连接权重.12。纯A的24个源给C细胞80的总兴奋为2.88，'
        '净抑制为1−.2×2.88=.424，总门槛2.424，余量+.456，因此C在第1微步激活。'
        '严格分离经历中A→C仍为0，C首步兴奋0，需先唤起B后到第2步才激活。两个序列在1、3、10次曝光均保持这个连接有无的区别。', '',
        '第三组A→B→(A+C)控制说明：A和C只在后一时刻同现，不等于A在前一时刻存在，所以A→C仍为0。'
        '原式学习的是相邻时刻的有向关联，不是只看同刻共现，也不是把图中的任意多步路径直接缩成边。', '',
        '重叠教学还会产生A→A或B→A等真实共现边。它们在部分参数/重复次数下让活动持续或循环；JSON保留了每个微步的完整活动和电流。'
        '这些是人工教学时序的后果，不能称为真实脑自动发现了持续状态，也不能仅因出现自关联就判断模型异常。', '',
        f'当前闭环在[brain.py:184]({(HERE/"brain.py").as_posix()}:184)先做内部微步，把当前感觉与回忆电流共同送入原PFC；'
        f'随后在[brain.py:217]({(HERE/"brain.py").as_posix()}:217)记录最终thought。'
        f'[original_runtime.py:190]({(HERE/"original_runtime.py").as_posix()}:190)只学习上一次记录的实际thought→本次实际thought。', '',
        '因此，如果A在一次记录的最终活动里仍存在，随后一次记录含C，现有实现就可能学出A→C；'
        '若这种A/C接续只发生在没有逐一记录的内部微步，当前写入方式不会直接学到该快捷边。'
        '这项条件来自源代码核查，尚未验证真实传感闭环会自然形成声音→路线快捷连接；目前真实环境没有已验证的路线记忆任务。', '',
        '声音经中间记忆到路线、熟练后更直接唤起路线，都是可接受的联想自动化方向，不必永久强制中间节点。'
        '本次结果验证的是原式允许快捷边形成的时序条件，不代表导航自动化、通用知识学习或AGI已经实现。正式代码与旧合取数据均未改。']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    append_gradual_shortcut()
    print('SHORTCUT_COMPLETE '+json.dumps({'cases': len(cases), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    if '--append-gradual' in sys.argv:
        append_gradual_shortcut()
    else:
        main()
