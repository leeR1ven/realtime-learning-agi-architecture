"""Artificial-neuron fixture for ORIGINAL PFC conjunction and repeat stability.

This is not a PCM/RGB experiment, autonomous discovery, a language rule learner,
or evidence of AGI. Named Boolean ensembles are externally clamped observations.
Only the original class's learning method creates or strengthens connections.
Every experimental parameter is fixed before the repetition/size/strength grid.
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
from 前额叶区_prefrontal import 前额叶联想区


WIDTH = 320
GROUP_NAMES = ('A', 'B', 'D', 'C', 'E', 'F', 'G', 'H')
REPETITIONS = (1, 3, 10, 30)
SIZES = (16, 24, 32)
STRENGTHS = (.75, 1., 1.25)
CONDITIONS = ('full', 'no_excitatory', 'no_disinhibition')
PRESETS = {
    'original_default': {'increment': 1., 'firing_threshold': 2., 'target_high': 24},
    # N=24,S=1: one input E=1.32<threshold2.736; two E=2.64>=2.472.
    'fixed_conjunction': {'increment': .055, 'firing_threshold': 2., 'target_high': 96},
    'lower_fixed_rate': {'increment': .0055, 'firing_threshold': 2., 'target_high': 96},
    'higher_fixed_threshold': {'increment': .055, 'firing_threshold': 6., 'target_high': 96},
    # The existing connection manager already implements this uniform decay.
    # A14-step recipe gives asymptotic AB->C weight d*.96**12/(1-.96**14).
    # This sets one predetermined steady window, not a per-repetition fit.
    'existing_uniform_decay': {'increment': .042, 'firing_threshold': 2., 'target_high': 96,
        'decay': .96, 'warning': 0, 'interval': 1, 'floor': .00005},
}


def groups(size):
    output = {}
    for index, name in enumerate(GROUP_NAMES):
        a = np.zeros(WIDTH, bool)
        a[index*40:index*40+size] = True
        output[name] = a
    output['zero'] = np.zeros(WIDTH, bool)
    output['AB'] = output['A'] | output['B']
    output['AD'] = output['A'] | output['D']
    output['BD'] = output['B'] | output['D']
    output['AH'] = output['A'] | output['H']
    return output


def recipe(double_chain=True):
    values = ('AB', 'C', 'F', 'zero', 'AD', 'E', 'G', 'zero')
    return values + (('C', 'F', 'zero', 'E', 'G', 'zero') if double_chain else ())


def birth(preset, strength):
    cfg = PRESETS[preset]
    pfc = 前额叶联想区(WIDTH, 门槛=cfg['firing_threshold'], 目标上限=cfg['target_high'], 强度初值=strength)
    for table in (pfc.联想, pfc.去抑):
        table.强化量 = cfg['increment']
        if 'decay' in cfg:
            table.衰减率 = cfg['decay']
            table.警戒线 = cfg['warning']
            table.衰减间隔 = cfg['interval']
            table.消失下限 = cfg['floor']
            table.上次衰减维护 = -cfg['interval']
    assert pfc.联想.条数() == pfc.去抑.条数() == 0
    return pfc


def learn_cycle(pfc, patterns, double_chain=True):
    previous = patterns['zero']
    for name in recipe(double_chain):
        following = patterns[name]
        pfc.学习(previous, following)
        previous = following


def table_digest(pfc):
    return hashlib.sha256(json.dumps([pfc.联想.表, pfc.去抑.表], sort_keys=True).encode()).hexdigest()


class CellArchive:
    def __init__(self):
        self.excitation = []
        self.net_inhibition = []
        self.threshold = []
        self.margin = []
        self.previous = []
        self.activity = []

    def microstep(self, pfc, previous, patterns):
        strength_before = pfc.强度
        excitation = pfc.驱动(previous)
        field = pfc.门场(previous)
        net = field[np.arange(WIDTH)//pfc.每段]
        threshold = pfc.门槛 + net
        margin = excitation-threshold
        output = pfc.微步(previous)
        assert np.array_equal(output, margin >= 0)
        index = len(self.excitation)
        for name, value in (('excitation', excitation), ('net_inhibition', net),
                            ('threshold', threshold), ('margin', margin),
                            ('previous', previous), ('activity', output)):
            getattr(self, name).append(value.copy())
        stats = {}
        for name in ('C', 'E', 'F', 'G'):
            chosen = patterns[name]
            stats[name] = {'active': int((output & chosen).sum()),
                'E_min': float(excitation[chosen].min()), 'E_max': float(excitation[chosen].max()),
                'net_I_min': float(net[chosen].min()), 'net_I_max': float(net[chosen].max()),
                'margin_min': float(margin[chosen].min()), 'margin_max': float(margin[chosen].max())}
        return output.copy(), {'array_row': index, 'active_count': int(output.sum()),
            'strength_before': strength_before, 'strength_after': pfc.强度, 'groups': stats}

    def save(self, path):
        np.savez_compressed(path, **{name: np.stack(getattr(self, name)) for name in
            ('excitation', 'net_inhibition', 'threshold', 'margin', 'previous', 'activity')})
        with np.load(path, allow_pickle=False) as stored:
            assert stored['excitation'].shape == (len(self.excitation), WIDTH)
            assert np.array_equal(stored['activity'], stored['margin'] >= 0)


def probes():
    return (('A_only', 'A', ('zero',)), ('B_only', 'B', ('zero',)),
            ('D_only', 'D', ('zero',)), ('H_unseen', 'H', ('zero',)),
            ('A_B', 'AB', ('C',)), ('A_D', 'AD', ('E',)),
            ('B_D_wrong', 'BD', ('zero',)), ('A_H_wrong', 'AH', ('zero',)),
            ('C_chain_entry', 'C', ('F',)), ('E_chain_entry', 'E', ('G',)),
            ('A_B_then_F', 'AB', ('C', 'F')), ('A_D_then_G', 'AD', ('E', 'G')))


def evaluate(pfc, patterns, archive):
    reports = {}
    conjunction_names = {name for name, _, _ in probes()[:8]}
    serial_names = {name for name, _, _ in probes()[8:]}
    for condition in CONDITIONS:
        clone = copy.deepcopy(pfc)
        if condition == 'no_excitatory':
            clone.联想.表 = {}
        elif condition == 'no_disinhibition':
            clone.去抑抵消 = 0.
        before = table_digest(clone)
        rows = []
        for name, initial, expected in probes():
            # Independent, explicit clamps avoid order-dependent test strength.
            clone.强度 = pfc.强度
            clone.念头[:] = False
            previous = patterns[initial].copy()
            stages = []
            for next_name in expected:
                previous, detail = archive.microstep(clone, previous, patterns)
                detail['expected_pattern'] = next_name
                detail['exact_expected_activity'] = bool(np.array_equal(previous, patterns[next_name]))
                stages.append(detail)
            rows.append({'probe': name, 'initial_clamp': initial, 'stages': stages,
                         'passed': all(row['exact_expected_activity'] for row in stages)})
        assert before == table_digest(clone)
        reports[condition] = {'probes': rows,
            'conjunction_pass': all(row['passed'] for row in rows if row['probe'] in conjunction_names),
            'serial_pass': all(row['passed'] for row in rows if row['probe'] in serial_names),
            'all_pass': all(row['passed'] for row in rows),
            'failed_probes': [row['probe'] for row in rows if not row['passed']],
            'connections_unchanged_during_probes': True}
    return reports


def learned_summary(pfc, patterns):
    def one(a, b):
        source = int(np.flatnonzero(patterns[a])[0])
        target = int(np.flatnonzero(patterns[b])[0])
        return {'source': source, 'destination': target, 'weight': pfc.联想.查(source).get(target, 0.),
                'disinhibitory_site': target//pfc.每段,
                'disinhibitory_weight': pfc.去抑.查(source).get(target//pfc.每段, 0.)}
    return {'E_edges': pfc.联想.条数(), 'DI_edges': pfc.去抑.条数(),
        'E_maintenance_calls': pfc.联想.维护次数, 'DI_maintenance_calls': pfc.去抑.维护次数,
        'last_E_decay_call': pfc.联想.上次衰减维护, 'last_DI_decay_call': pfc.去抑.上次衰减维护,
        'representative_edges': {a+'->'+b: one(a, b) for a, b in (('A', 'C'), ('B', 'C'), ('A', 'E'), ('D', 'E'), ('C', 'F'), ('E', 'G'))}}


def main():
    started = time.perf_counter()
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in ('前额叶区_prefrontal.py', '权重连接管理_weight_manager.py')}
    archive = CellArchive()
    cases = []
    for preset in PRESETS:
        for size in SIZES:
            patterns = groups(size)
            for strength in STRENGTHS:
                pfc = birth(preset, strength)
                for repetition in range(1, max(REPETITIONS)+1):
                    learn_cycle(pfc, patterns)
                    if repetition in REPETITIONS:
                        cases.append({'preset': preset, 'size': size, 'strength': strength,
                            'repetitions': repetition, 'learning': learned_summary(pfc, patterns),
                            'conditions': evaluate(pfc, patterns, archive)})
        central = [row for row in cases if row['preset'] == preset and row['size'] == 24 and row['strength'] == 1.]
        print('JOINT '+preset+' '+json.dumps([{'repetitions': row['repetitions'],
            'full': row['conditions']['full']['all_pass'],
            'failures': row['conditions']['full']['failed_probes']} for row in central]), flush=True)
    patterns = groups(24)
    equal = birth('fixed_conjunction', 1.)
    learn_cycle(equal, patterns, double_chain=False)
    equal_control = {'recipe': recipe(False), 'learning': learned_summary(equal, patterns),
                     'conditions': evaluate(equal, patterns, archive)}
    retention = []
    for preset, repeats in (('fixed_conjunction', 1), ('existing_uniform_decay', 30)):
        pfc = birth(preset, 1.)
        for _ in range(repeats):
            learn_cycle(pfc, patterns)
        for blank_steps in (0, 1, 3, 10, 30):
            clone = copy.deepcopy(pfc)
            for _ in range(blank_steps):
                clone.学习(patterns['zero'], patterns['zero'])
            retention.append({'preset': preset, 'training_repetitions': repeats,
                'subsequent_blank_observations': blank_steps,
                'learning': learned_summary(clone, patterns),
                'conditions': evaluate(clone, patterns, archive)})
    assert sources == {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    folder = HERE/'results'
    folder.mkdir(exist_ok=True)
    arrays_path = folder/'pfc_joint_mechanism_cells.npz'
    archive.save(arrays_path)
    report = {'status': 'complete', 'scope': 'Externally clamped artificial neuron mechanism fixture; no actual PCM, RGB, muscles, reward, navigation or autonomous knowledge discovery.',
        'source_sha256': sources, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'width': WIDTH, 'group_placement': 'Each named ensemble occupies the first N cells of its own original40-cell inhibitory site. This idealized placement avoids local target feedback on the preceding input; real mixed sensory ensembles need separate validation.',
        'group_order_by_site': GROUP_NAMES, 'presets': PRESETS, 'repetitions': REPETITIONS,
        'sizes': SIZES, 'strengths': STRENGTHS, 'recipe': recipe(),
        'training_semantics': 'Every adjacent pair of the listed timeline is passed to original PFC.学习. A+B->C and A+D->E each occur once per recipe; C->F and E->G twice. Blank frames avoid an accidental terminal-to-next-example transition; no edge is assigned its expected answer.',
        'parameters_not_overridden': {'site_size': 40, 'target_low': 4, 'strength_min': .15, 'strength_max': 15.,
            'strength_adjust_rate': .15, 'rest_inhibition': 1., 'feedback_inhibition': .12, 'DI_cancellation': .2,
            'default_connection_warning': 100000, 'default_decay_interval': 200, 'default_decay': .9, 'default_floor': .05},
        'cases': cases, 'equal_exposure_serial_control': equal_control, 'blank_experience_retention': retention,
        'per_cell_archive': {'path': str(arrays_path), 'rows': len(archive.excitation), 'shape': [len(archive.excitation), WIDTH],
            'fields': ['excitation', 'net_inhibition', 'threshold', 'margin', 'previous', 'activity'],
            'dtype_currents': 'float64', 'dtype_activities': 'bool', 'external_current': 'zero at every tested microstep',
            'lookup': 'Each probe stage array_row selects the exact per-cell row. margin=excitation-(original firing threshold+net inhibition).'},
        'analytic_limits': {'unconditional_growth': 'Below warning edge count, each frequently repeated AB->C edge is repetitions*increment. Finite fixed thresholds cannot keep a single input below threshold for arbitrarily many repeats.',
            'fixed_fixture_N24_R30': {'single_input_E': 24*.055*30,
                'minimum_strength_for_single_input_silence_before_DI_clipping': 1.2*24*.055*30-2,
                'original_maximum_strength': 15.},
            'existing_uniform_decay_AB_weight_limit': .042*.96**12/(1-.96**14),
            'existing_uniform_decay_AD_weight_limit': .042*.96**8/(1-.96**14),
            'decay_half_life_in_learning_calls': float(np.log(.5)/np.log(.96)),
            'boundary': 'Decay limits hold only for the declared stationary14-step repeated schedule. Relative exposure frequency and forgetting remain consequential.'},
        'elapsed_seconds': time.perf_counter()-started,
        'production_architecture_changed': False}
    output = folder/'pfc_joint_mechanism.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# 原前额叶多簇合取与重复学习夹具', '',
        '**这是人工神经元机制单测，不是真实感觉、自主发现、自然语言逻辑或AGI成绩。**', '',
        '直接实例化原前额叶联想区。A/B/D各为输入簇，C/E为中间簇，F/G为后继簇。'
        '每个簇放在独立的原40细胞抑制位点。所有连接从空表开始，只经原学习方法建立；没有手写答案权重。', '',
        '连续人工活动序列为 AB→C→F→空白→AD→E→G→空白→C→F→空白→E→G→空白。'
        '因此每轮合取经历一次，单簇后继经历两次。额外暴露是明示的人工教学条件。', '',
        'N=24、全局强度1时，固定强化.055：单簇E=1.32，门槛2.736；双簇E=2.64，門槛2.472。'
        '双簇跨门槛且单簇不跨门槛。去掉去抑制后门槛3使该双簇不能激活。', '',
        '| 固定参数方案 | 重复1次 | 3次 | 10次 | 30次 |', '|---|---|---|---|---|']
    for preset in PRESETS:
        row = [item for item in cases if item['preset'] == preset and item['size'] == 24 and item['strength'] == 1.]
        lines.append('| '+preset+' | '+' | '.join('合取及串联全过' if item['conditions']['full']['all_pass'] else '失败' for item in row)+' |')
    lines += ['', '相同次数C→F暴露的控制中，合取可以成立，但C单簇无法继续驱动F；在相同簇大小与相同连接增量下，'
        '它与被要求保持不激活的A单簇具有相同总电流。双倍后继暴露解决的是这一夹具电流条件，不能声称模型自己理解了子目标。', '',
        '完整JSON保留N=16/24/32、初始抑制强度.75/1/1.25、全部重复次数和E/去抑制消融。'
        '逐细胞电流、净抑制、放电门槛、边际与实际活动位于配套NPZ，可用allow_pickle=False读取。', '',
        '重复后的风险是边权按经历次数累加；本夹具最多6144条兴奋边，未触发原默认100000条警戒线的统一衰减。'
        '固定低学习率或固定较高门槛只能改变合取出现与退化的次数窗口。', '',
        '原连接表已有的无条件统一衰减能在这套固定频率配方下形成有界权重，但会降低未被反复激活的旧联系。'
        '空白经历对照单独保留如下：', '',
        '| 方案 | 后续空白学习时刻 | 合取及串联 | E边数 |', '|---|---:|---|---:|']
    for row in retention:
        lines.append(f'| {row["preset"]} | {row["subsequent_blank_observations"]} | '
            f'{row["conditions"]["full"]["all_pass"]} | {row["learning"]["E_edges"]} |')
    lines += ['', '结论边界：原式具备有限参数区间内的阈值合取与串联能力；它不是对任意重复次数、任意簇大小都稳定的逻辑机制。'
        '直接改变现有全局衰减参数会引入选择性与旧知识保持的权衡，不能仅凭本夹具通过就用于正式生活模型。'
        '若后续真实经历仍需修正，优先检验同一原连接表的通用突触总量稳态，并明确它是机制改动；本脚本没有实施该改动。']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print('JOINT_COMPLETE '+json.dumps({'cases': len(cases), 'microstep_rows': len(archive.excitation),
                                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
