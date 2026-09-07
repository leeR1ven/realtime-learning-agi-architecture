"""One pre-calculated soft-plasticity candidate; artificial neural codes only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np

from pfc_capacity_experiment import HERE, ROOT, sha, fixture, birth, learn_sequence, assess, probe
from weight_dependent_plasticity import SoftBoundConnectionTable


def model(scale, condition):
    rate = {'current': .0001, 'learnable_fast_forgetting_control': .012,
            'soft_slow_weight_dependent': .005}[condition]
    pfc = birth(scale, rate)
    if condition == 'soft_slow_weight_dependent':
        for name in ('联想', '去抑'):
            old = getattr(pfc, name)
            old.衰减率 = .9995
            table = SoftBoundConnectionTable.from_original(old, .15/scale, 'weight_dependent')
            setattr(pfc, name, table)
    return pfc


def measure(pfc, groups, scale):
    a = assess(pfc, groups, scale)
    decisions = []
    for row in a['rows']:
        i = row['route']
        recalls = row['original'][0]['middle_recall']
        best_other = max(v for j, v in enumerate(recalls) if j != i)
        cue_only = row['cue_only_first']['middle_recall']
        decisions.append(dict(route=i, correct_recall=recalls[i], strongest_other_recall=best_other,
            strict_wrong_winner=best_other>recalls[i], nonzero_tie=best_other==recalls[i] and recalls[i]>0,
            desired_unique_winner=recalls[i]>best_other,
            cue_only_valid_branch_recalls=[cue_only[2*(i//2)], cue_only[2*(i//2)+1]],
            cue_only_other_cue_max=max(v for j,v in enumerate(cue_only) if j//2 != i//2)))
    return dict(assessment=a, decisions=decisions,
        strict_wrong_winners=sum(r['strict_wrong_winner'] for r in decisions),
        nonzero_ties=sum(r['nonzero_tie'] for r in decisions),
        no_target_activity=sum(r['correct_recall']==0 for r in decisions))


def unrelated_patterns(groups, scale):
    occupied = np.logical_or.reduce(list(groups.values()))
    unused = np.flatnonzero(~occupied)
    count = 32*scale
    assert len(unused) >= 2*count
    u = np.zeros(len(occupied), bool); u[unused[:count]] = True
    v = np.zeros(len(occupied), bool); v[unused[count:2*count]] = True
    assert not np.any(u & occupied) and not np.any(v & occupied)
    return u, v, np.zeros(len(occupied), bool)


def normalization_check():
    # Deterministically replicate base cells. Doubling the population cannot
    # supply extra summed drive merely because there are more sources.
    g, routes = fixture(1)
    base = model(1, 'soft_slow_weight_dependent')
    for _ in range(24):
        for route in routes:
            learn_sequence(base, g, route)
    query = g['query0']
    expected_e = base.驱动(query)
    expected_gate = base.门场(query)
    expected_activity = base.微步(query).copy()
    rows = []
    for scale in (2, 4):
        expanded = model(scale, 'soft_slow_weight_dependent')
        expanded_groups = {name: np.repeat(pattern, scale) for name,pattern in g.items()}
        for _ in range(24):
            for route in routes:
                learn_sequence(expanded, expanded_groups, route)
        e = expanded.驱动(expanded_groups['query0'])
        gate = expanded.门场(expanded_groups['query0'])
        activity = expanded.微步(expanded_groups['query0'])
        max_difference = float(np.max(np.abs(e-np.repeat(expected_e, scale))))
        assert np.allclose(e, np.repeat(expected_e, scale), atol=1e-12, rtol=1e-12)
        assert np.allclose(gate, expected_gate, atol=1e-12, rtol=1e-12)
        assert np.array_equal(activity, np.repeat(expected_activity, scale))
        rows.append(dict(scale=scale, max_current_difference=max_difference,
            gates_match=True, activation_replicated_exactly=True,
            external_current_all_zero=True))
    return rows


def main():
    started = time.perf_counter()
    protected = [ROOT/name for name in ('海马体时间区.py','视觉记忆区.py','听觉记忆区.py','运动记忆区.py','前额叶区.py','权重连接管理.py')]
    protected += [HERE/name for name in ('brain.py','original_runtime.py','sensory_adapter.py','viewer.py','results/living_original.npz','results/pfc_capacity_experiment.json','results/pfc_capacity_experiment.md')]
    before = {str(p): sha(p) for p in protected}
    results = []
    for scale in (1, 2, 4):
        groups, routes = fixture(scale)
        u, v, zero = unrelated_patterns(groups, scale)
        for condition in ('current', 'learnable_fast_forgetting_control', 'soft_slow_weight_dependent'):
            pfc = model(scale, condition)
            for epoch in range(24):
                for route in (routes if epoch%2==0 else routes[::-1]):
                    learn_sequence(pfc, groups, route)
            checkpoints = [dict(unrelated_learning_calls=0, result=measure(pfc, groups, scale))]
            extended = copy.deepcopy(pfc)
            for calls in range(1, 1025):
                previous, following = ((zero,u),(u,v),(v,zero))[(calls-1)%3]
                extended.学习(previous, following)
                if calls in (256, 1024):
                    checkpoints.append(dict(unrelated_learning_calls=calls, result=measure(extended, groups, scale)))
            repeated = copy.deepcopy(pfc)
            for _ in range(32):
                learn_sequence(repeated, groups, routes[0])
            results.append(dict(scale=scale, width=640*scale, condition=condition,
                calls_per_epoch=32, initial_epochs=24, checkpoints=checkpoints,
                after_32_extra_same_sequence=measure(repeated, groups, scale),
                external_current_probes={str(gain): probe(pfc, groups, groups['query0'], 'middle0', external_gain=gain, steps=6)
                                         for gain in (0., 3.5, 6.)}))
            print(json.dumps(dict(scale=scale, condition=condition,
                checkpoints=[dict(unrelated=x['unrelated_learning_calls'],
                    unique=x['result']['assessment']['unique_target_count'],
                    wrong=x['result']['strict_wrong_winners'], ties=x['result']['nonzero_ties'],
                    target_recall=x['result']['assessment']['mean_expected_recall'],
                    jaccard=x['result']['assessment']['mean_first_jaccard']) for x in checkpoints])), flush=True)
    calibration = normalization_check()
    no_decay_weight = .15*(1-(1-.005/.15)**24)
    after = {str(p): sha(p) for p in protected}
    report = dict(scope='Artificial clamped-neuron sequences only. No actual sensory brain, navigation, language or AGI claim. No parameter search or production parameter change.',
        candidate=dict(increment_at_scale_one=.005, ceiling_at_scale_one=.15,
            depression='weight_dependent', decay_rate=.9995, decay_interval=25,
            scale_rule='gamma, ceiling, deletion floor divided by population scale; feedback divided by scale and site width multiplied by scale.',
            precomputed_no_decay_edge_after_24=no_decay_weight,
            precomputed_32_source_min_edge=2.5/32,
            precomputed_no_decay_E=32*no_decay_weight,
            caveat='Analytical bound assumes 32 equal source weights, rest inhibition 1 and a target site with no currently active source. Real codes overlap, and all measured gates/interference are retained.'),
        unrelated_experience='Exactly 256 or 1024 original learning calls for zero -> U -> V -> zero. U and V each have 5% active cells and are disjoint from every training code. Old associations receive maintenance but no reinforcement; separate memories are not introduced.',
        cue_only_interpretation='Pure-cue activation of either learned context branch is partial-cue completion, not a scored failure. Explicit-context wrong winners and ties are reported separately.',
        normalization_check=calibration, results=results,
        source_hashes_before=before, source_hashes_after=after,
        protected_files_changed_concurrently=[k for k in before if before[k]!=after[k]],
        script_sha256=sha(Path(__file__)), seconds=time.perf_counter()-started)
    path = HERE/'results'/'pfc_retention_candidate.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# 单个解析可塑性候选的稀疏记忆保留实验', '',
        '本实验仍是人工神经元机制实验，未加载或修改生产模型；原容量实验结果保持不变。它验证一个已有函数的参数候选，不能称为自主导航、语言或 AGI。', '',
        f'候选在 640 宽度时 η=.005、W=.15，采用已有权重依赖衰减函数 r=.9995，每 25 学习维护一次。24 次无遗忘时每边 {no_decay_weight:.6f}，32 源 E={32*no_decay_weight:.4f}，超过静息 I=1、DI=.2 的 2.5 电流门槛；因此只选这一组，不做搜索。扩大时 γ/W/删除阈值反比缩放，局部反馈同步归一化。', '',
        '先训练每条共享线索+上下文序列 24 次，再插入 256 / 1024 次无关原学习调用。无关模式与训练模式完全无神经元重合，仍共享同一张连接表，不分开存储。', '',
        '| 宽度 | 条件 | 无关学习调用 | 唯一正确目标 / 8 | 明确串错 / 8 | 非零并列 / 8 | 平均目标召回 | 平均首步 Jaccard |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        for ck in row['checkpoints']:
            a=ck['result']; m=a['assessment']
            lines.append(f"| {row['width']} | {row['condition']} | {ck['unrelated_learning_calls']} | {m['unique_target_count']} | {a['strict_wrong_winners']} | {a['nonzero_ties']} | {m['mean_expected_recall']:.3f} | {m['mean_first_jaccard']:.3f} |")
    lines += ['', '纯线索会丢失上下文信息，因此同时或单独补全已有分支不算错误；JSON 单独保留这一轨迹。明确上下文时目标与错误目标并列也不算成功。目标召回 100% 并不意味着无多余激活，故同时报告 Jaccard。', '',
        '复制同一组神经元到 2 / 4 倍宽度的校准已通过：每目标电流与局部门场在 1e-12 以内一致，放电模式恰好按人数复制。校准与主召回实验外部电流均为 0；外驱 3.5 / 6 的探针只在同一规模内比较，不把跨规模总电流增加解释为容量优势。', '',
        f"耗时 {report['seconds']:.2f} 秒；保护文件并行变化列表：{report['protected_files_changed_concurrently']}。"]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(dict(report=str(path), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
