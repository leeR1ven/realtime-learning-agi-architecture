"""Bounded, artificial-code PFC capacity/context/retention experiment.

This never loads or changes a living brain. All training patterns are explicitly
clamped synthetic neuron sets. The original learning and microstep equations are
used. It is neither navigation nor language nor a full-model scaling benchmark.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from fast_pfc_currents import FastPFC
from 前额叶区 import 前额叶联想区
from 权重连接管理 import 连接表


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(scale):
    """Fixed active fraction, input count and current expectation across sizes."""
    width, k = 640 * scale, 16 * scale
    rng = np.random.default_rng(20260906)
    groups = {}
    for prefix, number, quarter, count in [('cue', 4, 0, k), ('context', 2, 1, k),
                                          ('middle', 8, 2, 2*k), ('end', 8, 3, 2*k)]:
        pool = np.arange(quarter*160*scale, (quarter+1)*160*scale)
        for i in range(number):
            pattern = np.zeros(width, bool)
            pattern[rng.choice(pool, count, replace=False)] = True
            groups[f'{prefix}{i}'] = pattern
    groups['zero'] = np.zeros(width, bool)
    routes = []
    for cue in range(4):
        for context in range(2):
            i = 2*cue+context
            groups[f'query{i}'] = groups[f'cue{cue}'] | groups[f'context{context}']
            assert groups[f'query{i}'].sum() == 2*k
            routes.append((f'query{i}', f'middle{i}', f'end{i}', 'zero'))
    return groups, routes


def birth(scale, gamma):
    pfc = FastPFC(640*scale, 每段=40*scale, 目标下限=4*scale,
                  目标上限=40*scale, 反馈抑制=.12/scale)
    for table in (pfc.联想, pfc.去抑):
        table.强化量 = gamma/scale
        table.衰减率 = .9
        table.消失下限 = .00005/scale
        table.警戒线 = 1
        table.衰减间隔 = 25
        table.上次衰减维护 = -25
    assert FastPFC.学习 is 前额叶联想区.学习
    assert FastPFC.微步 is 前额叶联想区.微步
    return pfc


def learn_sequence(pfc, groups, route):
    previous = groups['zero']
    for name in route:
        following = groups[name]
        pfc.学习(previous, following)
        previous = following


def probe(pfc, groups, query, expected, external_gain=0., steps=3):
    # Only volatile state changes during a frozen probe, then is restored.
    old_strength, old_thought = pfc.强度, pfc.念头
    previous = query.copy()
    trace = []
    external = query.astype(float)*external_gain if external_gain else None
    try:
        for step in range(steps):
            e = pfc.驱动(previous)
            net = pfc.门场(previous)[np.arange(pfc.宽度)//pfc.每段]
            threshold = pfc.门槛+net
            active = pfc.微步(previous, 外部信号=external).copy()
            assert np.array_equal(active, e+(0 if external is None else external) >= threshold)
            middle = [float((active & groups[f'middle{i}']).sum()/groups[f'middle{i}'].sum()) for i in range(8)]
            end = [float((active & groups[f'end{i}']).sum()/groups[f'end{i}'].sum()) for i in range(8)]
            target = groups[expected]
            union = int((active | target).sum())
            trace.append(dict(step=step+1, active_count=int(active.sum()),
                expected_recall=float((active & target).sum()/target.sum()),
                expected_jaccard=float((active & target).sum()/union) if union else 1.,
                middle_recall=middle, end_recall=end,
                source_reactivated_fraction=float((active & query).sum()/max(1, query.sum())),
                desired_E_min=float(e[target].min()), desired_E_max=float(e[target].max()),
                desired_threshold_min=float(threshold[target].min()),
                desired_threshold_max=float(threshold[target].max()),
                inhibition_before=float(old_strength if step == 0 else trace[-1]['inhibition_after']),
                inhibition_after=float(pfc.强度)))
            previous = active
    finally:
        pfc.强度, pfc.念头 = old_strength, old_thought
    return trace


def assess(pfc, groups, scale):
    rows = []
    rng = np.random.default_rng(409)
    for i in range(8):
        cue = groups[f'query{i}']
        original = probe(pfc, groups, cue, f'middle{i}')
        context_changed = probe(pfc, groups, groups[f'query{i^1}'], f'middle{i^1}')
        # Same number of active inputs: replace 1/8 of context cells by unused
        # context-pool cells, never changing the cue or supplying an answer.
        perturbed = cue.copy()
        remove = rng.choice(np.flatnonzero(groups[f'context{i%2}']), 2*scale, replace=False)
        pool = np.arange(160*scale, 320*scale)
        add = rng.choice(pool[~perturbed[pool]], len(remove), replace=False)
        perturbed[remove] = False
        perturbed[add] = True
        assert perturbed.sum() == cue.sum()
        first = original[0]
        competitors = [v for j, v in enumerate(first['middle_recall']) if j != i]
        rows.append(dict(route=i, original=original,
            first_step_expected_minus_best_other=first['middle_recall'][i]-max(competitors),
            first_step_target_is_unique_best=first['middle_recall'][i]>max(competitors),
            exact_first_target=first['expected_jaccard']==1.,
            end_second_step_recall=original[1]['end_recall'][i],
            fully_changed_context_first=context_changed[0],
            eighth_context_replaced_first=probe(pfc, groups, perturbed, f'middle{i}', steps=1)[0],
            cue_only_first=probe(pfc, groups, groups[f'cue{i//2}'], f'middle{i}', steps=1)[0]))
    return dict(unique_target_count=sum(r['first_step_target_is_unique_best'] for r in rows),
                exact_first_target_count=sum(r['exact_first_target'] for r in rows),
                mean_expected_recall=float(np.mean([r['original'][0]['expected_recall'] for r in rows])),
                mean_first_jaccard=float(np.mean([r['original'][0]['expected_jaccard'] for r in rows])),
                mean_second_end_recall=float(np.mean([r['end_second_step_recall'] for r in rows])),
                mean_target_margin=float(np.mean([r['first_step_expected_minus_best_other'] for r in rows])),
                rows=rows)


def retention_edges():
    cases = []
    for rate in (.9, .9995):
        for period in (1, 32, 128):
            table = 连接表(强化量=.0001, 衰减率=rate, 消失下限=.00005, 警戒线=1, 衰减间隔=25)
            values = []
            for frame in range(8192):
                table.维护()
                if frame % period == 0:
                    table.学习(0, 1)
                # Unrelated experience also uses the same connection table.
                table.学习(2, 3)
                if frame+1 in (128, 512, 2048, 8192):
                    w = table.查(0).get(1, 0.)
                    values.append(dict(maintenance=frame+1, weight=w,
                        excitation_for_350_identical_sources=350*w,
                        best_case_threshold=2+max(0, 1-.2*350*w)))
            cases.append(dict(decay_rate=rate, exposure_every=period, samples=values))
    # One exposure, then genuine empty learning calls. Discover disappearance,
    # instead of reporting the continuous half-life as an exact delete time.
    pfc = birth(1, .0001)
    a = np.zeros(640, bool); a[0] = True
    b = np.zeros(640, bool); b[100] = True
    zero = np.zeros(640, bool)
    pfc.学习(a, b)
    disappearance = None
    for frame in range(1, 300):
        pfc.学习(zero, zero)
        if not pfc.联想.查(0).get(100, 0.):
            disappearance = frame
            break
    assert disappearance is not None
    return dict(original_edge_cases=cases,
        one_exposure_deleted_after_empty_learning_calls=disappearance,
        analytic_half_life_maintenance=25*math.log(.5)/math.log(.9),
        caveat='Scalar edge calculation is a best-case current bound, not a learned behavioral task. Slower uniform decay alone also retains dense connections more strongly.')


def main():
    started = time.perf_counter()
    protected = [ROOT/name for name in ('海马体时间区.py','视觉记忆区.py','听觉记忆区.py','运动记忆区.py','前额叶区.py','权重连接管理.py')]
    protected += [HERE/name for name in ('brain.py','original_runtime.py','sensory_adapter.py','viewer.py','results/living_original.npz')]
    before = {str(path): sha(path) for path in protected}
    results = []
    for scale in (1, 2, 4):
        groups, routes = fixture(scale)
        for label, gamma in [('current_normalized', .0001), ('analytical_mechanism_control', .012)]:
            pfc = birth(scale, gamma)
            milestones = []
            for epoch in range(1, 97):
                # Every route appears once per cycle; alternate direction to
                # reduce recency from a permanent route order.
                for route in (routes if epoch%2 else routes[::-1]):
                    learn_sequence(pfc, groups, route)
                if epoch in (1, 8, 32, 96):
                    measured = assess(pfc, groups, scale)
                    milestones.append(dict(epochs=epoch, learning_calls=pfc.联想.维护次数,
                        edge_count=pfc.联想.条数(), assessment=measured))
                    print(json.dumps(dict(scale=scale, condition=label, epoch=epoch,
                        unique=measured['unique_target_count'], exact=measured['exact_first_target_count'],
                        recall=measured['mean_expected_recall'], jaccard=measured['mean_first_jaccard'])), flush=True)
            external = {str(gain): probe(pfc, groups, groups['query0'], 'middle0', external_gain=gain, steps=6)
                        for gain in (0., 3.5, 6., 9.5)}
            # Preserve whole trained weights and run further repetitions on an
            # independent copy, then switch only the context/target experience.
            repeated = copy.deepcopy(pfc)
            for _ in range(32):
                learn_sequence(repeated, groups, routes[0])
            repeated_probe = assess(repeated, groups, scale)
            switched = copy.deepcopy(pfc)
            changed_route = ('query0', 'middle1', 'end1', 'zero')
            for _ in range(8):
                learn_sequence(switched, groups, changed_route)
            switched_probe = probe(switched, groups, groups['query0'], 'middle1')
            results.append(dict(scale=scale, width=640*scale, primitive_source_count=16*scale,
                query_count=32*scale, query_active_fraction=.05, gamma=gamma/scale,
                feedback=.12/scale, segment_width=40*scale, condition=label,
                milestones=milestones, constant_external_current_probes=external,
                extra_32_one_route_repetitions=repeated_probe,
                eight_changed_outcome_exposures=switched_probe))
    after = {str(path): sha(path) for path in protected}
    changed = [path for path in before if before[path] != after[path]]
    # Other parallel workers may be editing production files. We never write
    # them; retain both hashes transparently instead of falsely claiming static.
    report = dict(scope='Artificial clamped-neuron mechanism experiment; not real navigation, language or AGI, and not a full-model scaling experiment.',
        normalization='Source activity fraction fixed at 5%; gamma and deletion floor divided by scale, local feedback divided by scale and segment width multiplied by scale. Thus source-current expectation and local-feedback expectation do not rise merely from more cells. Random patterns within source/output pools may share cells; overlap fluctuations can differ with width.',
        curriculum='Eight synthetic cue+context -> middle -> end -> blank sequences; 4 cues and 2 contexts reused, random output codes also overlap. Every stored connection was created by original PFC learning; no direct weight installation.',
        fixed_parameter_choice='Two rates fixed before results: current .0001 and control .012. Control chosen from 32-source, once/32-maintenance steady current near threshold; not a parameter grid. This high-rate control is not proposed for production.',
        scales=results, edge_retention=retention_edges(),
        source_hashes_before=before, source_hashes_after=after,
        protected_files_changed_concurrently=changed, script_sha256=sha(Path(__file__)),
        seconds=time.perf_counter()-started)
    path = HERE/'results'/'pfc_capacity_experiment.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# 循环前额叶容量、上下文和遗忘机制实验', '',
        '这是人工指定神经元模式的机制实验；没有加载生活存档，没有向实际模型写入路线、目标或语言知识。不是仿真导航成功，也不是 AGI。', '',
        '三档宽度为 640 / 1280 / 2560，查询活跃比例均为 5%。学习增量与删除下限按人数反比缩放，局部反馈与位点大小同时归一化，避免更多人头直接提供更大电流。放电门槛、维护周期及衰减率不变。随机码在各池内可能重叠；只试一个固定随机种子，不能据此确定规模上限。', '',
        '训练 8 条共享线索和上下文的“查询→中间簇→末端簇→空白”序列。当前增量的归一化组和一个事先按门槛估算的高增量机制对照都使用原学习方程。高增量组不是生产参数推荐。', '',
        '| 条件 | 宽度 | 每序列重复 | 首步唯一正确目标 / 8 | 首步目标召回率 | 首步 Jaccard | 第二步末端召回率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        for m in row['milestones']:
            a=m['assessment']
            lines.append(f"| {row['condition']} | {row['width']} | {m['epochs']} | {a['unique_target_count']} | {a['mean_expected_recall']:.3f} | {a['mean_first_jaccard']:.3f} | {a['mean_second_end_recall']:.3f} |")
    ret = report['edge_retention']
    lines += ['', f"当前 0.9 / 25 的未强化连接半衰期为 {ret['analytic_half_life_maintenance']:.2f} 次维护；单次 .0001 新连接在本实验后续 {ret['one_exposure_deleted_after_empty_learning_calls']} 次空学习调用后删除。", '',
        '这里的每次维护是前额叶学习调用，不应无条件等同于画面帧。当前生活控制每步调用一次时，10 Hz 约对应 16.4 秒半衰期。完整 JSON 同时列出 .9 和 .9995 衰减下每 1 / 32 / 128 次维护出现一次的同边结果；慢衰减也会增强高频边，不能只改遗忘速度而不检查输入区分。', '',
        '完整 JSON 包括每目标召回、竞争目标、完整上下文替换、替换 1/8 上下文细胞、32 次局部重复、8 次改变后继的训练，以及 0 / 3.5 / 6 / 9.5 外驱的 6 微步轨迹。探测不学习并恢复瞬时抑制状态，继续学习使用独立副本。', '',
        f"执行耗时 {report['seconds']:.2f} 秒。保护文件在运行前后发生并行变化：{changed}。本脚本的写操作仅限两个独立结果文件。"]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps(dict(report=str(path), seconds=report['seconds'], changed=changed)), flush=True)


if __name__ == '__main__':
    main()
