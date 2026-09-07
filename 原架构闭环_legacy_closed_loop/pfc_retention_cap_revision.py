"""Analytical correction: do not cap a learned partial cue below its gate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from pfc_retention_candidate import model, measure, unrelated_patterns
from pfc_capacity_experiment import HERE, ROOT, sha, fixture, learn_sequence


def main():
    started = time.perf_counter()
    protected = [ROOT/'前额叶区_prefrontal.py', ROOT/'权重连接管理_weight_manager.py', HERE/'brain.py', HERE/'original_runtime.py',
        HERE/'results/living_original.npz', HERE/'results/pfc_capacity_experiment.json',
        HERE/'results/pfc_retention_candidate.json', HERE/'results/pfc_retention_candidate.md']
    before = {str(p):sha(p) for p in protected}
    results = []
    for scale in (1,2,4):
        groups, routes = fixture(scale)
        pfc = model(scale, 'soft_slow_weight_dependent')
        for table in (pfc.联想, pfc.去抑):
            assert table.条数() == 0
            table.ceiling = .2/scale
        for epoch in range(24):
            for route in (routes if epoch%2==0 else routes[::-1]):
                learn_sequence(pfc, groups, route)
        checkpoints = [dict(unrelated=0, result=measure(pfc, groups, scale))]
        extended = copy.deepcopy(pfc)
        u, v, zero = unrelated_patterns(groups, scale)
        for calls in range(1,1025):
            previous, following = ((zero,u),(u,v),(v,zero))[(calls-1)%3]
            extended.学习(previous, following)
            if calls in (256,1024):
                checkpoints.append(dict(unrelated=calls, result=measure(extended, groups, scale)))
        repeated = copy.deepcopy(pfc)
        repetitions = []
        for count in range(1,129):
            learn_sequence(repeated, groups, routes[0])
            if count in (32,128):
                repetitions.append(dict(additional_same_sequence=count, result=measure(repeated, groups, scale)))
        row = dict(scale=scale, width=640*scale, checkpoints=checkpoints, repeated=repetitions)
        results.append(row)
        print(json.dumps(dict(scale=scale,
            retained_unique=[x['result']['assessment']['unique_target_count'] for x in checkpoints],
            repeated_unique=[x['result']['assessment']['unique_target_count'] for x in repetitions],
            pure_cue_branch_zero=[x['result']['decisions'][0]['cue_only_valid_branch_recalls'] for x in repetitions])), flush=True)
    after = {str(p):sha(p) for p in protected}
    report = dict(scope='Artificial synthetic-neuron follow-up only; previous candidate data and living brain are not changed.',
        reason='The earlier .15 ceiling gives half-cue 16-source maximum E=2.4, below the resting E/DI joint requirement 2.5. Increasing W analytically to .2 permits a learned partial cue to cross that gate; no data-selected sweep.',
        gamma=.005, ceiling=.2, depression='weight_dependent', rate=.9995, interval=25,
        scaling='gamma, ceiling and deletion floor divided by scale; feedback divided by scale and site width multiplied by scale.',
        no_decay_after_24=.2*(1-(1-.005/.2)**24),
        half_cue_maximum_E=16*.2, resting_joint_required_E=2.5,
        results=results, protected_hashes_before=before, protected_hashes_after=after,
        protected_changed_concurrently=[k for k in before if before[k]!=after[k]],
        script_sha256=sha(Path(__file__)), seconds=time.perf_counter()-started)
    path = HERE/'results/pfc_retention_cap_revision.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    lines = ['# 保留候选的上限解析纠正与重复学习边界', '',
        '原 W=.15 实验全部保留。复查发现，在本实验固定静息抑制 I=1 下，半线索 16 源最大 E=2.4，低于同时计入去抑后的 2.5 要求。因此不能把该参数阻止纯线索补全当成优点。', '',
        '仅把 W 改为 .2，使半线索最大 E=3.2，理论上允许足够重复后的部分线索补全。η=.005、权重依赖衰减 .9995/25 不变，其余按人数归一化。这是门槛算式发现约束后的单次纠正，不是参数扫描。所有训练仍是人工夹持，不能解释成真实导航或 AGI。', '',
        '| 宽度 | 1024 无关更新后唯一目标 / 8 | 32 次偏重重复后唯一目标 / 8 | 128 次后唯一目标 / 8 | 128 次后纯线索对两条合法分支的召回 |',
        '|---:|---:|---:|---:|---|']
    for row in results:
        ret=row['checkpoints'][-1]['result']; r32=row['repeated'][0]['result']; r128=row['repeated'][1]['result']
        pure=r128['decisions'][0]['cue_only_valid_branch_recalls']
        lines.append(f"| {row['width']} | {ret['assessment']['unique_target_count']} | {r32['assessment']['unique_target_count']} | {r128['assessment']['unique_target_count']} | {pure} |")
    lines += ['', '只给共享线索时，激活重复较多的合法分支是部分线索补全，不记错误。明确提供另一上下文时，旧目标与被偏重目标并列则仍不能说已经正确选路；所有并列和真正错误胜出分别保留。目标召回 100% 不排除多余激活。', '',
        '这组函数能保存稀疏关联，并能在更多重复后产生部分线索补全；它仍不足以解决共享特征导致的多目标共同激活。不能把上限提高再慢遗忘直接发布到成熟生活模型。成熟生活已有权重，且活跃簇大小变化，进一步实测必须独立进行并按真实源数检查电流。', '',
        f"执行 {report['seconds']:.2f} 秒，保护文件变化：{report['protected_changed_concurrently']}。"]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(path),seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    main()
