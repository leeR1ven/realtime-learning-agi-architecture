"""Offline ambiguity audit; positions are evaluator truth, never model inputs.

Reads the frozen navigation checkpoint and its completed 36-episode training
trace. Compares continuous stored receptor inputs, their paired quantization,
the actual 8192-cell code, and an untrained 32768-cell code that preserves the
entire actual 8192-cell prefix. No training, environment mutation, or rollout.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')).hexdigest()


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), minimum=float(a.min()), p10=float(np.percentile(a, 10)),
                median=float(np.median(a)), p90=float(np.percentile(a, 90)), maximum=float(a.max()))


def code_matrix(quantized, signature, context_source, rule_source, view_sources,
                thresholds, levels):
    goal, rule = int(signature) & 7, int(signature) >> 3
    use = ((goal & (1 << context_source)) != 0) & ((rule & (1 << rule_source)) != 0)
    columns = np.flatnonzero(use)
    sources = view_sources[columns]
    first_channel = sources[:, 0] // (levels * 2)
    second_channel = sources[:, 1] // (levels * 2)
    first_level = (sources[:, 0] // 2) % levels
    second_level = (sources[:, 1] // 2) % levels
    first = quantized[:, first_channel] > first_level
    second = quantized[:, second_channel] > second_level
    first ^= (sources[:, 0] % 2 == 1)
    second ^= (sources[:, 1] % 2 == 1)
    matrix = (2 + first.astype(np.uint8) + second.astype(np.uint8)) >= thresholds[columns]
    return matrix.astype(np.float32), columns


def jaccard_distance(matrix, query_indices):
    query = matrix[query_indices]
    intersection = query @ matrix.T
    union = query.sum(axis=1)[:, None] + matrix.sum(axis=1)[None, :] - intersection
    return np.where(union > 0, 1. - intersection / np.maximum(union, 1.), 0.)


def nearest_rows(distances, mask):
    masked = np.where(mask, distances, np.inf)
    indices = np.argmin(masked, axis=1)
    good = np.isfinite(masked[np.arange(len(indices)), indices])
    return indices, good


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=HERE/'results'/'route_switch_evaluation.npz')
    parser.add_argument('--training-report', type=Path, default=HERE/'results'/'route_switch_evaluation.json')
    parser.add_argument('--output', type=Path, default=HERE/'results'/'route_ambiguity_audit.json')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260906)
    args = parser.parse_args()
    begin_wall, begin_cpu = time.perf_counter(), time.process_time()
    checkpoint_hash = sha256(args.checkpoint)
    training_report = json.loads(args.training_report.read_text(encoding='utf-8'))
    training = training_report['training']
    assert len(training) == 36, 'Require all 36 training episodes; paired probes may still be running.'
    training_hash = canonical_hash(training)
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        names = ('event_ids', 'event_views', 'event_context', 'event_strength', 'event_frame',
                 'event_muscles', 'feature_offsets', 'features', 'context_source', 'rule_source',
                 'view_sources', 'thresholds')
        arrays = {name: archive[name].copy() for name in names}
        meta = json.loads(archive['metadata'].tobytes().decode('utf-8'))
    ids = arrays['event_ids']
    valid_slots = np.flatnonzero(ids >= 0)
    count = len(valid_slots)
    assert len(np.unique(ids[valid_slots])) == count
    assert np.array_equal(np.sort(ids[valid_slots]), np.arange(count)), 'Overwrite/gaps invalidate count-to-ID mapping.'
    assert meta['next_event_id'] == count == training[-1]['events_after']
    slot_by_id = np.full(count, -1, dtype=np.int64)
    slot_by_id[ids[valid_slots]] = valid_slots
    positions = np.empty((count, 2), dtype=np.float64)
    headings = np.empty(count, dtype=np.float64)
    episodes = np.empty(count, dtype=np.int16)
    steps = np.empty(count, dtype=np.int32)
    route_names = {}
    previous_end = 0
    for row in training:
        start, end = int(row['events_before']), int(row['events_after'])
        n = int(row['control_steps'])
        assert start == previous_end and end - start == n
        trajectory = row['trajectory']
        assert len(trajectory) == n + 1
        assert np.allclose(np.diff([x['time'] for x in trajectory]), .1, atol=1e-8)
        event_ids = np.arange(start, end, dtype=np.int64)
        slots = slot_by_id[event_ids]
        assert np.array_equal(arrays['event_frame'][slots], event_ids)
        positions[event_ids] = [point['position'] for point in trajectory[:-1]]
        headings[event_ids] = [point['heading'] for point in trajectory[:-1]]
        episodes[event_ids] = int(row['episode'])
        steps[event_ids] = np.arange(n)
        signatures = np.unique(arrays['event_context'][slots])
        assert len(signatures) == 1
        signature = int(signatures[0])
        route_names[signature] = row['requested_route']
        assert bool(np.all(arrays['event_strength'][slots] > 0)) == bool(row['correct_arrival'])
        if not row['correct_arrival']:
            assert not arrays['event_strength'][slots].any()
        previous_end = end
    rewarded_ids = ids[(ids >= 0) & (arrays['event_strength'] > 0)]
    rewarded_ids.sort()
    signatures = arrays['event_context'][slot_by_id[rewarded_ids]]
    rng = np.random.default_rng(args.seed)
    groups = np.unique(signatures)
    sampled = []
    for index, signature in enumerate(groups):
        candidates = rewarded_ids[signatures == signature]
        number = min(len(candidates), args.samples // len(groups) + int(index < args.samples % len(groups)))
        sampled.extend(rng.choice(candidates, number, replace=False).tolist())
    sampled = np.asarray(sampled, dtype=np.int64)

    level_count = int(meta['receptor_levels'])
    levels = np.linspace(1 / (level_count + 1), level_count / (level_count + 1),
                         level_count, dtype=np.float32)
    extra_count = 32768 - int(meta['mixed_neurons'])
    assert extra_count > 0 and meta['mixed_neurons'] == 8192
    extra_rng = np.random.default_rng(args.seed + 32768)
    expanded_context = np.concatenate((arrays['context_source'], extra_rng.integers(0, 3, extra_count, dtype=np.int32)))
    expanded_rule = np.concatenate((arrays['rule_source'], extra_rng.integers(0, 3, extra_count, dtype=np.int32)))
    expanded_view = np.concatenate((arrays['view_sources'], extra_rng.integers(0, 129 * level_count * 2,
                                                                            (extra_count, 2), dtype=np.int32)))
    expanded_threshold = np.concatenate((arrays['thresholds'], np.full(extra_count, 3.5, dtype=np.float32)))
    records = []
    comparisons = []
    examples = []
    validations = 0
    for signature in groups:
        bank_ids = rewarded_ids[signatures == signature]
        bank_slots = slot_by_id[bank_ids]
        query_ids = sampled[arrays['event_context'][slot_by_id[sampled]] == signature]
        query_indices = np.searchsorted(bank_ids, query_ids)
        values = arrays['event_views'][bank_slots]
        quantized = (values[:, :, None] >= levels[None, None, :]).sum(axis=2).astype(np.int16)
        raw = values.astype(np.float64)
        qraw = raw[query_indices]
        squared = np.maximum(0., (qraw * qraw).sum(axis=1)[:, None] +
                             (raw * raw).sum(axis=1)[None, :] - 2 * qraw @ raw.T)
        continuous = np.sqrt(squared / raw.shape[1])
        paired = np.empty((len(query_ids), len(bank_ids)), dtype=np.float32)
        for start in range(0, len(query_ids), 16):
            stop = min(start + 16, len(query_ids))
            l1 = np.abs(quantized[query_indices[start:stop], None, :] - quantized[None, :, :]).sum(axis=2)
            paired[start:stop] = l1 / (raw.shape[1] * level_count)
        actual_matrix, actual_columns = code_matrix(quantized, signature, arrays['context_source'],
            arrays['rule_source'], arrays['view_sources'], arrays['thresholds'], level_count)
        for qindex in query_indices:
            slot = bank_slots[qindex]
            stored = arrays['features'][arrays['feature_offsets'][slot]:arrays['feature_offsets'][slot + 1]]
            reconstructed = actual_columns[np.flatnonzero(actual_matrix[qindex])]
            assert np.array_equal(stored, reconstructed), 'Audit projection differs from actual stored activity.'
            validations += 1
        mixed = jaccard_distance(actual_matrix, query_indices)
        expanded_matrix, expanded_columns = code_matrix(quantized, signature, expanded_context,
            expanded_rule, expanded_view, expanded_threshold, level_count)
        prefix_columns = expanded_columns < 8192
        assert np.array_equal(expanded_columns[prefix_columns], actual_columns)
        assert np.array_equal(expanded_matrix[:, prefix_columns], actual_matrix)
        expanded = jaccard_distance(expanded_matrix, query_indices)
        physical = np.linalg.norm(positions[query_ids, None, :] - positions[bank_ids][None, :, :], axis=2)
        near_time = ((episodes[query_ids, None] == episodes[bank_ids][None, :]) &
                     (np.abs(steps[query_ids, None] - steps[bank_ids][None, :]) <= 20))
        eligible = ~near_time & (query_ids[:, None] != bank_ids[None, :])
        local = eligible & (physical <= .5)
        far = eligible & (physical >= 2.)
        metric_arrays = {'continuous_input_rmse': continuous,
                         'paired_receptor_mismatch': paired,
                         'mixed_8192_jaccard_distance': mixed,
                         'expanded_32768_jaccard_distance': expanded}
        nearest = {}
        for metric, distances in metric_arrays.items():
            all_j, all_valid = nearest_rows(distances, eligible)
            local_j, local_valid = nearest_rows(distances, local)
            far_j, far_valid = nearest_rows(distances, far)
            nearest[metric] = all_j
            for qi, query_id in enumerate(query_ids):
                assert all_valid[qi]
                j = int(all_j[qi])
                entry = dict(query_id=int(query_id), context=int(signature), route=route_names[int(signature)],
                    metric=metric, nearest_id=int(bank_ids[j]), nearest_distance=float(distances[qi, j]),
                    nearest_spatial_m=float(physical[qi, j]),
                    nearest_continuous_rmse=float(continuous[qi, j]),
                    nearest_paired_mismatch=float(paired[qi, j]),
                    nearest_mixed_8192_jaccard=float(1 - mixed[qi, j]),
                    exact_far_collision=bool(np.any((distances[qi] <= 0.) & far[qi])),
                    local_id=int(bank_ids[local_j[qi]]) if local_valid[qi] else None,
                    local_distance=float(distances[qi, local_j[qi]]) if local_valid[qi] else None,
                    far_id=int(bank_ids[far_j[qi]]) if far_valid[qi] else None,
                    far_distance=float(distances[qi, far_j[qi]]) if far_valid[qi] else None)
                records.append(entry)
        for qi, query_id in enumerate(query_ids):
            comparisons.append(dict(query_id=int(query_id), route=route_names[int(signature)],
                **{metric: dict(neighbor=int(bank_ids[js[qi]]), spatial_m=float(physical[qi, js[qi]]),
                    continuous_rmse=float(continuous[qi, js[qi]]), paired_mismatch=float(paired[qi, js[qi]]))
                   for metric, js in nearest.items()}))
            far_j = int(nearest_rows(mixed[qi:qi + 1], far[qi:qi + 1])[0][0])
            other = int(bank_ids[far_j])
            qslot, oslot = slot_by_id[query_id], slot_by_id[other]
            delta = arrays['event_views'][qslot] - arrays['event_views'][oslot]
            examples.append(dict(query_id=int(query_id), other_id=other, route=route_names[int(signature)],
                query_episode=int(episodes[query_id]), other_episode=int(episodes[other]),
                query_step=int(steps[query_id]), other_step=int(steps[other]),
                query_position=positions[query_id].tolist(), other_position=positions[other].tolist(),
                separation_m=float(physical[qi, far_j]),
                heading_difference_deg=float(abs(np.angle(np.exp(1j * (headings[query_id] - headings[other])))) * 180 / np.pi),
                continuous_rmse=float(continuous[qi, far_j]),
                depth_proximity_rmse=float(np.sqrt(np.mean(delta[:31] ** 2))),
                rgb_rmse=float(np.sqrt(np.mean(delta[31:124] ** 2))),
                body_rmse=float(np.sqrt(np.mean(delta[124:] ** 2))),
                paired_mismatch=float(paired[qi, far_j]),
                mixed_8192_jaccard=float(1 - mixed[qi, far_j]),
                expanded_32768_jaccard=float(1 - expanded[qi, far_j]),
                query_muscles=arrays['event_muscles'][qslot].tolist(),
                other_muscles=arrays['event_muscles'][oslot].tolist()))
        print(f'context {int(signature)} {route_names[int(signature)]}: bank={len(bank_ids)}, queries={len(query_ids)}', flush=True)

    summaries = {}
    for metric in sorted({x['metric'] for x in records}):
        rows = [x for x in records if x['metric'] == metric]
        paired_rows = [x for x in rows if x['local_distance'] is not None and x['far_distance'] is not None]
        summaries[metric] = dict(samples=len(rows),
            nearest_is_at_least_2m_away_count=sum(x['nearest_spatial_m'] >= 2 for x in rows),
            nearest_spatial_m=describe([x['nearest_spatial_m'] for x in rows]),
            nearest_distance=describe([x['nearest_distance'] for x in rows]),
            nearest_continuous_rmse=describe([x['nearest_continuous_rmse'] for x in rows]),
            best_local_distance=describe([x['local_distance'] for x in rows if x['local_distance'] is not None]),
            best_far_distance=describe([x['far_distance'] for x in rows if x['far_distance'] is not None]),
            local_and_far_available=len(paired_rows),
            far_at_least_as_similar_as_best_local_count=sum(x['far_distance'] <= x['local_distance'] for x in paired_rows),
            exact_far_collision_queries=sum(x['exact_far_collision'] for x in rows))
    projected_change = dict(
        width_changes_neighbor_count=sum(x['mixed_8192_jaccard_distance']['neighbor'] !=
                                        x['expanded_32768_jaccard_distance']['neighbor'] for x in comparisons),
        expansion_reduces_paired_mismatch_count=sum(x['expanded_32768_jaccard_distance']['paired_mismatch'] <
            x['mixed_8192_jaccard_distance']['paired_mismatch'] for x in comparisons),
        expansion_increases_paired_mismatch_count=sum(x['expanded_32768_jaccard_distance']['paired_mismatch'] >
            x['mixed_8192_jaccard_distance']['paired_mismatch'] for x in comparisons),
        raw_and_8192_nearest_both_far_count=sum(x['continuous_input_rmse']['spatial_m'] >= 2 and
            x['mixed_8192_jaccard_distance']['spatial_m'] >= 2 for x in comparisons),
        raw_nearest_local_but_8192_nearest_far_count=sum(x['continuous_input_rmse']['spatial_m'] <= .5 and
            x['mixed_8192_jaccard_distance']['spatial_m'] >= 2 for x in comparisons),
        expansion_changes_far_8192_neighbor_to_local_count=sum(x['mixed_8192_jaccard_distance']['spatial_m'] >= 2 and
            x['expanded_32768_jaccard_distance']['spatial_m'] <= .5 for x in comparisons))
    best_examples = sorted(examples, key=lambda x: (-x['mixed_8192_jaccard'], x['continuous_rmse']))[:12]
    assert sha256(args.checkpoint) == checkpoint_hash, 'Checkpoint changed during audit.'
    current_training = json.loads(args.training_report.read_text(encoding='utf-8'))['training']
    assert canonical_hash(current_training) == training_hash, 'Training history changed during audit.'
    report = dict(protocol=dict(seed=args.seed, requested_samples=args.samples, actual_samples=len(sampled),
        sampling='Balanced without replacement across reinforced goal/rule contexts; complete same-context reinforced bank.',
        time_exclusion='Same episode +/-20 action steps (2 seconds), plus identical event ID.',
        local_definition='Position separation <=0.5 m, no heading restriction.',
        far_definition='Position separation >=2 m, no heading restriction; descriptive thresholds, not causal proof.',
        continuous_definition='129 stored float32 pre-action receptor inputs BEFORE 16-level quantization: exp(-ray_distance/2), RGB, scaled/clipped body variables; not uncompressed camera/PCM.',
        paired_definition='Fraction of disagreeing paired receptor activations; equals sum(abs(level_count_a-level_count_b))/(129*16).',
        mixed_definition='Jaccard distance over active fixed mixed cells; both goal and rule held to the event context.',
        expanded_definition='Pure encoding intervention: exact trained 8192 prefix plus 24576 independently sampled same-distribution fixed cells. No learning or policy evaluation.',
        expanded_seed=args.seed+32768,
        depth_first_bin_saturation_beyond_m=float(-2 * np.log(float(levels[0]))),
        positions_never_fed_to_model=True, checkpoint_unchanged=True),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        training_history_sha256=training_hash, source_sha256=sha256(__file__),
        validation=dict(training_episodes=36, all_events=count, reinforced_events=len(rewarded_ids),
            explicit_event_id_to_slot_map=True, no_overwrite_or_id_gap=True,
            trajectory_samples_are_before_action=True, trajectory_has_steps_plus_one=True,
            projection_exactly_matches_saved_activity_queries=validations, expanded_prefix_preserved=True),
        summaries=summaries, encoding_width_comparison=projected_change,
        far_position_examples=best_examples, query_comparisons=comparisons,
        query_records=records,
        limitations=['Nearest-neighbor feature similarity is a representation diagnostic, not the actual degree-normalized multi-event/sequence policy.',
            'Pose difference is not by itself a task-relevant alias: different places may require the same action; muscle examples are observed actions, not optimal labels.',
            'The balanced 256-event sample is not a frequency-weighted population success estimate.',
            'Only reinforced training events are studied; this does not measure held-out switch-state coverage or prove why a particular rollout failed.',
            'The 32768-cell comparison only tests fixed encoding resolution. No larger learner was trained, and it cannot recover information already lost at quantization.'],
        elapsed_wall_seconds=time.perf_counter()-begin_wall, elapsed_cpu_seconds=time.process_time()-begin_cpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    labels = [('连续、量化前的129维感觉', 'continuous_input_rmse'),
              ('16级成对受体', 'paired_receptor_mismatch'),
              ('实际8192混合细胞', 'mixed_8192_jaccard_distance'),
              ('保留前缀扩至32768，仅编码', 'expanded_32768_jaccard_distance')]
    lines = ['# 路线感觉歧义离线审计', '',
        '连续感觉层已经存在远位置的近似观察；8192混合编码又引入少量额外混淆。'
        '纯编码扩大减少了部分混淆，不能据此声称扩大模型解决导航。', '',
        f'已核对36回合、{count}个连续事件ID、{len(rewarded_ids)}个受奖事件。'
        f'两规则各固定抽取{len(sampled)//len(groups)}个查询；每次搜索同规则完整受奖库，'
        '排除同回合前后2秒。事件与动作前轨迹通过显式ID→槽映射关联。', '',
        '| 表征 | 最近邻相距≥2m | 存在≥2m的完全相同编码 |',
        '|---|---:|---:|']
    for label, metric in labels:
        s = summaries[metric]
        lines.append(f"| {label} | {s['nearest_is_at_least_2m_away_count']}/{s['samples']} | {s['exact_far_collision_queries']}/{s['samples']} |")
    lines += ['', f"8192扩至32768后，{projected_change['width_changes_neighbor_count']}个查询的最近邻改变；"
        f"其中相对于成对受体的失配减少{projected_change['expansion_reduces_paired_mismatch_count']}例、"
        f"增加{projected_change['expansion_increases_paired_mismatch_count']}例。新投影未训练，没有进行扩大模型的行为测试。", '',
        f"距离先经exp(-d/2)处理，再经16级阈值量化。约{report['protocol']['depth_first_bin_saturation_beyond_m']:.3f}m以外"
        '均落入第一个距离量化区间；增加后续混合细胞无法还原已经丢失的区间内差异。', '',
        '| 两个事件ID | 间距 | 朝向差 | 连续感觉RMSE | RGB RMSE | 8192 Jaccard | 32768 Jaccard |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for example in best_examples[:6]:
        lines.append(f"| {example['query_id']} / {example['other_id']} | {example['separation_m']:.2f}m | "
            f"{example['heading_difference_deg']:.1f}° | {example['continuous_rmse']:.5f} | {example['rgb_rmse']:.5f} | "
            f"{example['mixed_8192_jaccard']:.5f} | {example['expanded_32768_jaccard']:.5f} |")
    lines += ['', '上述位置只用于离线审计，未传入模型。远位置并不自动表示需要不同动作；'
        '最相似的两个完全编码碰撞样例都记录了后退动作。因此这些数据只能说明表征存在别名，'
        '不能证明它导致某次路线失败。', '',
        '“连续感觉”是存档内量化前的受体输入，包含转换后的距离和缩放/截断的身体量，'
        '不是完整原始相机图像或PCM。这里的最近邻/Jaccard也不等同于实际带连接度归一化、'
        '多事件竞争和序列传播的控制策略。只统计受奖训练事件，未覆盖所有途中切换时的新视野。', '',
        '下一轮先保持8192不变，仅增加真实可见墙面纹理，是较清晰的单因素机制试验；'
        '该建议是由本次诊断作出的实验选择，并非对改动效果的保证。', '',
        f"检查点SHA256：{checkpoint_hash}",
        f"CPU耗时：{report['elapsed_cpu_seconds']:.3f}秒。输入检查点和36回合训练历史均未改变。"]
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(validation=report['validation'], summaries=summaries,
                          encoding_width_comparison=projected_change,
                          elapsed_wall_seconds=report['elapsed_wall_seconds'],
                          elapsed_cpu_seconds=report['elapsed_cpu_seconds']), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
