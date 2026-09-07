"""Same-experience RGB-only texture counterfactual, without any learning.

Uses exactly the first six far-pose pairs from the existing gray-wall ambiguity
audit. Restores their recorded camera poses only for offline ray rendering.
Verifies gray-wall RGB and transformed depth against the stored pre-action
experience, then changes only its 93 RGB values to actual textured wall hits.
All other stored inputs, goal/rule activity and 8192 birth connections stay fixed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from audit_route_ambiguity import canonical_hash, code_matrix, sha256
from textured_route_environment import TexturedRouteEnv


HERE = Path(__file__).resolve().parent


def metrics(first, second, first_code, second_code, levels):
    first_q = (first[:, None] >= levels[None, :]).sum(axis=1)
    second_q = (second[:, None] >= levels[None, :]).sum(axis=1)
    intersection = int(np.count_nonzero(first_code & second_code))
    union = int(np.count_nonzero(first_code | second_code))
    denominator = len(first) * len(levels)
    return dict(continuous_input_rmse=float(np.sqrt(np.mean((first.astype(float) - second.astype(float)) ** 2))),
        rgb_rmse=float(np.sqrt(np.mean((first[31:124].astype(float) - second[31:124].astype(float)) ** 2))),
        paired_receptor_mismatch=float(np.abs(first_q - second_q).sum() / denominator),
        paired_changed_level_count=int(np.abs(first_q - second_q).sum()),
        mixed_8192_jaccard=float(intersection / union) if union else 1.,
        mixed_8192_equal=bool(np.array_equal(first_code, second_code)),
        mixed_intersection=intersection, mixed_union=union,
        first_mixed_active=int(first_code.sum()), second_mixed_active=int(second_code.sum()))


def restore_camera_pose(env, position, heading):
    """Restore a recorded simulator state for rendering, never an agent action.

    Initial-reset collision rejection is inappropriate for a real recorded
    contact state. The existing world's finite pose fields are restored exactly;
    no integration, collision correction, scoring step or learner is called.
    """
    position = np.asarray(position, dtype=np.float64)
    heading = float(heading)
    assert position.shape == (2,) and np.isfinite(position).all() and np.isfinite(heading)
    assert 0 <= position[0] <= env.scene.width and 0 <= position[1] <= env.scene.height
    env.world.position = position.copy()
    env.world.heading = heading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=HERE/'results'/'route_switch_evaluation.npz')
    parser.add_argument('--training-report', type=Path, default=HERE/'results'/'route_switch_evaluation.json')
    parser.add_argument('--ambiguity-report', type=Path, default=HERE/'results'/'route_ambiguity_audit.json')
    parser.add_argument('--output', type=Path, default=HERE/'results'/'texture_encoding_counterfactual.json')
    parser.add_argument('--environment-seed', type=int, default=20260906)
    parser.add_argument('--texture-seed', type=int, default=44017)
    args = parser.parse_args()
    started = time.perf_counter()
    initial_hashes = {str(path.resolve()): sha256(path) for path in (args.checkpoint, args.ambiguity_report)}
    history = json.loads(args.training_report.read_text(encoding='utf-8'))['training']
    assert len(history) == 36
    history_hash = canonical_hash(history)
    previous = json.loads(args.ambiguity_report.read_text(encoding='utf-8'))
    pairs = previous['far_position_examples'][:6]
    assert len(pairs) == 6
    chosen_ids = sorted({int(pair[key]) for pair in pairs for key in ('query_id', 'other_id')})
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        names = ('event_ids', 'event_views', 'event_context', 'event_muscles', 'feature_offsets',
                 'features', 'context_source', 'rule_source', 'view_sources', 'thresholds')
        arrays = {name: archive[name].copy() for name in names}
        metadata = json.loads(archive['metadata'].tobytes().decode('utf-8'))
    valid = np.flatnonzero(arrays['event_ids'] >= 0)
    assert np.array_equal(np.sort(arrays['event_ids'][valid]), np.arange(len(valid)))
    id_to_slot = {int(arrays['event_ids'][slot]): int(slot) for slot in valid}
    poses = {}
    for row in history:
        first, last = int(row['events_before']), int(row['events_after'])
        assert last - first == row['control_steps'] and len(row['trajectory']) == row['control_steps'] + 1
        for event_id in chosen_ids:
            if first <= event_id < last:
                action_index = event_id - first
                state = row['trajectory'][action_index]
                poses[event_id] = dict(position=state['position'], heading=state['heading'],
                                       episode=row['episode'], action_index=action_index)
    assert set(poses) == set(chosen_ids)
    assert metadata['mixed_neurons'] == 8192
    level_count = int(metadata['receptor_levels'])
    levels = np.linspace(1 / (level_count + 1), level_count / (level_count + 1),
                         level_count, dtype=np.float32)
    env = TexturedRouteEnv(seed=args.environment_seed, texture_seed=args.texture_seed,
                           texture_enabled=False)
    env.reset(start='left_middle', target='red', route=None)
    environment_rng = copy.deepcopy(env.rng.bit_generator.state)
    renderer_time = float(env.world.time)
    original_inputs, textured_inputs, original_codes, textured_codes = {}, {}, {}, {}
    pose_checks = []
    for event_id in chosen_ids:
        slot = id_to_slot[event_id]
        original = arrays['event_views'][slot].copy()
        pose = poses[event_id]
        restore_camera_pose(env, pose['position'], pose['heading'])
        env.set_texture_enabled(False)
        gray_distance, gray_rgb, gray_points, gray_kinds = env.world._rays()
        restored_rgb = np.asarray(gray_rgb, dtype=np.float32).ravel()
        restored_proximity = np.exp(-np.asarray(gray_distance, dtype=np.float32) / 2.)
        rgb_max_error = float(np.max(np.abs(restored_rgb - original[31:124])))
        depth_max_error = float(np.max(np.abs(restored_proximity - original[:31])))
        assert np.allclose(restored_rgb, original[31:124], atol=1e-7, rtol=0), ('gray RGB mismatch', event_id, rgb_max_error)
        assert np.allclose(restored_proximity, original[:31], atol=1e-7, rtol=0), ('pre-action depth mismatch', event_id, depth_max_error)
        env.set_texture_enabled(True)
        texture_distance, texture_rgb, texture_points, texture_kinds = env.world._rays()
        assert np.array_equal(gray_distance, texture_distance)
        assert np.array_equal(gray_points, texture_points) and gray_kinds == texture_kinds
        for ray, kind in enumerate(gray_kinds):
            if kind != 'wall':
                assert np.array_equal(gray_rgb[ray], texture_rgb[ray])
        changed = original.copy()
        changed[31:124] = np.asarray(texture_rgb, dtype=np.float32).ravel()
        assert np.array_equal(changed[:31], original[:31]) and np.array_equal(changed[124:], original[124:])
        signature = int(arrays['event_context'][slot])
        both = np.stack((original, changed))
        quantized = (both[:, :, None] >= levels[None, None, :]).sum(axis=2).astype(np.int16)
        matrix, columns = code_matrix(quantized, signature, arrays['context_source'], arrays['rule_source'],
            arrays['view_sources'], arrays['thresholds'], level_count)
        code = np.zeros((2, 8192), dtype=bool)
        code[:, columns] = matrix.astype(bool)
        stored = arrays['features'][arrays['feature_offsets'][slot]:arrays['feature_offsets'][slot + 1]]
        assert np.array_equal(np.flatnonzero(code[0]), stored)
        original_inputs[event_id], textured_inputs[event_id] = original, changed
        original_codes[event_id], textured_codes[event_id] = code[0], code[1]
        pose_checks.append(dict(event_id=event_id, **pose, context_signature=signature,
            gray_rgb_exact_float32=bool(np.array_equal(restored_rgb, original[31:124])),
            gray_rgb_max_error=rgb_max_error, gray_depth_proximity_max_error=depth_max_error,
            gray_depth_proximity_exact_float32=bool(np.array_equal(restored_proximity, original[:31])),
            original_mixed_matches_saved_activity=True, non_rgb_stored_inputs_exactly_preserved=True,
            wall_rays=sum(kind == 'wall' for kind in gray_kinds),
            texture_changed_rgb_components=int(np.count_nonzero(changed[31:124] != original[31:124])),
            texture_changed_mixed_cells=int(np.count_nonzero(code[0] != code[1]))))
    results = []
    for pair in pairs:
        first, second = int(pair['query_id']), int(pair['other_id'])
        assert arrays['event_context'][id_to_slot[first]] == arrays['event_context'][id_to_slot[second]]
        baseline = metrics(original_inputs[first], original_inputs[second], original_codes[first], original_codes[second], levels)
        counterfactual = metrics(textured_inputs[first], textured_inputs[second], textured_codes[first], textured_codes[second], levels)
        assert np.isclose(baseline['continuous_input_rmse'], pair['continuous_rmse'], atol=1e-7, rtol=0)
        assert np.isclose(baseline['mixed_8192_jaccard'], pair['mixed_8192_jaccard'], atol=1e-7, rtol=0)
        results.append(dict(event_ids=[first, second], route=pair['route'],
            separation_m=pair['separation_m'], heading_difference_deg=pair['heading_difference_deg'],
            gray=baseline, rgb_only_texture=counterfactual,
            exact_collision_split=bool(baseline['mixed_8192_equal'] and not counterfactual['mixed_8192_equal'])))
    assert env.rng.bit_generator.state == environment_rng
    assert env.world.time == renderer_time
    for path, expected in initial_hashes.items():
        assert sha256(path) == expected
    assert canonical_hash(json.loads(args.training_report.read_text(encoding='utf-8'))['training']) == history_hash
    report = dict(protocol=dict(selection='Exactly the first six far-pose pairs in the prior gray-wall audit; no reselection.',
        environment_seed=args.environment_seed, texture_seed=args.texture_seed,
        intervention='Only stored 31x3 RGB replaced by renderer output at the exact recorded pre-action camera pose.',
        fixed='All 31 stored depth inputs and five body/pain/touch inputs, goal/rule activity, 16 levels, and all 8192 birth connections.',
        pose_restore='Finite recorded world.position/heading restored solely for ray rendering; no step, collision correction, scoring, reset-to-contact or learning.',
        position_or_route_oracle_fed_to_model=False, model_instantiated=False, model_trained=False,
        actual_controller_actions_evaluated=False),
        source_sha256=sha256(__file__), input_sha256=initial_hashes, training_history_sha256=history_hash,
        verification=dict(unique_event_poses=len(chosen_ids), all_gray_rgb_match=True, all_gray_depth_match=True,
            all_original_codes_match_stored=True, all_non_rgb_inputs_exact=True,
            all_geometric_rays_and_hit_types_unchanged=True, all_birth_connections_reused=True,
            environment_rng_unchanged=True, renderer_time_unchanged=True, input_files_unchanged=True),
        summary=dict(pairs=6, baseline_exact_code_collisions=sum(x['gray']['mixed_8192_equal'] for x in results),
            collisions_split=sum(x['exact_collision_split'] for x in results),
            all_pairs_have_lower_jaccard=all(x['rgb_only_texture']['mixed_8192_jaccard'] < x['gray']['mixed_8192_jaccard'] for x in results),
            gray_jaccard_range=[min(x['gray']['mixed_8192_jaccard'] for x in results), max(x['gray']['mixed_8192_jaccard'] for x in results)],
            texture_jaccard_range=[min(x['rgb_only_texture']['mixed_8192_jaccard'] for x in results), max(x['rgb_only_texture']['mixed_8192_jaccard'] for x in results)]),
        pairs=results, pose_checks=pose_checks,
        limitations=['This is a deliberately selected six-pair white-box representation counterfactual, not a random population estimate.',
            'It establishes that visible texture separates these particular old sensory codes at the unchanged 8192 width; it does not establish the unique cause of navigation outcomes.',
            'No textured policy, recall competition, action rollout, reward, or learning was tested. The unchanged old memories were not reinterpreted as newly trained texture memories.'],
        elapsed_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 同经历、仅RGB变化的纹理编码反事实', '',
        '固定旧灰墙审计最相似的6对，不重新挑样本。先按动作前真实姿态复原射线，再只把实际墙面纹理RGB替换入旧感觉；'
        '距离、身体感觉、目标/规则和8192出生连接均不改变。', '',
        f"{len(chosen_ids)}个姿态的原灰墙RGB与量化前距离全部匹配存档，原混合活动全部重建一致。没有运行学习或动作控制。", '',
        '| 事件对 | 感觉RMSE 灰→纹理 | 成对受体失配 灰→纹理 | 8192 Jaccard 灰→纹理 |',
        '|---|---:|---:|---:|']
    for row in results:
        a, b = row['gray'], row['rgb_only_texture']
        lines.append(f"| {row['event_ids'][0]} / {row['event_ids'][1]} | {a['continuous_input_rmse']:.5f} → {b['continuous_input_rmse']:.5f} | "
            f"{a['paired_receptor_mismatch']:.5f} → {b['paired_receptor_mismatch']:.5f} | {a['mixed_8192_jaccard']:.5f} → {b['mixed_8192_jaccard']:.5f} |")
    lines += ['', f"原来完全相同的{report['summary']['baseline_exact_code_collisions']}对编码中，纹理拆开了{report['summary']['collisions_split']}对。"
        '这表明纹理确实能通过现有16级受体和8192混合细胞提供可区别的感觉；不是必须先扩大细胞总数才能看到区别。', '',
        '该结论仅针对这6对已选样本的感觉编码。没有测试旧模型遇到纹理后的行为，也不能把正式导航结果的全部变化归因于这一个机制。'
        '位置只用于离线渲染校验，没有作为模型输入。旧报告、检查点和训练历史未修改。', '',
        f"纹理种子：{args.texture_seed}；环境种子：{args.environment_seed}；耗时：{report['elapsed_seconds']:.3f}秒。"]
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(verification=report['verification'], summary=report['summary'], pairs=results,
                         elapsed_seconds=report['elapsed_seconds']), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
