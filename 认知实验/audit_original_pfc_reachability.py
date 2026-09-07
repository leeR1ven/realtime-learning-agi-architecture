"""Read-only reachability audit of the original fixed PFC source graph.

Extract only its class with AST, without importing sensory modules or running
their demonstrations. The original constructor supplies the unchanged source
tables in an isolated process; sampled weights are ignored, never trained or
saved. Structural ancestors are not a claim about threshold activation, PCM
semantics, hippocampal retrieval, learned association edges, or task success.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("vision", "audio", "motor")


def scalar_assignments(tree):
    """Evaluate only top-level numeric constants and arithmetic, never calls."""
    values = {}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise ValueError("not a numeric assignment")

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                try:
                    values[node.targets[0].id] = evaluate(node.value)
                except (KeyError, ValueError):
                    pass
    return values


def inclusive_ranges(indices):
    values = np.unique(np.asarray(indices, dtype=np.int64))
    if not values.size:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 1) + 1]
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    return [[int(values[a]), int(values[b])] for a, b in zip(starts, ends)]


def summarize(mask):
    return dict(count=int(mask.sum()), fraction=float(mask.mean()),
                percent=float(mask.mean() * 100),
                output_ids_inclusive=inclusive_ranges(np.flatnonzero(mask)))


def audit():
    paths = {name: ROOT / name for name in
             ("视觉前处理.py", "听觉前处理.py", "运动输出区.py", "前额叶区.py")}
    originals = {name: path.read_bytes() for name, path in paths.items()}
    trees = {name: ast.parse(data.decode("utf-8-sig"), filename=str(paths[name]))
             for name, data in originals.items()}
    settings = {name: scalar_assignments(tree) for name, tree in trees.items()}
    widths = [int(settings[name]["对数"] * 2) for name in
              ("视觉前处理.py", "听觉前处理.py", "运动输出区.py")]
    for name in ("视觉前处理.py", "听觉前处理.py", "运动输出区.py"):
        assignments = [n for n in ast.walk(trees[name]) if isinstance(n, ast.Assign)]
        # Fail visibly if the original width relationship stops being true.
        expected = {"self.总数": "对数 * 2", "self.每层数": "输入层.总数",
                    "self.输出层数量": "self.每层数"}
        for target, expression in expected.items():
            assert any(any(ast.unparse(t) == target for t in n.targets)
                       and ast.unparse(n.value) == expression for n in assignments)

    pfc_tree = trees["前额叶区.py"]
    pfc_class = next(n for n in pfc_tree.body
                     if isinstance(n, ast.ClassDef) and n.name == "前额叶神经网络")
    forward = next(n for n in pfc_class.body
                   if isinstance(n, ast.FunctionDef) and n.name == "前向传播")
    concat = next(n for n in ast.walk(forward) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "np.concatenate")
    assert [ast.unparse(n.args[0]) for n in concat.args[0].elts] == [
        "视觉输出", "听觉输出", "运动输入"]
    namespace = {"np": np}
    extracted = ast.fix_missing_locations(ast.Module(body=[pfc_class], type_ignores=[]))
    exec(compile(extracted, str(paths["前额叶区.py"]), "exec"), namespace)
    config = settings["前额叶区.py"]
    total = sum(widths)
    # This is a disposable instance of the original class, not a new controller.
    state = np.random.get_state()
    try:
        np.random.seed(260906)
        network = namespace["前额叶神经网络"](
            total, 层数=int(config["层数"]), 连接半径=int(config["连接半径"]),
            权重范围=config["权重范围"], 隐藏层阈值=config["隐藏层阈值"],
            输出层阈值=config["输出层阈值"], 扩散宽度=int(config["扩散宽度"]))
    finally:
        np.random.set_state(state)
    boundaries = np.cumsum([0] + widths)
    tags = np.concatenate([np.full(width, 1 << k, dtype=np.uint8)
                           for k, width in enumerate(widths)])
    reachable = tags.copy()
    layer_rows = []
    radius = int(config["连接半径"])
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    expected_sources = (np.arange(total)[:, None] + offsets[None, :]) % total
    ancestor_offsets = np.array([0], dtype=np.int64)
    for layer, sources in enumerate(network.来源):
        # Enumerate the actual source table, with no threshold approximations.
        assert np.array_equal(sources, expected_sources)
        reachable = np.bitwise_or.reduce(reachable[sources], axis=1)
        ancestor_offsets = np.unique((ancestor_offsets[:, None] + offsets).ravel())
        layer_rows.append(dict(layer=layer + 1,
            vision_audio=summarize((reachable & 3) == 3),
            any_cross_modal=summarize(np.isin(reachable, [3, 5, 6, 7]))))

    # Equal circular source tables make these the exact input ancestor IDs
    # for EVERY output, not a sample or a geometric distance approximation.
    ancestors = (np.arange(total)[:, None] + ancestor_offsets[None, :]) % total
    exact_masks = np.bitwise_or.reduce(tags[ancestors], axis=1)
    assert np.array_equal(exact_masks, reachable)
    masks = {str(mask): dict(modalities=[m for k, m in enumerate(MODALITIES)
                                        if mask & (1 << k)],
                            **summarize(reachable == mask))
             for mask in range(1, 8)}
    examples = []
    selected_outputs = sorted({int((boundary + delta) % total)
                               for boundary in boundaries[:-1]
                               for delta in (-21, -20, -1, 0, 19, 20)})
    diffusion = int(config["扩散宽度"])
    for output_id in selected_outputs:
        ids = ancestors[output_id]
        expanded = []
        modality_details = {}
        for k, modality in enumerate(MODALITIES):
            start, end = int(boundaries[k]), int(boundaries[k + 1])
            positions = ids[(ids >= start) & (ids < end)]
            # Original 扩散 wraps WITHIN EACH modality, never across modalities.
            raw = np.unique(start + ((positions[:, None] - start +
                             np.arange(-diffusion, diffusion + 1)) % widths[k]))
            expanded.extend(raw.tolist())
            modality_details[modality] = dict(
                concatenated_input_ancestors=inclusive_ranges(positions),
                undiffused_modality_output_ancestors=inclusive_ranges(raw),
                count_before_diffusion=len(raw))
        raw_mask = int(np.bitwise_or.reduce(tags[expanded], initial=0))
        assert raw_mask == int(reachable[output_id])
        examples.append(dict(output_id=output_id, modality_mask=raw_mask,
                             ancestors=modality_details))

    unchanged = all(paths[name].read_bytes() == original
                    for name, original in originals.items())
    assert unchanged
    return {
        "scope": "原始固定前额叶来源图静态可达性；不修改结构、不训练、不执行感觉模块",
        "source_files": {name: {"path": str(paths[name]),
            "sha256": hashlib.sha256(data).hexdigest()} for name, data in originals.items()},
        "source_files_unchanged": unchanged,
        "input_blocks": {name: dict(width=widths[k], start_inclusive=int(boundaries[k]),
                                     end_exclusive=int(boundaries[k + 1]))
                         for k, name in enumerate(MODALITIES)},
        "total_output_neurons": total,
        "configuration": {key: config[key] for key in
            ("层数", "连接半径", "扩散宽度", "隐藏层阈值", "输出层阈值")},
        "propagation_layers": len(network.来源),
        "actual_source_tables_match_original_ring_formula": True,
        "ancestor_formula": "For output i: (i + each ancestor_offset) % total_output_neurons",
        "ancestor_offsets": ancestor_offsets.tolist(),
        "ancestors_per_output_after_modality_diffusion": len(ancestor_offsets),
        "source_table_propagation_matches_full_ancestor_enumeration": True,
        "vision_audio": summarize((reachable & 3) == 3),
        "vision_motor": summarize((reachable & 5) == 5),
        "audio_motor": summarize((reachable & 6) == 6),
        "all_three": summarize(reachable == 7),
        "any_cross_modal": summarize(np.isin(reachable, [3, 5, 6, 7])),
        "per_output_modality_mask_runs": masks,
        "by_propagation_layer": layer_rows,
        "boundary_examples": examples,
        "diffusion_effect": "每路先在自己的段内扩散；改变原始特征祖先范围，不能把本来只依赖单模态的输出变成跨模态输出。因此上述模态共可达数量对扩散前后相同。",
        "limitations": [
            "只审原固定前额网，未把海马共享时间关联、可塑前额联想、返回线多轮循环并入此静态图。",
            "可达不等于达到放电阈值，也不等于一个细胞必须同时依赖两路才放电。",
            "不可从跨模态细胞数量稀少直接推出某个PCM不能引起视觉联想；需要实际活动、已学连接和抑制状态探针。",
            "输出只有三路并排拼接，并不代表任意视觉与任意听觉细胞都在固定前向网内混合。",
            "本次未运行任何训练、导航、声音识别或目标图像保持实验。",
        ],
        "line_references": {
            "vision_layout": "视觉前处理.py:68-81", "audio_layout": "听觉前处理.py:68-79",
            "motor_layout": "运动输出区.py:94-103", "ring_sources": "前额叶区.py:39-50",
            "diffusion_and_concat": "前额叶区.py:8-15,54-67",
            "global_pfc_config": "前额叶区.py:308-319",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent /
                        "results" / "original_pfc_reachability.json")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
        ("total_output_neurons", "vision_audio", "any_cross_modal", "all_three", "source_files_unchanged")},
        ensure_ascii=False, indent=2))
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
