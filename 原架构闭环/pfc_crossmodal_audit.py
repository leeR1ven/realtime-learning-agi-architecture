"""Read-only born-PFC audit with real rendering and crossed ordinary tones.

Nothing is trained, no production brain is loaded, and no world action is
claimed. Modality removal is an explicit neural clamp after real encoding.
All positions, colors and tone names below belong to the experiment only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from sensory_adapter import SensoryAdapter, _LocalNumpy
from experience_environment import ExperienceEnvironment


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GROUPS = ('rgb', 'depth', 'body', 'audio', 'motor')
COLORS = ('red', 'blue')
TONES = (220., 660.)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def ids(array):
    return np.flatnonzero(array).astype(int).tolist()


def ranges(indices):
    values = sorted(set(map(int, indices)))
    result = []
    for value in values:
        if result and result[-1][1]+1 == value:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def original_forward(width, seed, config):
    path = ROOT / '前额叶区_prefrontal.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    wanted = {'扩散', '前额叶神经网络'}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    namespace = {'np': _LocalNumpy(seed+10001)}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['前额叶神经网络'](width,
        隐藏层阈值=config['pfc_hidden_threshold'], 输出层阈值=config['pfc_output_threshold'],
        扩散宽度=config['pfc_spread'])


def born_hash(network):
    h = hashlib.sha256()
    for collection in (network.来源, network.权重, network.阈值):
        for array in collection:
            h.update(array.tobytes())
    return h.hexdigest()


def reachability(network, adapter):
    lengths = [adapter.scalar_widths[g]*adapter.scalar_levels[g]*2 for g in GROUPS]
    boundaries = np.cumsum([0]+lengths)
    labels = np.concatenate([np.full(width, 1 << i, np.uint8) for i, width in enumerate(lengths)])
    modality_labels = np.concatenate([np.full(adapter.widths[g], 1 << i, np.uint8)
                                      for i, g in enumerate(('visual', 'audio', 'motor'))])
    rows = []
    direct = labels.copy()
    modal = modality_labels.copy()
    # Source indices address the expanded input. Account separately for its
    # within-modality circular spread back to the raw adapter output cells.
    expanded = labels.copy()
    start = 0
    for name in ('visual', 'audio', 'motor'):
        width = adapter.widths[name]
        source = (np.arange(width)[:, None]+np.arange(-network.扩散宽度, network.扩散宽度+1)) % width
        expanded[start:start+width] = np.bitwise_or.reduce(labels[start+source], axis=1)
        start += width

    def measure(mask):
        return dict(count=int(mask.sum()), fraction=float(mask.mean()),
                    ids_inclusive=ranges(np.flatnonzero(mask)))

    for layer, sources in enumerate(network.来源):
        direct = np.bitwise_or.reduce(direct[sources], axis=1)
        expanded = np.bitwise_or.reduce(expanded[sources], axis=1)
        modal = np.bitwise_or.reduce(modal[sources], axis=1)
        rows.append(dict(layer=layer+1,
            vision_audio_after_spread=measure((modal & 3) == 3),
            rgb_audio_ignoring_spread=measure((direct & 9) == 9),
            rgb_audio_including_spread=measure((expanded & 9) == 9),
            depth_audio_including_spread=measure((expanded & 10) == 10),
            body_audio_including_spread=measure((expanded & 12) == 12),
            any_multiple_modalities=measure(np.isin(modal, [3, 5, 6, 7])),
            all_three_modalities=measure(modal == 7)))
    graph = dict(groups={g:dict(start=int(boundaries[i]), end_exclusive=int(boundaries[i+1]))
                         for i,g in enumerate(GROUPS)}, width=int(sum(lengths)), layers=rows)
    graph['complete_output_ancestry_groups'] = {
        '+'.join(g for i,g in enumerate(GROUPS) if mask & (1 << i)):
            measure(expanded == mask) for mask in sorted(set(expanded.tolist()))}
    return graph


def source_trace(network, output_id, adapter):
    hidden_ids = network.来源[-1][output_id]
    expanded_inputs = np.unique(network.来源[0][hidden_ids])
    raw_inputs = set()
    offsets = np.cumsum([0]+[adapter.widths[g] for g in ('visual','audio','motor')])
    for index in expanded_inputs:
        segment = int(np.searchsorted(offsets[1:], index, side='right'))
        start, end = offsets[segment:segment+2]
        for delta in range(-network.扩散宽度, network.扩散宽度+1):
            raw_inputs.add(int(start+(index-start+delta) % (end-start)))
    lengths = [adapter.scalar_widths[g]*adapter.scalar_levels[g]*2 for g in GROUPS]
    boundaries = np.cumsum([0]+lengths)
    return dict(output_id=int(output_id), hidden_sources=hidden_ids.tolist(),
        pre_spread_ancestors={g:ranges(i for i in raw_inputs if boundaries[k] <= i < boundaries[k+1])
                              for k,g in enumerate(GROUPS)})


def project(network, visual, audio, motor, trace=False):
    result = network.前向传播(visual, audio, motor).astype(bool).copy()
    if not trace:
        return result
    previous = network.输入激活.astype(float)
    currents = []
    for layer in range(len(network.来源)):
        currents.append((network.权重[layer]*previous[network.来源[layer]]).sum(axis=1))
        previous = network.普通层激活[layer].astype(float)
    return result, currents


def sensory_hash(encoded):
    h = hashlib.sha256()
    for name in ('visual','audio','motor'):
        h.update(encoded[name].tobytes())
    return h.hexdigest()


def possible_influence(network, visual, audio, motor):
    """Exact nonzero-source ancestry, including the original circular spread.

    This is reachability of changed raw adapter cells, not threshold firing.
    """
    parts=[]
    for values in (visual,audio,motor):
        values=np.asarray(values,bool)
        sources=(np.arange(len(values))[:,None]+
                 np.arange(-network.扩散宽度,network.扩散宽度+1)) % len(values)
        parts.append(np.any(values[sources],axis=1))
    current=np.concatenate(parts)
    for sources,weights in zip(network.来源,network.权重):
        current=np.any(current[sources] & (weights>0),axis=1)
    return current


def run(config_path, seeds):
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    protected = [ROOT/'前额叶区_prefrontal.py', HERE/'sensory_adapter.py', HERE/'brain.py', HERE/'viewer.py',
                 HERE/'results'/'living_original.npz']
    hashes_before = {str(p):file_hash(p) for p in protected}
    poses = list(itertools.product(((3.,3.),(5.,4.),(4.,5.)),(-.8,0.,.8),(-.3,0.,.3),(1.2,2.)))
    reports = []
    for seed in seeds:
        adapter = SensoryAdapter(seed=seed, threshold=config['sensory_threshold'],
            visual_threshold=config['visual_threshold'], audio_threshold=config['audio_threshold'],
            motor_threshold=config['motor_threshold'], motor_levels=config['motor_levels'],
            motor_reverse_window=config['motor_reverse_window'], contrast_to_silence=config['contrast_to_silence'])
        network = original_forward(sum(adapter.widths.values()), seed, config)
        fixed_before = born_hash(network)
        graph = reachability(network, adapter)
        zero_v = np.zeros(adapter.widths['visual'],bool)
        zero_a = np.zeros(adapter.widths['audio'],bool)
        zero_m = np.zeros(adapter.widths['motor'],bool)
        rows, examples = [], []
        all_conjunction = {name:set() for name in ('motor_clamped_zero','actual_rest_motor_code')}
        all_interaction = {name:set() for name in all_conjunction}
        all_specific = {name:set() for name in all_conjunction}
        all_pattern_counts = {name:np.zeros(16,int) for name in all_conjunction}
        for pose_id, (position, heading, bearing, distance) in enumerate(poses):
            env = ExperienceEnvironment(seed=seed)
            env.reset(start=(*position,heading))
            object_position = np.array(position)+distance*np.array([np.cos(heading+bearing),np.sin(heading+bearing)])
            encoded_pairs, packets, rendered = [], [], []
            for color, tone in itertools.product(COLORS,TONES):
                packet = env.setup_pairing(color,tone,duration=.8,presentation_position=object_position)
                assert set(packet)=={'observation','waveform'}
                assert set(packet['observation'])=={'ray_distances','ray_colors','velocity','angular_velocity','pain','touch'}
                visible = env.visible_beacon_rays()
                assert visible[color] and not visible[COLORS[1-COLORS.index(color)]]
                encoded_pairs.append(adapter.encode(packet,np.zeros(4)))
                packets.append(packet)
                rendered.append(visible[color])
            # Geometry/body are equal for all four alternatives. Color alone
            # changes RGB; tone alone changes PCM, with no metadata in packets.
            for packet in packets[1:]:
                for key in ('ray_distances','velocity','angular_velocity','pain','touch'):
                    assert np.array_equal(packet['observation'][key],packets[0]['observation'][key])
            for a,b in ((0,1),(2,3)):
                assert np.array_equal(packets[a]['observation']['ray_colors'],packets[b]['observation']['ray_colors'])
                assert np.array_equal(encoded_pairs[a]['visual'],encoded_pairs[b]['visual'])
            for a,b in ((0,2),(1,3)):
                assert np.array_equal(packets[a]['waveform'],packets[b]['waveform'])
                assert np.array_equal(encoded_pairs[a]['audio'],encoded_pairs[b]['audio'])
            silent=adapter.encode(env.stop_tone(),np.zeros(4))
            assert not silent['audio'].any()
            changed_color=encoded_pairs[0]['visual'] ^ encoded_pairs[2]['visual']
            changed_tone=encoded_pairs[0]['audio'] ^ encoded_pairs[1]['audio']
            color_reachable=possible_influence(network,changed_color,zero_a,zero_m)
            tone_reachable=possible_influence(network,zero_v,changed_tone,zero_m)
            row=dict(pose_id=pose_id, position=list(position),heading=heading,bearing=bearing,distance=distance,
                visible_rays=rendered, sensory_code_hashes=[sensory_hash(e) for e in encoded_pairs],
                adapter_visual_color_hamming=int(changed_color.sum()),
                adapter_audio_tone_hamming=int(changed_tone.sum()),
                output_ids_with_both_changed_color_and_changed_tone_ancestors=ids(color_reachable & tone_reachable),
                conditions={})
            for condition in all_conjunction:
                motor=zero_m if condition=='motor_clamped_zero' else encoded_pairs[0]['motor']
                baseline=project(network,zero_v,zero_a,motor)
                fulls, combo_rows = [], []
                for combo,(encoded,(color,tone)) in enumerate(zip(encoded_pairs,itertools.product(COLORS,TONES))):
                    v,a=encoded['visual'],encoded['audio']
                    full=project(network,v,a,motor)
                    v_only=project(network,v,zero_a,motor)
                    a_only=project(network,zero_v,a,motor)
                    conjunction=full & ~v_only & ~a_only & ~baseline
                    assert np.all(full | ~(v_only | a_only | baseline))
                    all_conjunction[condition].update(ids(conjunction))
                    fulls.append(full)
                    combo_rows.append(dict(color=color,tone_hz=tone,
                        visual_active=int(v.sum()),audio_active=int(a.sum()),motor_active=int(motor.sum()),
                        full_active=int(full.sum()),visual_only_active=int(v_only.sum()),
                        audio_only_active=int(a_only.sum()),baseline_active=int(baseline.sum()),
                        conjunction_ids=ids(conjunction),
                        exactly_single_modality_union=bool(np.array_equal(full,v_only | a_only | baseline))))
                outputs=np.asarray(fulls)
                contrast=outputs[0].astype(int)-outputs[1].astype(int)-outputs[2].astype(int)+outputs[3].astype(int)
                interaction=contrast!=0
                specific=outputs.sum(axis=0)==1
                all_interaction[condition].update(ids(interaction))
                all_specific[condition].update(ids(specific))
                patterns=(outputs.T*np.array([1,2,4,8])).sum(axis=1)
                counts=np.bincount(patterns,minlength=16)
                all_pattern_counts[condition]+=counts
                row['conditions'][condition]=dict(combos=combo_rows,
                    all_four_full_codes_distinct=len({o.tobytes() for o in outputs})==4,
                    pfc_color_hamming_by_tone=[int(np.count_nonzero(outputs[0]^outputs[2])),
                                               int(np.count_nonzero(outputs[1]^outputs[3]))],
                    exactly_one_combination_ids=ids(specific),factorial_interaction_ids=ids(interaction),
                    response_pattern_counts=counts.tolist())
                if pose_id==len(poses)//2 and condition=='motor_clamped_zero':
                    probe=encoded_pairs[0]
                    full,currents=project(network,probe['visual'],probe['audio'],motor,trace=True)
                    boundary=adapter.widths['visual']
                    traces=[]
                    for cell in (boundary-20,boundary-5,boundary-1,boundary,boundary+5,boundary+19):
                        item=source_trace(network,cell,adapter)
                        item.update(active=bool(full[cell]),output_current=float(currents[-1][cell]),
                            output_threshold=float(network.阈值[-1][cell]),
                            hidden_active_ids=network.来源[-1][cell][network.普通层激活[0][network.来源[-1][cell]]].tolist())
                        traces.append(item)
                    examples.append(dict(pose_id=pose_id,color='red',tone_hz=220,
                        visual_active_ids=ids(probe['visual']),audio_active_ids=ids(probe['audio']),
                        full_pfc_active_ids=ids(full),boundary_traces=traces))
            rows.append(row)
        assert born_hash(network)==fixed_before
        assert adapter.motor_learning_steps==0 and adapter.motor_reverse.条数()==0
        reports.append(dict(seed=seed,adapter_birth_hash=adapter.birth_hash,pfc_born_hash=fixed_before,
            graph=graph,rows=rows,examples=examples,
            input_audit=dict(adapter_red_blue_collision_poses=sum(r['adapter_visual_color_hamming']==0 for r in rows),
                adapter_tone_collision_poses=sum(r['adapter_audio_tone_hamming']==0 for r in rows),
                poses_with_any_cell_reachable_from_both_changed_inputs=sum(bool(r['output_ids_with_both_changed_color_and_changed_tone_ancestors']) for r in rows)),
            summaries={condition:dict(unique_conjunction_cells=sorted(all_conjunction[condition]),
                unique_factorial_interaction_cells=sorted(all_interaction[condition]),
                unique_single_combination_cells=sorted(all_specific[condition]),
                total_cell_response_pattern_counts=all_pattern_counts[condition].tolist(),
                union_equal_cases=sum(c['exactly_single_modality_union'] for r in rows
                                      for c in r['conditions'][condition]['combos']),
                total_cases=len(rows)*4,
                four_distinct_code_poses=sum(r['conditions'][condition]['all_four_full_codes_distinct'] for r in rows))
                for condition in all_conjunction}))
    hashes_after={str(p):file_hash(p) for p in protected}
    assert hashes_before==hashes_after, 'Protected source or checkpoint changed concurrently; audit did not write it'
    return dict(protocol=dict(config_path=str(config_path),config=config,birth_seeds=seeds,
        poses_per_birth=len(poses),rendered_combinations_per_pose=4,
        combination_order=['red/220','red/660','blue/220','blue/660'],
        neural_controls=['visual+audio+fixed motor','visual+fixed motor','audio+fixed motor','fixed motor'],
        conjunction_definition='full active AND visual-only inactive AND audio-only inactive AND baseline inactive',
        factorial_interaction='f(red,220)-f(red,660)-f(blue,220)+f(blue,660) != 0',
        learning_updates=0,teacher_actions=0,rewards=0,executed_actions=0,
        scope='Born forward network only. Actual render/PCM; single-modality removal is a neural clamp.'),
        seeds=reports,protected_hashes_unchanged=hashes_before,seconds=time.perf_counter()-started,
        limitations=[
            'No conjunction firing here does not mean the population lacks both stimuli: separate codes can coexist.',
            'This is not a test of learned recurrent PFC, hippocampal temporal recall, movement selection or language.',
            'History has no separate input namespace in the born network: brain.py projects current and remembered modalities separately, then sums PFC currents.',
            'More experience cannot change this fixed source ancestry; it can change separate original recurrent/index/motor connections.',
            '31 RGB/depth rays and generic FFT bands do not provide a text-token or book-document interface. File import is not knowledge acquisition.'
        ],
        minimal_unimplemented_born_connection_candidate={
            'change':'Keep width, two layers, 21 inputs, original positive birth weights and thresholds; replace one local patch per target with three independent generic patches of seven source cells, two with uniformly sampled remote centers.',
            'purpose':'Preserve local coactivation while allowing multiple distant sensory clusters to enter the same threshold cell. No color, tone, rule or answer labels choose sources.',
            'requirements':'A new disclosed birth schema/hash and matching original reciprocal-return source tables; retain frozen old models. Re-test sparsity, combination selectivity, nearby stimuli, motor recovery and sustained online stability before use.',
            'status':'Suggestion only. No source table, model, parameter or checkpoint was modified by this audit.'
        },
        existing_sensor_experience_options=[
            'Continuously move the real four-muscle body across varied positions, headings, object distances, occlusion, wall geometry and pigment, preserving the shared time sequence.',
            'Present arbitrary PCM frequencies, mixtures, envelopes and temporal order while varying physical visual scenes independently; retain silent and unseen controls.',
            'Record actual pre-action sensors, executed muscles and post-action sensors; validate later recall and action consequences on reserved environments, with no labels entering packets.'
        ])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'results'/'index_margin_validation_config.json')
    parser.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    parser.add_argument('--output',type=Path,default=HERE/'results'/'pfc_crossmodal_audit.json')
    args=parser.parse_args()
    report=run(args.config,args.seeds)
    report['script_sha256']=file_hash(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),seconds=report['seconds'],
        summaries=[dict(seed=r['seed'],input_audit=r['input_audit'],conditions=r['summaries']) for r in report['seeds']]),ensure_ascii=True))


if __name__=='__main__':
    main()
