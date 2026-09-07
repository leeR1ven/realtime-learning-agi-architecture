"""Bounded calibration of parameters in the unchanged original PFC equations.

Nine predeclared settings, two actual presentations of each of three colors,
then same-state sound-only physical probes. This changes no model equation or
memory system. Offline labels score recalled time entries and never enter brain.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from brain import BrainConfig, OriginalBrain
from experience_environment import ExperienceEnvironment
from run_closed_loop import COLORS, TONES, compact, completion_score, evaluate, plain, train


HERE = Path(__file__).resolve().parent
RATES = (.0001, .0005, .001)
FEEDBACK = (.03, .06, .12)
DEPENDENCIES = ('brain.py', 'sensory_adapter.py', 'original_runtime.py',
                'experience_environment.py', 'run_closed_loop.py', 'checkpoint.py')


def hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in DEPENDENCIES}


def configuration(recommended, seed, rate, feedback):
    allowed = {field.name for field in fields(BrainConfig)}
    config = {'seed': seed, 'pfc_hebb_increment': rate,
              'pfc_feedback_inhibition': feedback, 'pfc_target_high': 350}
    for key, value in recommended['adapter_kwargs'].items():
        if key == 'threshold':
            key = 'sensory_threshold'
        if key == 'motor_once':
            assert value is True, 'OriginalBrain uses original one-shot reverse Hebb'
            continue
        if key not in allowed:
            raise ValueError('Recommended adapter field missing from BrainConfig: '+key)
        config[key] = value
    config.update(recommended['pfc_kwargs'])
    return config


def frozen_digest(brain):
    digest = hashlib.sha256()
    tables = [table for memory in brain.runtime.memories.values()
              for table in (memory.特征到时间, memory.时间到特征)]
    tables += [brain.runtime.pfc.联想, brain.runtime.pfc.去抑]
    for table in tables:
        digest.update(str((table.连接条数, table.维护次数, table.上次衰减维护)).encode())
        for source, row in table.表.items():
            digest.update(str(source).encode())
            digest.update(np.asarray(list(row.items()), np.float64).tobytes())
    digest.update(json.dumps({t: sorted(cells) for t, cells in brain.runtime.index.时间到输出.items()}).encode())
    digest.update(brain.adapter.motor_reverse.矩阵.tobytes())
    return digest.hexdigest()


def current_summary(info, brain):
    index_time = info['auditory_recall_time']
    target = np.asarray(sorted(brain.runtime.index.时间到输出.get(index_time, ())), np.int64)
    result = []
    for micro in info['microsteps']:
        row = {key: plain(value) for key, value in micro.items() if np.isscalar(value)}
        for name in ('excitatory_current', 'external_current', 'inhibitory_site_current',
                     'disinhibitory_site_current', 'net_inhibitory_site_current'):
            if name in micro:
                values = np.asarray(micro[name])
                row[name] = {'max': float(values.max(initial=0.)),
                             'p95': float(np.percentile(values, 95)), 'mean': float(values.mean())}
                if len(target):
                    selected = target//brain.runtime.pfc.每段 if 'site_' in name else target
                    row[name]['at_initial_index_cells_mean'] = float(values[selected].mean())
                    row[name]['at_initial_index_cells_max'] = float(values[selected].max(initial=0.))
        result.append(row)
    return result


def probe_subset(snapshot, environment, catalog, *, conditions=('full',), steps=8):
    """Same cloning/physics procedure as run_closed_loop.evaluate, selected arms.

    Full current diagnostics are requested on outcome as well as first action,
    so cached following actions contain measured currents during silence.
    """
    results = []
    for color_index, color in enumerate(COLORS):
        for condition in conditions:
            brain = OriginalBrain.from_snapshot(snapshot)
            # Discard only the pending action produced before the new stimulus;
            # retain learned connections, all PFC dynamics, RNG and body state.
            brain.prepared = None
            brain.pending = None
            brain.flags['exploration'] = False
            brain.flags['reflex'] = False
            if condition != 'full':
                brain.flags[condition.removeprefix('no_')] = False
            env = copy.deepcopy(environment)
            env.hide_beacons()
            packet = env.emit_tone(TONES[color_index], duration=.6)
            before = frozen_digest(brain)
            rows, timings = [], []
            start_position = env.world.position.copy()
            for frame in range(steps):
                begun = time.perf_counter()
                muscles, info = brain.observe_then_act(packet, learning=False, trace=True)
                score = completion_score(info, catalog, color)
                entry = catalog.get(str(info['final_recall_time']))
                score['final_same_stored_visual'] = bool(entry and
                    set(info['final_visual_cells']) == set(entry['actual_visual_cells']))
                rows.append({'frame': frame, 'physical_time': float(env.world.time),
                    'silent': bool(info['sound_rms'] < brain.config.auditory_onset_rms),
                    'sound_rms': float(info['sound_rms']), 'score': score,
                    'thought_active': len(info['thought_cells']),
                    'muscles': muscles.tolist(), 'source': info['source'],
                    'currents': current_summary(info, brain)})
                packet = env.step(muscles)
                brain.observe_outcome(packet, muscles, terminal=frame==steps-1, trace=True)
                timings.append(1000*(time.perf_counter()-begun))
            assert frozen_digest(brain) == before
            assert all(not visible for visible in env.visible_beacon_rays().values())
            results.append({'color': color, 'condition': condition, 'learning': False,
                'reward': 0, 'no_beacons_visible': True, 'first_completion': rows[0]['score'],
                'trace': rows, 'p95_ms': float(np.percentile(timings, 95)),
                'movement_distance': float(np.linalg.norm(env.world.position-start_position)),
                'frozen_weights_index_maintenance_and_reverse_motor_unchanged': True})
    return results


def summarize(results):
    by_condition = {}
    for condition in dict.fromkeys(row['condition'] for row in results):
        trials = [row for row in results if row['condition'] == condition]
        frames = []
        for trial in trials:
            for item in trial['trace']:
                if 'silent' in item:
                    silent = item['silent']
                else:
                    silent = item['info']['sound_rms'] < .02
                frames.append((silent, item['score']))
        silent_scores = [score for silent, score in frames if silent]
        first = [row['first_completion'] for row in trials]
        by_condition[condition] = {'trials': len(trials),
            'correct_initial': sum(row['correct_initial_color'] for row in first),
            'correct_first_final': sum(row['correct_final_color'] for row in first),
            'initial_exact_stored_visual': sum(row['initial_same_stored_visual'] for row in first),
            'correct_final_frames': sum(row['correct_final_color'] for _, row in frames),
            'total_frames': len(frames), 'silent_frames': len(silent_scores),
            'correct_silent_final_frames': sum(row['correct_final_color'] for row in silent_scores),
            'silent_wrong_color_frames': sum(row['final_color'] is not None and not row['correct_final_color'] for row in silent_scores),
            'silent_no_recall_frames': sum(row['final_color'] is None for row in silent_scores),
            'maximum_trial_p95_ms': max(row['p95_ms'] for row in trials),
            'first_colors': [{'expected': row['color'], 'initial': row['first_completion']['initial_color'],
                              'final': row['first_completion']['final_color']} for row in trials]}
    return by_condition


def rank(row):
    score = row['summary']['full']
    return (score['correct_initial'], score['correct_first_final'],
            score['correct_silent_final_frames'], score['correct_final_frames'],
            -score['silent_wrong_color_frames'])


def train_candidate(config):
    brain = OriginalBrain(config)
    assert brain.width == 5520
    env = ExperienceEnvironment(seed=20260909)
    training = train(brain, env, repeats=2, hold_frames=8)
    assert training['teacher_actions'] == 0 and training['reward'] == 0
    return brain.snapshot(), env, training, brain.birth_hash


def write_report(path, report):
    path.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recommended', type=Path, default=HERE/'results'/'recommended_config.json')
    parser.add_argument('--output', type=Path, default=HERE/'results'/'dynamic_calibration.json')
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True)
    recommended = json.loads(args.recommended.read_text(encoding='utf-8'))
    source_hashes = hashes()
    started = time.perf_counter()
    report = {'status': 'running', 'protocol': {
        'rates': RATES, 'feedback': FEEDBACK, 'target_high': 350,
        'selection_seed': 2026, 'validation_seed': 2027,
        'training_repeats_per_color': 2, 'training_hold_frames': 8,
        'grid_probe_frames_per_color': 8, 'confirmation_probe_frames_per_color': 16,
        'tone_duration_seconds': .6, 'reward': 0, 'teacher_actions': 0,
        'selection_order': ['correct_initial', 'correct_first_final', 'correct_silent_final_frames',
                            'correct_final_frames', 'fewer_wrong_color_silent_frames', 'fixed_grid_order'],
        'evaluation_state': 'Clone complete learned brain, persistent PFC state, RNG and final physical world; clear only pending/precomputed action, disable exploration and reflex.'},
        'source_sha256': source_hashes,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recommended_source': str(args.recommended.resolve()),
        'recommended_sha256': hashlib.sha256(args.recommended.read_bytes()).hexdigest(),
        'recommended': recommended, 'grid': [], 'confirmations': []}
    retained = {}
    for rate in RATES:
        for feedback in FEEDBACK:
            candidate = f'rate_{rate:g}_feedback_{feedback:g}'
            config = configuration(recommended, 2026, rate, feedback)
            snapshot, env, training, birth = train_candidate(config)
            probes = probe_subset(snapshot, env, training['catalog'], steps=8)
            row = {'candidate': candidate, 'config': config, 'birth_hash': birth,
                   'training': training, 'probes': probes, 'summary': summarize(probes)}
            report['grid'].append(row)
            retained[candidate] = (snapshot, env, training)
            top_ids = {r['candidate'] for r in sorted(report['grid'], key=rank, reverse=True)[:2]}
            retained = {key: value for key, value in retained.items() if key in top_ids}
            assert hashes() == source_hashes, 'Dependencies changed during calibration'
            write_report(args.output, report)
            print('GRID '+candidate+' '+json.dumps(row['summary']['full']), flush=True)
    selected = sorted(report['grid'], key=rank, reverse=True)[:2]
    report['selected_candidates'] = [row['candidate'] for row in selected]
    for selected_row in selected:
        for seed in (2026, 2027):
            config = dict(selected_row['config'], seed=seed)
            if seed == 2026:
                snapshot, env, training = retained[selected_row['candidate']]
            else:
                snapshot, env, training, _ = train_candidate(config)
            # Call the original complete evaluator unchanged, including its
            # selective no-E arm which retains previous DI/local inhibition.
            baseline = evaluate(snapshot, env, training['catalog'], (0, 1, 2), steps=16)
            # Add the existing no-DI flag and measured-current full/no-E arms.
            detail = probe_subset(snapshot, env, training['catalog'],
                                  conditions=('full', 'no_pfc_recurrence', 'no_disinhibition'), steps=16)
            item = {'candidate': selected_row['candidate'], 'seed': seed, 'config': config,
                    'original_evaluate': baseline, 'original_summary': summarize(baseline),
                    'current_diagnostics': detail, 'diagnostic_summary': summarize(detail)}
            if seed == 2027:
                item['training'] = training
            report['confirmations'].append(item)
            assert hashes() == source_hashes, 'Dependencies changed during confirmation'
            write_report(args.output, report)
            print('CONFIRM '+selected_row['candidate']+' seed='+str(seed)+' '+json.dumps(item['diagnostic_summary']), flush=True)
    accepted = []
    for selected_row in selected:
        checks = [row['diagnostic_summary']['full'] for row in report['confirmations']
                  if row['candidate'] == selected_row['candidate']]
        if len(checks) == 2 and all(row['correct_initial'] == 3 and row['correct_first_final'] == 3
            and row['silent_frames'] > 0 and row['correct_silent_final_frames'] == row['silent_frames'] for row in checks):
            accepted.append(selected_row['candidate'])
    best = next((row for row in selected if row['candidate'] in accepted), selected[0])
    config_path = args.output.with_name('dynamic_best_tested_config.json')
    config_path.write_text(json.dumps(best['config'], ensure_ascii=False, indent=2), encoding='utf-8')
    report.update(status='complete', elapsed_seconds=time.perf_counter()-started,
        accepted_candidates=accepted, best_tested_candidate=best['candidate'],
        best_tested_config_path=str(config_path.resolve()),
        accepted_for_stable_recall=bool(accepted),
        conclusion_boundary='Parameter calibration on three presented tone/color associations; not navigation, language, AGI, or proof that recurrence is necessary.')
    write_report(args.output, report)
    lines = ['# Original PFC dynamic calibration', '',
        'Nine fixed parameter settings; original equations unchanged. Each received two actual autonomous presentations of each color with no reward or teacher action.', '',
        '| Hebb increment | Feedback | Initial correct | First final correct | Silent final correct | Max probe p95 ms |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in report['grid']:
        score = row['summary']['full']
        lines.append(f'| {row["config"]["pfc_hebb_increment"]:g} | {row["config"]["pfc_feedback_inhibition"]:g} | '
                     f'{score["correct_initial"]}/3 | {score["correct_first_final"]}/3 | '
                     f'{score["correct_silent_final_frames"]}/{score["silent_frames"]} | {score["maximum_trial_p95_ms"]:.1f} |')
    lines += ['', 'Selected settings were checked with all five unchanged original evaluator conditions, plus independent E and disinhibition current diagnostics, at birth seeds 2026 and 2027.', '',
              'Accepted for correct initial/final and every silent frame at both tested births: '+(', '.join(accepted) if accepted else 'none')+'.',
              'The best-tested config is an experiment candidate; a failed acceptance does not make it a recommended solution. Detailed JSON retains full confirmation traces and excitation/inhibition statistics.']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('COMPLETE '+json.dumps({'elapsed_seconds': report['elapsed_seconds'], 'accepted': accepted,
                                'best_tested': best['candidate']}), flush=True)


if __name__ == '__main__':
    main()
