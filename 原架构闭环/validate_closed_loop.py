"""Held-out actual-experience checks, without changing a neuron or choosing parameters.

Labels only score memory contents here. Births, permutations and longer exposure
are fixed in advance; failures are retained. Physical drift is never navigation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from brain import OriginalBrain
from experience_environment import ExperienceEnvironment
from run_closed_loop import COLORS, TONES, plain, train
from dynamic_calibration import frozen_digest, hashes

HERE = Path(__file__).resolve().parent


def score_content(info, catalog, expected):
    result = {}
    for prefix, time_key, cells_key in (
        ('initial', 'auditory_recall_time', 'initially_recalled_visual_cells'),
        ('final', 'final_recall_time', 'final_visual_cells')):
        row = catalog.get(str(info[time_key]))
        cells = set(map(int, info[cells_key]))
        original = set(row['actual_visual_cells']) if row else set()
        exact = bool(cells and original and cells == original)
        right = bool(row and row['color'] == expected and row['visually_present'])
        result.update({prefix+'_time': info[time_key],
                       prefix+'_color': None if row is None else row['color'],
                       prefix+'_exact_nonempty_visual': exact,
                       prefix+'_correct_content': right and exact})
    return result


def probe(snapshot, environment, catalog, *, frequency, expected,
          condition='full', amplitude=.45, steps=16, switch_frequency=None,
          switch_expected=None):
    brain = OriginalBrain.from_snapshot(snapshot)
    brain.prepared = None
    brain.pending = None
    brain.flags.update(exploration=False, reflex=False)
    if condition != 'full':
        brain.flags[condition.removeprefix('no_')] = False
    env = copy.deepcopy(environment)
    env.hide_beacons()
    packet = env.emit_tone(frequency, amplitude=amplitude)
    start = env.world.position.copy()
    original_weights = frozen_digest(brain)
    rows, elapsed = [], []
    for frame in range(steps):
        begin = time.perf_counter()
        muscles, info = brain.observe_then_act(packet, learning=False)
        target = switch_expected if switch_frequency is not None and frame >= 8 else expected
        rows.append(dict(frame=frame, physical_time=float(env.world.time),
            position=env.world.position.tolist(), expected=target,
            sound_rms=info['sound_rms'], muscles=muscles.tolist(),
            motor_cells=plain(info['motor_command_cells']), source=info['source'],
            thought_cells=plain(info['thought_cells']),
            initially_recalled_visual_cells=plain(info['initially_recalled_visual_cells']),
            final_visual_cells=plain(info['final_visual_cells']),
            score=score_content(info, catalog, target)))
        packet = env.step(muscles)
        if switch_frequency is not None and frame == 7:
            packet = env.emit_tone(switch_frequency, amplitude=amplitude)
        brain.observe_outcome(packet, muscles, terminal=frame == steps-1)
        elapsed.append((time.perf_counter()-begin)*1000)
    assert frozen_digest(brain) == original_weights
    assert not any(env.visible_beacon_rays().values())
    silent = [row for row in rows if row['sound_rms'] < .02]
    return dict(frequency=frequency, amplitude=amplitude, expected=expected,
        condition=condition, switch_frequency=switch_frequency,
        switch_expected=switch_expected, initial=rows[0]['score'],
        frames=rows, final_correct_frames=sum(row['score']['final_correct_content'] for row in rows),
        silent_correct_frames=sum(row['score']['final_correct_content'] for row in silent),
        silent_frames=len(silent),
        displacement_including_initial_inertia=float(np.linalg.norm(env.world.position-start)),
        p95_ms=float(np.percentile(elapsed,95)), frozen_weights=True)


def summarize(rows):
    return dict(trials=len(rows),
        initial_correct=sum(row['initial']['initial_correct_content'] for row in rows),
        first_final_correct=sum(row['initial']['final_correct_content'] for row in rows),
        final_correct_frames=sum(row['final_correct_frames'] for row in rows),
        frames=sum(len(row['frames']) for row in rows),
        silent_correct_frames=sum(row['silent_correct_frames'] for row in rows),
        silent_frames=sum(row['silent_frames'] for row in rows),
        p95_ms_max=max((row['p95_ms'] for row in rows), default=0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'heldout_validation.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    provenance = hashes()
    report = dict(status='running', config=config, source_sha256=provenance,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='Held-out birth 2028 with standard and cyclically permuted learned pairs; birth 2026 with six repeats per color; no parameter selection from these checks.',
        scoring='Correct visible historical event AND exact nonempty recovered visual neuron set; no semantic labels enter the brain.',
        limitations='Motion includes inherited inertia; this protocol does not establish goal-directed navigation, route switching, language or AGI.',
        cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, seed, permutation, repeats in (
        ('heldout_birth', 2028, (0,1,2), 2),
        ('heldout_permuted', 2028, (2,0,1), 2),
        ('longer_experience', 2026, (0,1,2), 6)):
        brain = OriginalBrain(dict(config, seed=seed))
        env = ExperienceEnvironment(seed=20260909)
        training = train(brain, env, repeats=repeats, permutation=permutation, hold_frames=8)
        snapshot = brain.snapshot()
        checkpoint = args.output.with_name(args.output.stem+'_'+name+'.npz')
        if checkpoint.exists():
            raise FileExistsError('Use a new output name to preserve this experiment: '+str(checkpoint))
        brain.save(checkpoint)
        case = dict(name=name, seed=seed, permutation=permutation, training=training,
            checkpoint=str(checkpoint), birth_hash=brain.birth_hash, probes=[], amplitude_probes=[],
            unknown_sound_probes=[], newborn_probes=[], cue_switches=[])
        for condition in ('full','no_auditory_index','no_recalled_drive',
                          'no_pfc_recurrence','no_disinhibition','no_motor_recall'):
            for color_index, color in enumerate(COLORS):
                case['probes'].append(probe(snapshot, env, training['catalog'],
                    frequency=TONES[permutation[color_index]], expected=color, condition=condition))
        if name == 'heldout_birth':
            for amplitude in (.3, .6):
                for color_index, color in enumerate(COLORS):
                    case['amplitude_probes'].append(probe(snapshot, env, training['catalog'],
                        frequency=TONES[permutation[color_index]], expected=color,
                        amplitude=amplitude, steps=1))
            for frequency in (777., 2047.):
                case['unknown_sound_probes'].append(probe(snapshot, env, training['catalog'],
                    frequency=frequency, expected=None, steps=1))
            newborn = OriginalBrain(dict(config, seed=seed)).snapshot()
            for color_index, color in enumerate(COLORS):
                case['newborn_probes'].append(probe(newborn, env, {},
                    frequency=TONES[color_index], expected=color, steps=1))
            for source, dest in ((0,1),(1,2),(2,0)):
                case['cue_switches'].append(probe(snapshot, env, training['catalog'],
                    frequency=TONES[source], expected=COLORS[source],
                    switch_frequency=TONES[dest], switch_expected=COLORS[dest], steps=16))
        case['summary'] = {condition:summarize([r for r in case['probes'] if r['condition']==condition])
            for condition in dict.fromkeys(r['condition'] for r in case['probes'])}
        assert hashes() == provenance
        report['cases'].append(case)
        args.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf-8')
        print('VALIDATION '+name+' '+json.dumps(case['summary']), flush=True)
    report['status'] = 'complete'
    args.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
