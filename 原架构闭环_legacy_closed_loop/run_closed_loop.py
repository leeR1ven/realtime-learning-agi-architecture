"""Real co-experience -> original hippocampal completion -> PFC -> muscles.

This experiment supplies stimuli, never routes or muscle answers. Scores and
color names stay in this file, outside the brain's sensor-only interface.
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

HERE = Path(__file__).resolve().parent
COLORS = ('red', 'green', 'blue')
TONES = (220., 440., 660.)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def compact(info):
    fields = ('frame', 'index_time', 'clock', 'time_active_count', 'sound_rms',
        'sound_cue_cells', 'external_pfc_cells', 'thought_cells', 'auditory_recall_time',
        'auditory_recall_ratio', 'initially_recalled_visual_cells',
        'initially_recalled_motor_cells', 'final_recall_time', 'final_recall_ratio',
        'final_visual_cells', 'motor_command_cells', 'remembered_muscles', 'muscles',
        'source', 'connections', 'actual_motor_in_memory', 'learning')
    result = {name: plain(info[name]) for name in fields}
    result['microsteps'] = [{k: v for k, v in row.items() if np.isscalar(v)} for row in info['microsteps']]
    return result


def present(env, color, frequency, duration):
    # Experimenter presents a visible object. This changes neither the body
    # nor its action. It is not a navigation target-location input to the brain.
    heading = np.array([np.cos(env.world.heading), np.sin(env.world.heading)])
    position = np.clip(env.world.position+1.4*heading, (.31, .31), (9.69, 7.69))
    return env.setup_pairing(color, frequency, duration=duration,
                             presentation_position=position)


def train(brain, env, *, repeats=4, permutation=(0, 1, 2), hold_frames=8):
    plan = [(COLORS[color], TONES[permutation[color]])
            for repeat in range(repeats) for color in ((repeat+np.arange(3)) % 3)]
    catalog, trace, timings = {}, [], []
    phase = 0
    color, frequency = plan[phase]
    packet = present(env, color, frequency, hold_frames*.1)

    def remember(info, label, tone, visible):
        catalog[str(info['index_time'])] = {'color': label, 'frequency': tone,
            'visually_present': bool(visible), 'physical_time': float(env.world.time),
            'actual_visual_cells': np.flatnonzero(brain.runtime.visual_memory.激活).tolist(),
            'actual_motor_cells': np.flatnonzero(brain.runtime.motor_memory.激活).tolist(),
            'actual_muscles': brain.last_executed.tolist()}

    for frame in range(len(plan)*hold_frames):
        begin = time.perf_counter()
        muscles, info = brain.observe_then_act(packet, learning=True)
        visible = bool(env.visible_beacon_rays()[color])
        if str(info['index_time']) not in catalog:
            remember(info, color, frequency, visible)
        trace.append({'physical_time': float(env.world.time), 'color': color,
            'frequency': frequency, 'visible': visible, 'info': compact(info)})
        packet = env.step(muscles)
        if (frame+1) % hold_frames == 0 and phase+1 < len(plan):
            phase += 1
            color, frequency = plan[phase]
            packet = present(env, color, frequency, hold_frames*.1)
        outcome = brain.observe_outcome(packet, muscles)
        remember(outcome, color, frequency, bool(env.visible_beacon_rays()[color]))
        timings.append((time.perf_counter()-begin)*1000)
        if (frame+1) % hold_frames == 0:
            print(f'PRESENTATION {(frame+1)//hold_frames}/{len(plan)} '
                f'clock={brain.runtime.clock.当前时间} '
                f'pfc_edges={brain.runtime.pfc.联想.条数()} '
                f'p95={np.percentile(timings,95):.1f}ms', flush=True)
    return {'presentation_schedule': plan, 'frames': len(trace), 'catalog': catalog,
        'trace': trace, 'p50_ms': float(np.percentile(timings,50)),
        'p95_ms': float(np.percentile(timings,95)), 'runtime_seconds': sum(timings)/1000,
        'teacher_actions': 0, 'reward': 0, 'clock': int(brain.runtime.clock.当前时间),
        'single_active_time_cell': int(brain.runtime.clock.时间激活.sum()),
        'connections': brain.runtime.connection_counts(),
        'motor_reverse_connections': brain.adapter.motor_reverse.条数()}


def completion_score(info, catalog, expected_color):
    initial = catalog.get(str(info['auditory_recall_time']))
    final = catalog.get(str(info['final_recall_time']))
    recalled_cells = set(info['initially_recalled_visual_cells'])
    original_cells = set(initial['actual_visual_cells']) if initial else set()
    return {'initial_time': info['auditory_recall_time'],
        'initial_color': None if initial is None else initial['color'],
        'initial_visible': False if initial is None else initial['visually_present'],
        'initial_same_stored_visual': bool(initial and recalled_cells == original_cells),
        'correct_initial_color': bool(initial and initial['color'] == expected_color and initial['visually_present']),
        'final_time': info['final_recall_time'],
        'final_color': None if final is None else final['color'],
        'correct_final_color': bool(final and final['color'] == expected_color and final['visually_present'])}


def evaluate(snapshot, env, catalog, permutation, *, steps=16):
    results = []
    for color in range(3):
        for condition in ('full', 'no_auditory_index', 'no_recalled_drive', 'no_pfc_recurrence', 'no_motor_recall'):
            brain = OriginalBrain.from_snapshot(snapshot)
            # Remove a previously computed action, not knowledge or thought.
            brain.prepared = None
            brain.pending = None
            brain.flags['exploration'] = False
            brain.flags['reflex'] = False
            if condition != 'full':
                brain.flags[condition.removeprefix('no_')] = False
            e = copy.deepcopy(env)
            e.hide_beacons()
            packet = e.emit_tone(TONES[permutation[color]], duration=.6)
            before_counts = brain.runtime.connection_counts()
            before_motor = brain.adapter.motor_reverse.矩阵.copy()
            trace, timings = [], []
            start = e.world.position.copy()
            for frame in range(steps):
                begin = time.perf_counter()
                muscles, info = brain.observe_then_act(packet, learning=False, trace=(frame==0))
                score = completion_score(info, catalog, COLORS[color])
                trace.append({'time': float(e.world.time), 'position': e.world.position.tolist(),
                    'heading': float(e.world.heading), 'score': score, 'info': compact(info)})
                packet = e.step(muscles)
                brain.observe_outcome(packet, muscles, terminal=frame==steps-1)
                timings.append((time.perf_counter()-begin)*1000)
            assert brain.runtime.connection_counts() == before_counts
            assert np.array_equal(brain.adapter.motor_reverse.矩阵, before_motor)
            result = {'color': COLORS[color], 'frequency': TONES[permutation[color]],
                'condition': condition, 'no_beacons_visible': all(not x for x in e.visible_beacon_rays().values()),
                'learning': False, 'reward': 0, 'trace': trace,
                'first_completion': trace[0]['score'],
                'movement_distance': float(np.linalg.norm(e.world.position-start)),
                'heading_change': float(e.world.heading-env.world.heading),
                'motor_recall_frames': sum(r['info']['source']=='pfc_time_motor_recall' for r in trace),
                'p95_ms': float(np.percentile(timings,95))}
            results.append(result)
            print(f'PROBE {COLORS[color]} {condition} first={result["first_completion"]["initial_color"]} '
                  f'final={result["first_completion"]["final_color"]} '
                  f'move={result["movement_distance"]:.3f}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--hold-frames', type=int, default=8)
    parser.add_argument('--probe-steps', type=int, default=16)
    parser.add_argument('--permuted', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE/'results'/'closed_loop.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    brain = OriginalBrain(config)
    env = ExperienceEnvironment(seed=20260909)
    permutation = (2, 0, 1) if args.permuted else (0, 1, 2)
    report = {'status': 'running', 'config': vars(brain.config), 'birth_hash': brain.birth_hash,
        'scope': 'Original shared-time AV completion and physical motor readout; not navigation or AGI acceptance',
        'permutation': permutation, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.glob('*.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report['training'] = train(brain, env, repeats=args.repeats, permutation=permutation, hold_frames=args.hold_frames)
    checkpoint = args.output.with_suffix('.npz')
    brain.save(checkpoint)
    snapshot = brain.snapshot()
    report['checkpoint'] = str(checkpoint.resolve())
    report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2), encoding='utf-8')
    report['evaluation'] = evaluate(snapshot, env, report['training']['catalog'], permutation, steps=args.probe_steps)
    report['status'] = 'complete'
    report['summary'] = {condition: {'correct_initial': sum(r['first_completion']['correct_initial_color'] for r in report['evaluation'] if r['condition']==condition),
        'correct_final': sum(r['first_completion']['correct_final_color'] for r in report['evaluation'] if r['condition']==condition),
        'trials': sum(r['condition']==condition for r in report['evaluation'])}
        for condition in ('full','no_auditory_index','no_recalled_drive','no_pfc_recurrence','no_motor_recall')}
    args.output.write_text(json.dumps(plain(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
