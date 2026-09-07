"""A separate, single-time original-neuron individual learning its own voice.

Born continuous random exploration executes vocal actuators first; measured
PCM and those actual motor commands are then recorded on the SAME original
time ring. No text/phonemes/word dictionary/desired acoustic answer is supplied
to the brain. The motor channels of this candidate body are vocal, not wheels.
It has NOT been merged with the default navigating individual.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brain import OriginalBrain
from checkpoint import load_tree, save_tree
from vocal_body import VocalBody, mix_at_ear, write_pcm16


def packet(ear_pcm):
    """An ordinary stationary empty room: no instruction or identifying cue."""
    return {'observation': {
        'ray_colors': np.full((31, 3), .45), 'ray_distances': np.full(31, 5.),
        'velocity': np.zeros(2), 'angular_velocity': 0., 'pain': 0., 'touch': np.zeros(4),
    }, 'waveform': np.asarray(ear_pcm, dtype=np.float64).copy()}


def tree_equal(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    return left == right


def content_catalog(runtime):
    return {name: {int(t): set(row) for t, row in memory.时间到特征.表.items()}
            for name, memory in runtime.memories.items()}


def retained_catalog(runtime, catalog):
    return {name: {'old_times': len(rows), 'retained_exact_feature_sets': sum(
        set(runtime.memories[name].时间到特征.查(t)) == features for t, features in rows.items())}
        for name, rows in catalog.items()}


def measured_pitch(pcm):
    if float(np.sqrt(np.mean(pcm * pcm))) < 1e-8:
        return None
    magnitude = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm))))
    frequency = np.fft.rfftfreq(len(pcm), 1. / 8000.)
    indexes = np.flatnonzero((frequency >= 70.) & (frequency <= 350.))
    peaks = indexes[(magnitude[indexes] > magnitude[indexes-1])
                    & (magnitude[indexes] >= magnitude[indexes+1])
                    & (magnitude[indexes] >= .005 * magnitude.max())]
    if not len(peaks):
        return None
    candidates = []
    energy = magnitude * magnitude
    resolution = 8000. / len(pcm)
    for k in peaks:
        local = np.log(np.maximum(magnitude[k-1:k+2], 1e-100))
        denominator = float(local[0] - 2 * local[1] + local[2])
        offset = .5 * float(local[0]-local[2]) / denominator if denominator else 0.
        estimate = float((k + np.clip(offset, -.5, .5)) * resolution)
        distance = np.abs(frequency / estimate - np.rint(frequency / estimate)) * estimate
        coverage = float(energy[distance <= 3 * resolution].sum() / energy.sum())
        candidates.append((estimate, coverage))
    # The largest spacing that explains essentially all harmonic energy avoids
    # a common doubled-period autocorrelation error in strongly resonant voices.
    best = max(coverage for _, coverage in candidates)
    return max(estimate for estimate, coverage in candidates if coverage >= best - .005)


def acoustic_score(reference, response):
    def features(x):
        w = np.hanning(len(x))
        return np.abs(np.fft.rfft(x * w))
    a, b = features(reference), features(response)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    fine_cosine = float(np.dot(a, b) / denominator) if denominator else 0.
    def auditory_bands(x):
        bands = []
        frequencies = np.fft.rfftfreq(800, 1. / 8000.)
        ids = np.minimum((frequencies * 128 / 4000.).astype(int), 127)
        for frame in np.asarray(x).reshape(-1, 800):
            magnitude = np.abs(np.fft.rfft(frame * np.hanning(800)))
            bands.append(np.sqrt(np.bincount(ids, weights=magnitude*magnitude, minlength=128)))
        return np.mean(bands, axis=0)
    first, second = auditory_bands(reference), auditory_bands(response)
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    cosine = float(np.dot(first, second) / norm) if norm else 0.
    original, reproduced = measured_pitch(reference), measured_pitch(response)
    return {
        'spectral_cosine': cosine,
        'fine_fft_spectral_cosine': fine_cosine,
        'reference_measured_fundamental_hz': original,
        'response_measured_fundamental_hz': reproduced,
        'absolute_fundamental_error_hz': None if original is None or reproduced is None else abs(original-reproduced),
        'response_rms': float(np.sqrt(np.mean(response * response))),
    }


def sounding(control):
    body = VocalBody()
    for _ in range(6):
        body.render_frame(control)
    return np.concatenate([body.render_frame(control) for _ in range(4)])


class VocalLearningLife:
    def __init__(self, *, seed=917001, config=None):
        if config is None:
            config = json.loads((HERE / 'results' / 'continuous_default_config.json').read_text(encoding='utf-8'))
        self.brain = OriginalBrain(config)
        self.body = VocalBody()
        self.rng = np.random.default_rng(seed)
        self.frames = 0
        self.last_pcm = np.zeros(800)

    def spontaneous_control(self):
        # Continuous, sound-independent motor exploration. The airflow floor
        # simply makes most exploratory movements audible; there is no target.
        return self.rng.uniform([.6, .05, .05, .05], [.95, .95, .95, .95])

    def step(self, command):
        own_pcm = self.body.render_frame(command)  # Actual action occurs FIRST.
        ear_pcm = mix_at_ear(np.zeros(800), own_pcm)
        self.brain.last_executed = np.asarray(command, float).copy()
        self.brain.adapter.learn_executed(self.brain.last_executed)
        # All production PFC pathways remain active. Born exploration already
        # chose the action just physically executed; only observe its outcome.
        _, info = self.brain._process(packet(ear_pcm), learning=True, choose_action=False)
        self.last_pcm = own_pcm.copy()
        self.frames += 1
        assert self.brain.runtime.clock.时间激活.sum() == 1
        assert self.brain.runtime.frames_recorded == self.frames
        assert self.body.samples_rendered == 800 * self.frames
        assert np.array_equal(info['actual_motor_in_memory'], command)
        return own_pcm, info

    def snapshot(self):
        return {'kind': 'vocal_original_single_time_life_v1', 'brain': self.brain.snapshot(),
                'body': self.body.snapshot(), 'rng': deepcopy(self.rng.bit_generator.state),
                'frames': self.frames, 'last_pcm': self.last_pcm.copy()}

    @classmethod
    def from_snapshot(cls, state):
        fields = {'kind', 'brain', 'body', 'rng', 'frames', 'last_pcm'}
        if not isinstance(state, dict) or set(state) != fields or state.get('kind') != 'vocal_original_single_time_life_v1':
            raise ValueError('not a vocal single-time life')
        if type(state['frames']) is not int or state['frames'] < 0:
            raise ValueError('invalid vocal life frame count')
        obj = cls(config=state['brain']['config'])
        obj.brain = OriginalBrain.from_snapshot(state['brain'])
        obj.body = VocalBody.from_snapshot(state['body'])
        obj.rng.bit_generator.state = deepcopy(state['rng'])
        obj.frames = state['frames']
        obj.last_pcm = np.asarray(state['last_pcm'], float).copy()
        if obj.last_pcm.shape != (800,) or not np.isfinite(obj.last_pcm).all() or np.any(np.abs(obj.last_pcm) > 1.):
            raise ValueError('invalid physical vocal PCM')
        if (obj.frames < 0 or obj.body.samples_rendered != obj.frames * 800
                or obj.brain.runtime.frames_recorded != obj.frames
                or obj.brain.runtime.clock.时间激活.sum() != 1):
            raise ValueError('physical/neural timeline mismatch')
        return obj


def frozen_probe(snapshot, examples, *, label):
    """One independent frozen life per cue; all originate in identical state."""
    rows, wave_examples = [], []
    for example in examples:
        brain = OriginalBrain.from_snapshot(snapshot['brain'])
        before_runtime = brain.runtime.snapshot()
        before_reverse = brain.adapter.motor_reverse.矩阵.copy()
        p = packet(example['ear_pcm'])
        encoded = brain.adapter.encode(p, brain.last_executed)
        native = brain.runtime.recall_from_audio(encoded['audio'], steps=1, threshold=brain.config.memory_threshold)
        native_motor = (native['playback'][0]['motor'] if native['playback'] else np.zeros(brain.adapter.widths['motor'], bool))
        native_control = brain.adapter.decode_motor(native_motor) if native_motor.any() else np.zeros(4)
        _, info = brain._process(p, learning=False, choose_action=False, trace=False)
        initial_motor = info['initially_recalled_motor_cells']
        initial_code = np.zeros(brain.adapter.widths['motor'], bool)
        initial_code[initial_motor] = True
        initial_control = brain.adapter.decode_motor(initial_code) if initial_code.any() else np.zeros(4)
        final_control = info['remembered_muscles']
        response = sounding(final_control)
        initial_response = sounding(initial_control)
        native_response = sounding(native_control)
        after_runtime = brain.runtime.snapshot()
        # Frozen means no synaptic weight/maintenance/index learning. Dynamic
        # PFC activity, inhibition and the one sensory clock may still advance.
        for name in ('visual', 'audio', 'motor'):
            for direction in ('feature_to_time', 'time_to_feature'):
                assert tree_equal(before_runtime['memories'][name][direction], after_runtime['memories'][name][direction])
        for name in ('excitatory', 'disinhibitory'):
            assert tree_equal(before_runtime['pfc'][name], after_runtime['pfc'][name])
        assert tree_equal(before_runtime['index'], after_runtime['index'])
        assert np.array_equal(before_reverse, brain.adapter.motor_reverse.矩阵)
        row = {'cue_id': example['id'], 'cue_kind': label,
               'initial_pfc_index_time': info['auditory_recall_time'],
               'initial_pfc_index_ratio': info['auditory_recall_ratio'],
               'final_pfc_index_time': info['final_recall_time'],
               'final_pfc_index_ratio': info['final_recall_ratio'],
               'final_thought_cells': info['thought_cells'].tolist(),
               'initial_controls': initial_control.tolist(), 'final_controls': final_control.tolist(),
               'native_controls': native_control.tolist(),
               'final_pfc_response': acoustic_score(example['reference'], response),
               'initial_pfc_memory_response': acoustic_score(example['reference'], initial_response),
               'native_audio_memory_response': acoustic_score(example['reference'], native_response)}
        rows.append(row)
        if len(wave_examples) < 3:
            wave_examples.append(np.r_[example['reference'], np.zeros(1600), response, np.zeros(1600)])
    return rows, wave_examples


def summarize(rows):
    result = {'cues': len(rows), 'distinct_final_thought_sets': len({tuple(row['final_thought_cells']) for row in rows})}
    for route in ('final_pfc_response', 'initial_pfc_memory_response', 'native_audio_memory_response'):
        scores = [r[route] for r in rows]
        errors = [s['absolute_fundamental_error_hz'] for s in scores if s['absolute_fundamental_error_hz'] is not None]
        result[route] = {
            'mean_spectral_cosine': float(np.mean([s['spectral_cosine'] for s in scores])),
            'median_fundamental_error_hz': float(np.median(errors)) if errors else None,
            'audible_responses': sum(s['response_rms'] > .001 for s in scores),
            'spectral_cosine_at_least_0_8': sum(s['spectral_cosine'] >= .8 for s in scores),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--hold', type=int, default=12)
    parser.add_argument('--seed', type=int, default=917001)
    parser.add_argument('--prefix', default='vocal_learning_24')
    parser.add_argument('--motor-levels', type=int, default=None)
    args = parser.parse_args()
    if args.episodes < 2 or args.hold < 4:
        raise ValueError('at least two exploratory movements per stage, held for at least four frames')
    directory = HERE / 'results'
    directory.mkdir(exist_ok=True)
    protected = HERE / 'results' / 'living_original.npz'
    default_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
    config = json.loads((HERE / 'results' / 'continuous_default_config.json').read_text(encoding='utf-8'))
    if args.motor_levels is not None:
        config['motor_levels'] = args.motor_levels
    life = VocalLearningLife(seed=args.seed, config=config)
    initial_snapshot = life.snapshot()
    examples, physical = [], []
    stages, probe_rows = {}, {}
    started = time.perf_counter()
    old_catalog = None
    for stage in range(2):
        for number in range(args.episodes):
            command = life.spontaneous_control()
            recent = []
            for _ in range(args.hold):
                pcm, info = life.step(command)
                recent.append(pcm)
                physical.append({'frame': life.frames, 'command': command.copy(), 'pcm': pcm.copy(),
                                 'index_time': info['index_time'], 'clock': info['clock']})
            reference = np.concatenate(recent[-4:])
            examples.append({'id': len(examples), 'command': command.copy(),
                             'ear_pcm': .35 * pcm, 'reference': reference})
        snapshot = life.snapshot()
        rows, waves = frozen_probe(snapshot, examples[:args.episodes], label='old_self_sounds')
        key = 'after_first_experience' if stage == 0 else 'after_new_experience'
        stages[key] = summarize(rows)
        probe_rows[key] = rows
        if stage == 0:
            old_catalog = content_catalog(life.brain.runtime)
        else:
            stages['old_memory_content_retention'] = retained_catalog(life.brain.runtime, old_catalog)
            if waves:
                write_pcm16(directory / (args.prefix + '_heard_then_final_response.wav'), np.concatenate(waves))
        print(json.dumps({'stage': key, 'frames': life.frames, 'summary': stages[key]}, ensure_ascii=False), flush=True)

    final_snapshot = life.snapshot()
    new_rows, _ = frozen_probe(final_snapshot, examples[args.episodes:], label='new_self_sounds')
    stages['new_self_sounds'] = summarize(new_rows)
    probe_rows['new_self_sounds'] = new_rows
    # Perturb controls only in the external sound-producing body. The target
    # parameters and evaluator IDs are NEVER inputs to the tested brain.
    neighbours = []
    test_rng = np.random.default_rng(args.seed + 1)
    for example in examples[:args.episodes]:
        control = np.clip(example['command'] + test_rng.choice([-1., 1.], 4) * .025, 0., 1.)
        assert not any(np.array_equal(control, e['command']) for e in examples)
        reference = sounding(control)
        neighbours.append({'id': example['id'], 'command': control,
                           'ear_pcm': .35 * reference[-800:], 'reference': reference})
    near_rows, _ = frozen_probe(final_snapshot, neighbours, label='unexperienced_nearby_sounds')
    stages['unexperienced_nearby_sounds'] = summarize(near_rows)
    probe_rows['unexperienced_nearby_sounds'] = near_rows
    before_rows, _ = frozen_probe(initial_snapshot, examples[:args.episodes], label='before_any_voice_experience')
    stages['before_any_voice_experience'] = summarize(before_rows)
    probe_rows['before_any_voice_experience'] = before_rows

    checkpoint_path = directory / (args.prefix + '.npz')
    save_tree(checkpoint_path, final_snapshot)
    restored = VocalLearningLife.from_snapshot(load_tree(checkpoint_path))
    assert tree_equal(final_snapshot, restored.snapshot())
    continuation = VocalLearningLife.from_snapshot(final_snapshot)
    for _ in range(8):
        left, right = continuation.spontaneous_control(), restored.spontaneous_control()
        assert np.array_equal(left, right)
        pcm1, _ = continuation.step(left)
        pcm2, _ = restored.step(right)
        assert np.array_equal(pcm1, pcm2)
        assert tree_equal(continuation.snapshot(), restored.snapshot())
    # Independent replay of actual executed controls verifies stored PCM exactly.
    replay_body = VocalBody()
    for event in physical:
        assert np.array_equal(replay_body.render_frame(event['command']), event['pcm'])
    assert tree_equal(replay_body.snapshot(), life.body.snapshot())
    assert tree_equal(final_snapshot, life.snapshot())
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == default_hash
    np.savez_compressed(directory / (args.prefix + '_actual_voice.npz'),
                        commands=np.stack([e['command'] for e in physical]),
                        pcm=np.stack([e['pcm'] for e in physical]),
                        index_times=np.array([e['index_time'] for e in physical]),
                        clocks=np.array([e['clock'] for e in physical]))
    report = {
        'scope': 'Separate single-time candidate with four vocal actuators; no navigation integration, words, semantics, or AGI claim.',
        'configuration': life.brain.snapshot()['config'],
        'seed': args.seed, 'actual_frames': life.frames, 'physical_seconds': life.frames / 10.,
        'clock': int(life.brain.runtime.clock.当前时间), 'one_current_time_neuron': int(life.brain.runtime.clock.时间激活.sum()) == 1,
        'all_pfc_pathways_active': all(life.brain.flags.values()),
        'training_origin': 'Seeded continuous motor exploration, measured self PCM, and actual executed command only.',
        'episode_controls_for_evaluator_only': [e['command'].tolist() for e in examples],
        'unexperienced_controls_for_evaluator_only': [e['command'].tolist() for e in neighbours],
        'probe_protocol': 'Independent frozen clone per sound; identical pre-probe state; no teacher motor or sound identifier input.',
        'stages': stages, 'probes': probe_rows,
        'safe_checkpoint_roundtrip_and_8_learning_frames_exact': True,
        'actual_pcm_replay_bitwise_equal': True,
        'default_life_unchanged': True, 'default_life_sha256': default_hash,
        'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'wall_seconds': time.perf_counter() - started,
        'limitations': [
            'This candidate has a vocal body only; the default navigating brain was not replaced.',
            'Initial/native memory readouts are diagnostic comparisons; the final recurrent PFC readout is reported separately.',
            'Known sounds can test sensorimotor memory, and adjacent sounds can test local acoustic transfer, not language.',
            'The actuator parameters and control schedules are evaluator data, absent from auditory or visual brain inputs.',
        ],
    }
    report_path = directory / (args.prefix + '.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('actual_frames', 'clock', 'stages', 'wall_seconds')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
