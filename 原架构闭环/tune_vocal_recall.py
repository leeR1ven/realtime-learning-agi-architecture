"""Bounded frozen-clone tuning of existing PFC input/recall parameters only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from checkpoint import load_tree, save_tree
from vocal_learning_experiment import (frozen_probe, measured_pitch, sounding, summarize,
                                       VocalLearningLife, tree_equal)


HERE = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    directory = HERE / 'results'
    base = 'vocal_learning_24_motor64'
    original = load_tree(directory / (base + '.npz'))
    original_report = json.loads((directory / (base + '.json')).read_text(encoding='utf-8'))
    with np.load(directory / (base + '_actual_voice.npz'), allow_pickle=False) as archive:
        waves = archive['pcm'].copy()
    controls = original_report['episode_controls_for_evaluator_only']
    known = [{'id': i, 'command': np.asarray(c), 'ear_pcm': .35 * waves[12*i+11],
              'reference': np.concatenate(waves[12*i+8:12*i+12])} for i, c in enumerate(controls)]
    neighbours = []
    for i, c in enumerate(original_report['unexperienced_controls_for_evaluator_only']):
        reference = sounding(c)
        neighbours.append({'id': i, 'command': np.asarray(c),
                           'ear_pcm': .35 * reference[-800:], 'reference': reference})
    # Selection uses four old cues and four external near-neighbours only.
    # Other old cues, all later self sounds, and other near-neighbours are held
    # out from selection. This is a small engineering tuning set, not proof of
    # broad statistical generalization or an autonomous reward mechanism.
    calibration = known[:4] + neighbours[:4]
    rows = []
    for threshold in (.55, .7, .8):
        for gain in (6., 10., 20.):
            candidate = deepcopy(original)
            candidate['brain']['config']['index_threshold'] = threshold
            candidate['brain']['config']['recall_gain'] = gain
            probes, _ = frozen_probe(candidate, calibration, label='parameter_calibration')
            summary = summarize(probes)
            score = summary['final_pfc_response']['mean_spectral_cosine']
            rows.append({'index_threshold': threshold, 'recall_gain': gain,
                         'score': score, 'summary': summary})
    rows.sort(key=lambda row: (-row['score'], -row['index_threshold'], row['recall_gain']))
    selected = rows[0]
    final = deepcopy(original)
    final['brain']['config']['index_threshold'] = selected['index_threshold']
    final['brain']['config']['recall_gain'] = selected['recall_gain']
    groups = {'held_out_old_self_sounds': known[4:12],
              'held_out_new_self_sounds': known[12:],
              'held_out_unexperienced_nearby_sounds': neighbours[4:]}
    final_rows, summaries = {}, {}
    for name, examples in groups.items():
        probes, demo = frozen_probe(final, examples, label=name)
        final_rows[name], summaries[name] = probes, summarize(probes)
    # An independent signal-estimator check against known body physics. These
    # actual controls are held solely by the evaluator and never enter a cue.
    rng = np.random.default_rng(190733)
    pitch_errors = []
    for _ in range(32):
        c = rng.uniform([.6, .05, .05, .05], [.95, .95, .95, .95])
        estimate = measured_pitch(sounding(c))
        if estimate is None:
            raise AssertionError('voiced validation signal produced no pitch estimate')
        pitch_errors.append(abs(estimate - (80. + 260. * c[1])))
    assert max(pitch_errors) < .2, pitch_errors
    path = directory / 'vocal_learning_24_tuned.npz'
    save_tree(path, final)
    life = VocalLearningLife.from_snapshot(load_tree(path))
    assert tree_equal(final, life.snapshot())
    assert tree_equal(final['brain']['runtime'], original['brain']['runtime'])
    assert tree_equal(final['brain']['adapter'], original['brain']['adapter'])
    continued = VocalLearningLife.from_snapshot(final)
    for _ in range(8):
        left, right = life.spontaneous_control(), continued.spontaneous_control()
        assert np.array_equal(left, right)
        assert np.array_equal(life.step(left)[0], continued.step(right)[0])
        assert tree_equal(life.snapshot(), continued.snapshot())
    report = {'scope': 'Parameter-only input/recall balance on a frozen vocal candidate; default navigating life unchanged.',
              'selection_examples': 'First four old and first four near-neighbour cues; all other cues held out from selection.',
              'selected_parameters': selected, 'calibration_candidates': rows,
              'held_out_summaries': summaries, 'held_out_probes': final_rows,
              'parameters_only_no_weight_or_memory_rewrite': True,
              'safe_restore_and_8_actual_learning_frames_exact': True,
              'pitch_estimator_validation': {'independent_physical_samples': 32, 'maximum_error_hz': max(pitch_errors),
                                             'median_error_hz': float(np.median(pitch_errors))},
              'candidate_path': str(path), 'candidate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'wall_seconds': time.perf_counter() - started}
    output = directory / 'vocal_recall_parameter_tuning.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('selected_parameters', 'held_out_summaries', 'pitch_estimator_validation', 'wall_seconds')}, indent=2))


if __name__ == '__main__':
    main()
