"""Read-only sensory-boundary diagnostics; no training or frozen-code edits."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'色标导航'))
from route_switch_controller import RouteSwitchController, _five_frequency_receptors
from route_switch_environment import RouteSwitchEnv


def load_source(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    distances = np.array([6., 9., 12.], dtype=np.float32)
    old_levels = np.linspace(1/17, 16/17, 16, dtype=np.float32)
    old = np.sum(np.exp(-distances[:, None]/2.) >= old_levels[None, :], axis=1)
    assert np.array_equal(old, [0, 0, 0])
    fixed_levels = np.linspace(1/65, 64/65, 64, dtype=np.float32)
    repaired = np.sum((distances/16.)[:, None] >= fixed_levels[None, :], axis=1)
    assert len(np.unique(repaired)) == 3
    report = {'distance_encoding': {
        'source': '色标导航/associative_controller.py:104,181,213',
        'all_zero_positive_beyond_metres': float(2*np.log(17)),
        'example_distances': distances.tolist(), 'old_positive_counts': old.tolist(),
        'proposed_fixed_d_over_16_with_64_levels_counts': repaired.tolist(),
        'classification': 'Confirmed information loss in navigation adapter; physical range is still correctly measured'}}

    env = RouteSwitchEnv(seed=13)
    packet = env.reset(route='bottom')
    controller = RouteSwitchController(mixed_neurons=64, max_events=1)
    other = copy.deepcopy(packet)
    packet['observation']['touch'] = np.array([1., 0., 0., 0.])
    other['observation']['touch'] = np.array([0., 0., 1., 0.])
    values_a, _ = controller._observe(packet)
    values_b, _ = controller._observe(other)
    assert np.array_equal(values_a, values_b)
    report['directional_touch'] = {
        'source': '闭环仿真/world.py:230; 色标导航/associative_controller.py:178',
        'physical_order': ['front', 'back', 'left', 'right'],
        'different_front_and_left_touch_produce_identical_old_values': True,
        'proposed_fix': 'Keep all four already available body-frame channels; 129 sensory values become 132',
        'classification': 'Confirmed information loss in navigation adapter, not a world collision defect'}

    inventory = []
    for filename in ('route_switch_evaluation.npz', 'route_texture_evaluation.npz'):
        path = ROOT/'色标导航'/'results'/filename
        with np.load(path, allow_pickle=False) as archive:
            valid = archive['event_ids'] >= 0
            values = archive['event_views'][valid]
        velocity, angular = values[:, 124:126], values[:, 126]
        inventory.append({
            'checkpoint': filename, 'recorded_events': len(values),
            'body_velocity_values_at_clip_endpoints': int(np.count_nonzero((velocity <= 0) | (velocity >= 1))),
            'angular_values_at_clip_endpoints': int(np.count_nonzero((angular <= 0) | (angular >= 1))),
            'decoded_recorded_velocity_min': float((velocity.min()*6)-3),
            'decoded_recorded_velocity_max': float((velocity.max()*6)-3),
            'decoded_recorded_angular_min': float((angular.min()*8)-4),
            'decoded_recorded_angular_max': float((angular.max()*8)-4),
            'depth_rays_below_first_old_threshold_fraction': float(np.mean(values[:, :31] < old_levels[0]))})
    report['velocity_angular_clipping'] = {
        'source': '色标导航/associative_controller.py:182',
        'limits': {'body_velocity_m_per_s': [-3, 3], 'angular_velocity_rad_per_s': [-4, 4]},
        'recorded_checkpoint_checks': inventory,
        'interpretation': 'Check actual endpoint counts before attributing failure to clipping. These archives cover recorded pre-action observations, not every internal 5ms physics substep.'}

    waveform = .45*np.sin(2*np.pi*1100*np.arange(800)/8000)
    receptors = _five_frequency_receptors(waveform)
    assert np.max(receptors) < 1e-10
    report['restricted_audio'] = {
        'source': '色标导航/route_switch_controller.py:28,41',
        'frequencies_hz': [220, 440, 660, 990, 1320],
        '1100_hz_pure_tone_max_receptor_response': float(np.max(receptors)),
        'classification': 'Deliberate five-band experimental adapter; not general audition or a learned language decoder'}

    vision = load_source('_sensory_audit_original_vision', '视觉前处理.py')
    audio = load_source('_sensory_audit_original_audio', '听觉前处理.py')
    small_image_error = None
    try:
        with np.errstate(all='raise'):
            vision.降采样(np.full((1, 31, 3), 128, dtype=np.uint8))
    except FloatingPointError as exc:
        small_image_error = type(exc).__name__
    assert small_image_error == 'FloatingPointError'
    valid_image = vision.降采样(np.full((24, 16, 3), 128, dtype=np.uint8))
    assert np.isfinite(valid_image).all()
    spectrum_error = None
    spectrum = np.abs(np.fft.rfft(waveform))
    try:
        audio.压缩频谱(spectrum)
    except ValueError as exc:
        spectrum_error = str(exc)
    assert spectrum_error is not None and len(spectrum) == 401
    report['original_module_adapter_requirements'] = {
        'vision_source': '视觉前处理.py:68,90',
        'one_by_31_ray_image_causes_zero_area_downsampling': small_image_error,
        '24_by_16_image_is_finite': True,
        'audio_source': '听觉前处理.py:69,97',
        'eight_khz_point_one_second_rfft_bins': len(spectrum),
        'original_minimum_spectrum_bins': audio.频带数,
        'direct_audio_adapter_error': spectrum_error,
        'interpretation': 'Original modules define their own sensor layouts. A 31-ray strip and 800-sample PCM chunk cannot silently be passed as full compatible image/spectrum inputs.'}

    report['action_consequence_timing'] = {
        'sources': ['色标导航/associative_controller.py:492', '色标导航/route_switch_experiment.py:73',
                    '色标导航/route_switch_experiment.py:77', '色标导航/route_switch_experiment.py:88'],
        'existing_order': 'Record current sensory values and proposed command, execute world step, reward if terminal, then break.',
        'confirmed_gap': 'The final post-action terminal observation is never passed into the old learner. Nonterminal consequences appear only as the next frame sensory state.',
        'proposed_fix': 'act prepares a pending action; after successful physics, observe_outcome commits actual applied muscles and post-action sensory values, including terminal outcomes without another action.'}
    report['priorities'] = [
        'Replace lossy navigation receptor adapter: fixed d/16 with 64 threshold levels, all four directional touch values; innate reflex reads physical distances directly.',
        'Commit real action consequences after physics, including terminal sensory outcomes; scheduled route changes provide actual cross-rule experience without labels entering the brain.']
    report['no_training_or_frozen_code_changes'] = True
    output = Path(__file__).parent/'results'/'sensory_boundary_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
