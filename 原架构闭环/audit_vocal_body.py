"""Independent physical/signal checks for the optional vocal body.

These are actuator experiments, not evidence of language learning. The audit
does not load, alter, train, or save the default neural life.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from vocal_body import FRAME_SAMPLES, SAMPLE_RATE, VocalBody, mix_at_ear, write_pcm16


HERE = Path(__file__).resolve().parent


def equal_state(left, right):
    return all(np.array_equal(left[k], right[k]) if isinstance(left[k], np.ndarray)
               else left[k] == right[k] for k in left) and left.keys() == right.keys()


def settled(controls):
    body = VocalBody()
    for _ in range(12):
        body.render_frame(controls)
    return np.concatenate([body.render_frame(controls) for _ in range(10)])


def spectrum(pcm):
    return np.fft.rfftfreq(len(pcm), 1. / SAMPLE_RATE), np.abs(np.fft.rfft(pcm)) ** 2


def band_energy(pcm, low, high):
    frequency, energy = spectrum(pcm)
    return float(energy[(frequency >= low) & (frequency <= high)].sum())


def fundamental_by_autocorrelation(pcm):
    # Search one plausible source period. Integer lag is sufficient to
    # independently distinguish a 120 Hz source from a 240 Hz source.
    ac = np.fft.irfft(np.abs(np.fft.rfft(pcm, n=2 * len(pcm))) ** 2)
    lags = np.arange(20, 101)
    candidates = lags[(ac[lags] > ac[lags - 1]) & (ac[lags] >= ac[lags + 1])]
    highest = np.max(ac[candidates])
    # Earliest nearly-full correlation avoids returning a double period.
    lag = candidates[ac[candidates] >= .97 * highest][0]
    return float(SAMPLE_RATE / lag)


def audit():
    checks, result = {}, {}
    source_before = hashlib.sha256((HERE / 'vocal_body.py').read_bytes()).hexdigest()
    body = VocalBody()
    silence = body.render_frame([0., .8, .4, .7])
    checks['fresh_no_airflow_is_exact_silence'] = bool(np.array_equal(silence, np.zeros(800)))
    controls = [1., .42, .65, .71]
    for _ in range(20):
        body.render_frame(controls)
    decay_rms = []
    for _ in range(5):
        pcm = body.render_frame([0., .42, .65, .71])
        decay_rms.append(float(np.sqrt(np.mean(pcm * pcm))))
    checks['released_airflow_decays_without_reset'] = bool(
        all(a > b for a, b in zip(decay_rms, decay_rms[1:])) and decay_rms[-1] < 1e-14)
    result['airflow_release_rms'] = decay_rms

    low = settled([1., (120. - 80.) / 260., .4, .5])
    high = settled([1., (240. - 80.) / 260., .4, .5])
    measured_pitch = [fundamental_by_autocorrelation(x) for x in (low, high)]
    checks['tension_changes_measured_fundamental'] = bool(
        abs(measured_pitch[0] - 120.) < 3. and abs(measured_pitch[1] - 240.) < 5.)
    result['measured_fundamentals_hz'] = measured_pitch

    aperture0 = settled([1., 20. / 260., 0., .8])
    aperture1 = settled([1., 20. / 260., 1., .8])
    f1_ratios = [band_energy(x, 750., 1050.) / band_energy(x, 150., 400.)
                 for x in (aperture0, aperture1)]
    checks['aperture_moves_first_resonance_spectrum'] = f1_ratios[1] > 10. * f1_ratios[0]
    result['first_resonance_high_to_low_band_energy'] = f1_ratios
    tongue0 = settled([1., 20. / 260., .1, 0.])
    tongue1 = settled([1., 20. / 260., .1, 1.])
    f2_ratios = [band_energy(x, 2100., 2700.) / band_energy(x, 800., 1100.)
                 for x in (tongue0, tongue1)]
    checks['tongue_moves_second_resonance_spectrum'] = f2_ratios[1] > 10. * f2_ratios[0]
    result['second_resonance_high_to_low_band_energy'] = f2_ratios
    quiet = settled([.25, .3, .4, .5])
    loud = settled([.75, .3, .4, .5])
    checks['airflow_scales_actual_acoustic_pressure'] = bool(np.allclose(loud, 3. * quiet, atol=1e-14, rtol=1e-13))

    whole, split = VocalBody(), VocalBody()
    whole_pcm = whole.render_samples(controls, 8000)
    split_pcm = np.concatenate([split.render_frame(controls) for _ in range(10)])
    partition_error = float(np.max(np.abs(whole_pcm - split_pcm)))
    checks['phase_and_actuators_continuous_across_frames'] = partition_error < 2e-10
    result['one_second_partition_max_absolute_difference'] = partition_error
    fresh_frame = VocalBody().render_frame(controls)
    checks['later_frame_is_not_restarted_first_frame'] = not np.array_equal(split_pcm[-800:], fresh_frame)

    rng = np.random.default_rng(901781)
    body = VocalBody()
    peak, elapsed = 0., []
    for _ in range(1000):
        control = rng.uniform(0., 1., 4)
        started = time.perf_counter()
        pcm = body.render_frame(control)
        elapsed.append((time.perf_counter() - started) * 1000.)
        assert pcm.shape == (800,) and pcm.dtype == np.float64 and np.isfinite(pcm).all()
        assert np.all((body.actuators >= 0.) & (body.actuators <= 1.))
        peak = max(peak, float(np.max(np.abs(pcm))))
    checks['one_thousand_variable_frames_stable_and_bounded'] = peak < .95 and body.samples_rendered == 800000
    result['random_stream'] = {
        'frames': 1000, 'physical_seconds': 100., 'maximum_absolute_pcm': peak,
        'median_render_ms': float(np.median(elapsed)), 'p95_render_ms': float(np.percentile(elapsed, 95)),
    }
    bad_controls = [[0.] * 3, [0.] * 5, [np.nan, 0, 0, 0], [np.inf, 0, 0, 0],
                    [-.01, 0, 0, 0], [1.01, 0, 0, 0], [[0.] * 4]]
    rejected = 0
    for bad in bad_controls:
        before = body.snapshot()
        try:
            body.render_frame(bad)
        except ValueError:
            rejected += 1
        assert equal_state(before, body.snapshot())
    for count in (0, -1, 8001, 2.5, True):
        before = body.snapshot()
        try:
            body.render_samples(controls, count)
        except ValueError:
            rejected += 1
        assert equal_state(before, body.snapshot())
    checks['invalid_actions_rejected_before_body_mutation'] = rejected == len(bad_controls) + 5
    result['invalid_actions_rejected'] = rejected

    with tempfile.TemporaryDirectory(prefix='vocal_body_audit_') as directory:
        checkpoint = Path(directory) / 'body.npz'
        body.save(checkpoint)
        with np.load(checkpoint, allow_pickle=False) as archive:
            assert all(archive[k].dtype.kind in 'biuf' for k in archive.files)
        restored = VocalBody.load(checkpoint)
        checks['safe_npz_roundtrip_exact'] = equal_state(body.snapshot(), restored.snapshot())
        for _ in range(32):
            control = rng.uniform(0., 1., 4)
            assert np.array_equal(body.render_frame(control), restored.render_frame(control))
            assert equal_state(body.snapshot(), restored.snapshot())
        checks['restore_preserves_32_continued_frames_bitwise'] = True
    invalid_states = []
    for key, value in (('kind', 'other'), ('schema_version', True), ('schema_version', 2),
                       ('sample_rate', 16000), ('phase', float('nan')), ('phase', 7.),
                       ('phase', -.1), ('samples_rendered', -1), ('samples_rendered', 1.5),
                       ('actuators', np.ones(4) * 2.)):
        invalid = copy.deepcopy(body.snapshot())
        invalid[key] = value
        invalid_states.append(invalid)
    invalid_states.append({**body.snapshot(), 'unexpected': 1})
    rejected_states = 0
    for invalid in invalid_states:
        try:
            VocalBody.from_snapshot(invalid)
        except ValueError:
            rejected_states += 1
    checks['malformed_snapshot_rejected'] = rejected_states == len(invalid_states)

    external, own = rng.uniform(-1., 1., 800), rng.uniform(-1., 1., 800)
    external_before, own_before = external.copy(), own.copy()
    mixed = mix_at_ear(external, own)
    checks['self_hearing_is_exact_linear_physical_mixture'] = bool(np.array_equal(mixed, .65 * external + .35 * own))
    checks['self_hearing_does_not_modify_input_buffers'] = bool(np.array_equal(external, external_before) and np.array_equal(own, own_before))
    checks['self_hearing_has_no_voice_isolation_or_labels'] = bool(np.array_equal(mix_at_ear(external, own, environment_gain=1., self_gain=0.), external))
    rejected_mix = 0
    for eg, sg in ((1., 1.), (-.1, .1), (.1, np.nan), (np.inf, 0.)):
        try:
            mix_at_ear(external, own, environment_gain=eg, self_gain=sg)
        except ValueError:
            rejected_mix += 1
    checks['invalid_acoustic_gains_rejected'] = rejected_mix == 4

    # Feed only measured PCM to the real, unchanged auditory frontend.
    from sensory_adapter import SensoryAdapter
    # Match the selected life's actual auditory threshold; the old uncalibrated
    # constructor default .60 has a different silent frontend response.
    adapter = SensoryAdapter(audio_threshold=.52, contrast_to_silence=True)
    packets = [low[:800], high[:800], aperture0[:800], aperture1[:800], tongue0[:800], tongue1[:800]]
    features = []
    for pcm in packets:
        spectrum_value = adapter.spectrum(mix_at_ear(np.zeros(800), pcm))
        signal = adapter.thermometer(spectrum_value, adapter.audio_levels)
        feature = adapter.networks['audio'].前向传播(signal).astype(bool) ^ adapter.audio_silence_baseline
        features.append(np.flatnonzero(feature))
        adapter.reset_activity()
    distinct = len({tuple(x) for x in features})
    checks['actual_audio_frontend_receives_distinct_self_sounds'] = distinct == len(packets)
    result['self_heard_auditory_features'] = {'distinct_sets': distinct, 'samples': len(packets),
                                           'active_counts': [len(x) for x in features]}

    results_dir = HERE / 'results'
    results_dir.mkdir(exist_ok=True)
    demo = VocalBody()
    demo_frames = []
    schedule = ([0., .2, .3, .4], [.8, .2, .3, .4], [.8, .7, .3, .4],
                [.8, .2, .9, .4], [.8, .2, .3, .9], [0., .2, .3, .4])
    for control in schedule:
        demo_frames.extend(demo.render_frame(control) for _ in range(10))
    demo_path = results_dir / 'vocal_body_actuator_demo.wav'
    write_pcm16(demo_path, np.concatenate(demo_frames))
    checks['source_unchanged_during_audit'] = hashlib.sha256((HERE / 'vocal_body.py').read_bytes()).hexdigest() == source_before
    report = {
        'scope': 'Only a vocal actuator and self-hearing substrate; no neural training or language achievement.',
        'checks': checks, 'all_passed': all(checks.values()), 'measurements': result,
        'body_source_sha256': source_before,
        'demo_wav': str(demo_path),
        'demo_note': 'Six seconds of scripted body controls, not learned speech or a neural policy.',
    }
    report_path = results_dir / 'vocal_body_audit.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['all_passed']:
        raise AssertionError('Failed vocal body checks: ' + ', '.join(k for k, v in checks.items() if not v))
    return report


if __name__ == '__main__':
    audit()
