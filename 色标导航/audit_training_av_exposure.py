"""Read-only audit of actual sound-window / red-ray co-exposure in training.

No model is instantiated, trained or modified. event_views contains the actual
pre-action sensor values, not imagined/recalled pictures. Global event_frame is
mapped to every complete training trial, then to the logged pre-action pose.

Checkpoints did not store PCM itself. Actual sound windows are reconstructed
from the frozen environment generator, the original per-trial seed and recorded
simulation time. Retained event_context is reported separately and is NEVER
used to decide whether a sound was physically present.

Selected logged poses are rendered offline to verify archived RGB/range values.
This changes only a disposable renderer's pose; it supplies no coordinates or
answers to a model and executes no training or simulated navigation actions.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from environment import BEACON_COLORS, BEACON_NAMES
from route_switch_environment import RouteSwitchEnv
from textured_route_environment import TexturedRouteEnv


HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
RGB_COLORS = np.asarray(BEACON_COLORS, dtype=np.float32)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_evidence(env, trajectory, row_values, row_contexts, row_ids, local_frame):
    pose = trajectory[local_frame]
    env.world.position[:] = pose['position']
    env.world.heading = float(pose['heading'])
    env.world.time = float(pose['time'])
    packet = env.sensor_packet(dt=.1)
    obs = packet['observation']
    rgb = row_values[local_frame, 31:124].reshape(31, 3)
    assert np.array_equal(np.asarray(obs['ray_colors'], dtype=np.float32), rgb)
    encoded_depth = np.exp(-np.asarray(obs['ray_distances'], dtype=np.float32)/2.)
    assert np.array_equal(encoded_depth, row_values[local_frame, :31])
    colors = {name: np.flatnonzero(np.all(rgb == color, axis=1)).tolist()
              for name, color in zip(BEACON_NAMES, RGB_COLORS)}
    wave = packet['waveform']
    n = len(wave)
    frequencies = [env._tone_map['red'], env._route_tone_map[env._requested_route]]
    phase = 2*np.pi*np.asarray(frequencies)[:, None]*np.arange(n)[None, :]/8000.
    amplitudes = 2*np.abs(np.exp(-1j*phase)@wave)/n
    audible = bool(np.count_nonzero(np.abs(wave) > 1e-12))
    return dict(local_control_frame=int(local_frame), event_id=int(row_ids[local_frame]),
        observation_time_seconds=float(pose['time']), position=list(pose['position']), heading=float(pose['heading']),
        actual_pcm_window_reconstructed=True, pcm_nonzero_samples_above_1e_minus_12=int(np.count_nonzero(np.abs(wave)>1e-12)),
        actual_audio_present=audible, target_tone_hz=frequencies[0], route_tone_hz=frequencies[1],
        target_tone_amplitude=float(amplitudes[0]), route_tone_amplitude=float(amplitudes[1]),
        visible_beacon_ray_indices=colors, red_ray_count=len(colors['red']),
        retained_event_context_signature=int(row_contexts[local_frame]),
        rgb_and_encoded_depth_match_independent_pose_render=True)


def inspect_condition(stem):
    source = RESULTS/(stem+'.json')
    checkpoint = RESULTS/(stem+'.npz')
    source_hash, checkpoint_hash = sha256(source), sha256(checkpoint)
    report = json.loads(source.read_text(encoding='utf-8'))
    assert report['status'] == 'complete' and len(report['training']) == 36
    assert checkpoint_hash == report['checkpoint_sha256']
    with np.load(checkpoint, allow_pickle=False) as archive:
        ids = archive['event_ids']
        valid = ids >= 0
        frames = archive['event_frame'][valid]
        ids = ids[valid]
        views = archive['event_views'][valid]
        contexts = archive['event_context'][valid]
        kinds = archive['event_kind'][valid]
    order = np.argsort(frames)
    frames, ids, views, contexts, kinds = (array[order] for array in (frames, ids, views, contexts, kinds))
    total = sum(row['control_steps'] for row in report['training'])
    assert total == len(frames)
    assert np.array_equal(frames, np.arange(total)) and np.array_equal(ids, np.arange(total))
    assert views.shape == (total, 129) and np.all(kinds == 1)
    assert report['protocol']['dt'] == .1
    texture_seed = report['protocol'].get('wall_texture_seed')
    trials, grouped = [], defaultdict(lambda: {'trials': 0, 'sound_frames': 0, 'sound_red_ray_samples': 0,
        'trials_with_red_during_sound': 0, 'first_red_times': []})
    cursor = 0
    verified_render_count = 0
    for row in report['training']:
        n = row['control_steps']
        assert row['events_before'] == cursor and row['events_after'] == cursor+n
        row_views, row_contexts, row_ids = views[cursor:cursor+n], contexts[cursor:cursor+n], ids[cursor:cursor+n]
        rgb = row_views[:, 31:124].reshape(n, 31, 3)
        red_counts = np.all(rgb == RGB_COLORS[0], axis=2).sum(axis=1)
        visible = np.flatnonzero(red_counts)
        first = int(visible[0]) if len(visible) else None
        trajectory = row['trajectory']
        assert len(trajectory) == n+1  # last post-action/terminal pose has no old memory event
        assert np.allclose([sample['time'] for sample in trajectory], np.arange(n+1)*.1, atol=1e-7, rtol=0)
        assert row['target'] == 'red' and len(row['route_requests']) == 1
        env = (RouteSwitchEnv(seed=row['seed']) if texture_seed is None else
               TexturedRouteEnv(seed=row['seed'], texture_seed=texture_seed))
        env.reset(start=row['initial_pose'], target='red', route=row['requested_route'])
        assert env._tone_map == row['tone_map'] and env._route_tone_map == row['route_tone_map']
        assert env.cue_seconds == env.route_cue_seconds == .6
        sound_frames = [i for i in range(n) if trajectory[i]['time'] < .6-1e-10]
        assert sound_frames == list(range(6))
        evidence = [render_evidence(env, trajectory, row_views, row_contexts, row_ids, i) for i in sound_frames]
        verified_render_count += len(evidence)
        assert all(e['actual_audio_present'] for e in evidence)
        assert all(abs(e['target_tone_amplitude']-.45) < 1e-8 and abs(e['route_tone_amplitude']-.45) < 1e-8 for e in evidence)
        silent_boundary = render_evidence(env, trajectory, row_views, row_contexts, row_ids, 6)
        verified_render_count += 1
        assert not silent_boundary['actual_audio_present']
        first_evidence = None if first is None else render_evidence(env, trajectory, row_views, row_contexts, row_ids, first)
        verified_render_count += int(first is not None)
        previous_evidence = None if first is None or first == 0 else render_evidence(
            env, trajectory, row_views, row_contexts, row_ids, first-1)
        verified_render_count += int(previous_evidence is not None)
        if previous_evidence is not None:
            assert previous_evidence['red_ray_count'] == 0
        sound_red_rays = int(sum(e['red_ray_count'] for e in evidence))
        sound_beacon_rays = {name: sum(len(e['visible_beacon_ray_indices'][name]) for e in evidence) for name in BEACON_NAMES}
        current = dict(episode=row['episode'], environment_seed=row['seed'], start=row['start_name'],
            requested_route=row['requested_route'], initial_pose=row['initial_pose'],
            event_id_start=cursor, event_id_stop_exclusive=cursor+n, control_steps=n,
            actual_sound_interval_seconds=[0., .6], actual_sound_control_frames=sound_frames,
            sound_window_red_ray_samples=sound_red_rays,
            sound_window_beacon_ray_samples=sound_beacon_rays,
            red_visible_during_actual_sound=sound_red_rays > 0,
            audio_window_pose_evidence=evidence, first_silent_boundary=silent_boundary,
            first_red_visible=first_evidence, last_observation_before_first_red=previous_evidence,
            whole_trial_red_visible_control_frames=int(np.count_nonzero(red_counts)),
            physical_red_arrival=row['target_reached'], route_correct_arrival=row['correct_arrival'],
            first_target_time=row['first_target_time'],
            context_signatures_after_sound=np.unique(row_contexts[6:]).astype(int).tolist())
        trials.append(current)
        cell = grouped[row['start_name'], row['requested_route']]
        cell['trials'] += 1
        cell['sound_frames'] += len(sound_frames)
        cell['sound_red_ray_samples'] += sound_red_rays
        cell['trials_with_red_during_sound'] += int(sound_red_rays > 0)
        if first_evidence:
            cell['first_red_times'].append(first_evidence['observation_time_seconds'])
        cursor += n
    first_times = [trial['first_red_visible']['observation_time_seconds'] for trial in trials if trial['first_red_visible']]
    summary = dict(trials=36, complete_stored_pre_action_events=total,
        no_event_gaps_or_overwrite=True, actual_sound_frames=sum(len(t['actual_sound_control_frames']) for t in trials),
        red_ray_samples_during_actual_sound=sum(t['sound_window_red_ray_samples'] for t in trials),
        trials_red_visible_during_actual_sound=sum(t['red_visible_during_actual_sound'] for t in trials),
        trials_eventually_red_visible=len(first_times),
        earliest_first_red_seconds=min(first_times), median_first_red_seconds=float(np.median(first_times)),
        latest_first_red_seconds=max(first_times),
        first_red_observations_with_actual_pcm=sum(bool(t['first_red_visible'] and t['first_red_visible']['actual_audio_present']) for t in trials),
        independent_pose_renders_checked=verified_render_count,
        sound_window_beacon_ray_samples={name: sum(t['sound_window_beacon_ray_samples'][name] for t in trials) for name in BEACON_NAMES},
        sound_window_beacon_visible_trials={name: sum(t['sound_window_beacon_ray_samples'][name] > 0 for t in trials) for name in BEACON_NAMES},
        sound_window_beacon_visible_control_frames={name: sum(bool(e['visible_beacon_ray_indices'][name])
            for t in trials for e in t['audio_window_pose_evidence']) for name in BEACON_NAMES})
    assert sha256(source) == source_hash and sha256(checkpoint) == checkpoint_hash
    return dict(source=str(source), source_sha256=source_hash, checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash, texture_seed=texture_seed, summary=summary,
        groups=[dict(start=start, route=route, **cell) for (start, route), cell in grouped.items()], trials=trials)


def main():
    conditions = {name: inspect_condition(stem) for name, stem in (
        ('gray', 'route_switch_evaluation'), ('textured', 'route_texture_evaluation'))}
    result = dict(status='complete', conditions=conditions,
        method={
            'stored_RGB': 'event_views columns 31:124 are unmodified actual pre-action RGB, cast to float32 by the old input adapter.',
            'red_detection': 'Exact equality to the fixed red beacon RGB [1,.04,.04], not redness of textured wall pixels.',
            'time_mapping': 'Global event_frame and event_id both contiguous and equal cumulative per-trial control frames; observation pose is trajectory[local_frame].',
            'audio_presence': 'Reconstructed from original environment seed/time and frozen 0.6s PCM generator. The checkpoint stores no raw PCM; retained context never establishes sound presence.',
            'render_validation': 'Every sounding frame, first silent boundary, first visible red frame and preceding frame re-rendered at recorded pose; RGB and encoded range match checkpoint exactly.',
            'terminal_boundary': 'The extra final post-action pose is excluded because the old training loop stored no terminal sensor event.'},
        conclusions=[
            'Both formal training sets have zero real-PCM/red-vision co-exposure during all 36 initial sound windows.',
            'Every trial eventually observed red after the sound had stopped. Persistent tone context can co-occur with later red, but that is different evidence from hearing the actual sound beside a visible red beacon.',
            'These data do not establish a learned sound-alone -> red-visual-memory association. A separate causal memory recall test is needed.',
            'No architecture, controller, checkpoint, training report or GUI was changed; no training was run.'])
    output = RESULTS/'training_av_exposure_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), **{name: value['summary'] for name, value in conditions.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
