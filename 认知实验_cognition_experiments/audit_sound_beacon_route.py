"""Actual PCM/RGB co-exposure and read-only sound-to-image pathway probes.

No model source is changed. A disposable newborn navigation prototype observes
simultaneous sounds and visible objects, executes its own muscles, and receives
no teacher or reward. Registered checkpoints are only queried with learning off.
Colors/positions below belong to stimulus generation and offline scoring only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
NAV = HERE.parent/'色标导航_color_beacon_navigation'
sys.path.insert(0, str(NAV))
from environment import BeaconNavigationEnv, Scene, BEACON_NAMES, BEACON_COLORS
from route_switch_controller import RouteSwitchController
from textured_route_environment import TexturedRouteEnv
from route_state_digest import digest


def stimulus(color, *, hidden=False, seed=81000):
    # One visible foreground beacon; other beacons are behind the viewer.
    positions = [(.7, 1.), (.7, 3.), (.7, 5.)]
    if not hidden:
        positions[color] = (4.5, 3.)
    scene = Scene('av_pairing', 6., 6., (), tuple(positions),
                  {'presentation': (2.7, 3., 0.)})
    env = BeaconNavigationEnv(scene, seed=seed)
    env.reset(start='presentation', target=BEACON_NAMES[color])
    return env


def color_rays(values):
    rgb = np.asarray(values).reshape(31, 3)
    return [int(np.count_nonzero(np.all(np.isclose(rgb, c, atol=1e-6), axis=1)))
            for c in BEACON_COLORS]


def fixed_hash(brain):
    return digest({name: getattr(brain, name) for name in (
        'context_source', 'rule_source', 'view_sources', 'thresholds', 'levels',
        'event_ids', 'event_previous', 'event_next', 'event_frame', 'event_context',
        'event_features', 'event_views', 'event_muscles', 'event_strength',
        'event_kind', 'feature_to_events', 'readout_feature_to_events')})


def receptor_description(brain, receptor, values):
    channel, pair = divmod(int(receptor), brain.receptor_levels*2)
    level, side = divmod(pair, 2)
    if channel < 31:
        meaning = f'depth_ray_{channel}'
    elif channel < 124:
        ray, component = divmod(channel-31, 3)
        meaning = f'rgb_ray_{ray}_{"RGB"[component]}'
    else:
        meaning = ('forward_velocity', 'left_velocity', 'angular_velocity', 'pain', 'touch_max')[channel-124]
    return {'receptor': int(receptor), 'channel': channel, 'meaning': meaning,
            'value': float(values[channel]), 'threshold': float(brain.levels[level]),
            'side': 'greater_or_equal' if side == 0 else 'less'}


def probe(brain, packet, *, name, visual_lesion=False, prelude=None):
    brain.reset_activity()
    # Freeze the nonplastic frame counter as well as neural short-term state.
    brain.frame = 0
    brain.rng = np.random.default_rng(440031)
    original_view = brain.view_enabled
    brain.view_enabled = not visual_lesion
    before_count = brain.next_event_id
    if prelude is not None:
        brain.observe_then_act(prelude, learning=False)
    values, audio = brain._observe(packet)
    muscles, info = brain.observe_then_act(packet, learning=False)
    active = np.asarray(info['pfc_active'], np.int32)
    facts_reachable = sum(len(brain.feature_to_events[int(c)]) for c in active)
    readout_reachable = sum(len(brain.readout_feature_to_events[int(c)]) for c in active)
    recalled = []
    for event_id, weight in zip(info['recall_events'], info['recall_weights']):
        slot = brain.event_slot_by_id[event_id]
        recalled.append({'event_id': int(event_id), 'weight': float(weight),
            'strength': float(brain.event_strength[slot]),
            'color_ray_counts': color_rays(brain.event_views[slot, 31:124]),
            'next_event': int(brain.event_next[slot]),
            'muscles': brain.event_muscles[slot].tolist()})
    connection = None
    if recalled:
        slot = brain.event_slot_by_id[recalled[0]['event_id']]
        incoming = np.intersect1d(active, brain.event_features[slot])
        contributions = np.array([1/np.sqrt(len(brain.readout_feature_to_events[int(c)]))
            /np.sqrt(brain.event_active_count[slot]) for c in incoming])
        cell = int(incoming[np.argmax(contributions)])
        connection = {'mixed_cell': cell,
            'goal_input_cell': int(brain.context_source[cell]),
            'goal_input_activity': float(brain.context[brain.context_source[cell]]),
            'rule_input_cell': int(brain.rule_source[cell]),
            'rule_input_activity': float(brain.rule_context[brain.rule_source[cell]]),
            'current_sensory_inputs': [receptor_description(brain, r, values) for r in brain.view_sources[cell]],
            'mixed_cell_threshold': float(brain.thresholds[cell]),
            'destination_event_id': recalled[0]['event_id'],
            'cell_to_event_current': float(contributions.max()),
            'fixed_input_connections': True, 'event_connection_from_experience': True}
    result = {'name': name, 'visual_input_lesion': visual_lesion,
        'preceded_by_rule_only_sensory_probe': prelude is not None,
        'current_color_ray_counts': color_rays(packet['observation']['ray_colors']),
        'heard_receptors': audio.tolist(), 'goal_activity': info['goal_active'],
        'rule_activity': info['rule_active'], 'pfc_active_count': len(active),
        'actual_recorded_feature_event_connections_reachable': facts_reachable,
        'reward_gated_connections_reachable': readout_reachable,
        'recall_count': info['recall_count'], 'recalled_events': recalled,
        'recalled_view': info['recalled_view'], 'feedback_pfc_count': info['feedback_pfc_active_count'],
        'sequence_supported_events': info['sequence_supported_events'],
        'source': info['source'], 'muscles': muscles.tolist(),
        'strongest_connection_in_first_recalled_event': connection,
        'learning': False, 'reward': 0, 'no_physical_rollout': True}
    brain.view_enabled = original_view
    assert brain.next_event_id == before_count
    return result


def coexposure():
    # Same birth connections as the registered model; only storage allocation is
    # reduced, with no capacity reached. Do not reset neural state between pairs.
    brain = RouteSwitchController(seed=2026, mixed_neurons=8192, max_events=2048)
    history, exact_packets = [], {}
    for repetition in range(4):
        for color in range(3):
            env = stimulus(color, seed=81000+3*repetition+color)
            packet = env.sensor_packet()
            exact_packets[color] = copy.deepcopy(packet)
            for frame in range(6):
                visible = color_rays(packet['observation']['ray_colors'])
                muscles, info = brain.observe_then_act(packet, learning=True)
                assert not info['teaching']
                history.append({'event_id': brain.next_event_id-1, 'repetition': repetition,
                    'stimulus_color': BEACON_NAMES[color], 'time': float(env.world.time),
                    'color_ray_counts': visible, 'heard_receptors': info['heard'],
                    'joint_visible_and_audible': bool(visible[color] > 0 and info['heard'][color] > .9),
                    'muscles_executed': muscles.tolist()})
                packet = env.step(muscles, .1)
    memory_before = fixed_hash(brain)
    cases = []
    for color in range(3):
        cases.append(probe(brain, exact_packets[color], name=BEACON_NAMES[color]+'_same_av_view'))
        empty = stimulus(color, hidden=True).sensor_packet()
        assert sum(color_rays(empty['observation']['ray_colors'])) == 0
        cases.append(probe(brain, empty, name=BEACON_NAMES[color]+'_tone_without_visible_beacon'))
        cases.append(probe(brain, empty, name=BEACON_NAMES[color]+'_sensory_input_lesion', visual_lesion=True))
    assert fixed_hash(brain) == memory_before
    return {'protocol': '12 presentations x 6 actual physics steps; same 8192-cell birth; no model reset between presentations; no reward, teacher, or model changes',
        'birth_hash': brain.birth_hash, 'events': brain.next_event_id,
        'joint_exposure_frames': sum(r['joint_visible_and_audible'] for r in history),
        'joint_exposure_by_color': {c: sum(r['joint_visible_and_audible'] and r['stimulus_color'] == c for r in history) for c in BEACON_NAMES},
        'stored_feature_event_connections': sum(map(len, brain.feature_to_events)),
        'reward_gated_readout_connections': sum(map(len, brain.readout_feature_to_events)),
        'valued_events': int(np.count_nonzero(brain.event_strength)),
        'teacher_events': int(np.count_nonzero(brain.event_kind == 2)),
        'chain_starts': int(np.count_nonzero((brain.event_ids >= 0) & (brain.event_previous < 0))),
        'evaluation_preserved_all_learned_connections': True,
        'presentations': history, 'probes': cases}


def registered_checkpoint():
    path = NAV/'results'/'route_texture_evaluation.npz'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    brain = RouteSwitchController.load(path)
    weight_sha = fixed_hash(brain)
    cases = []
    for route in ('bottom', 'top'):
        for name, start in (('middle', (1.3, 3.4, 0.)), ('lower', (1.3, 1.4, 0.)), ('upper', (1.3, 6.6, 0.))):
            env = TexturedRouteEnv(seed=41011, texture_seed=44017)
            packet = env.reset(route=route, target='red', start=start)
            cases.append(probe(brain, packet, name=name+'_'+route+'_goal_and_rule_tones'))
            silent = copy.deepcopy(packet)
            silent['waveform'] = np.zeros_like(packet['waveform'])
            cases.append(probe(brain, silent, name=name+'_'+route+'_silence'))
            if name == 'middle':
                # Base environment renders exactly the same world and target
                # PCM, omitting only the subclass's route tone addition.
                goal_only = BeaconNavigationEnv.sensor_packet(env)
                rule_only = copy.deepcopy(packet)
                rule_only['waveform'] = packet['waveform']-goal_only['waveform']
                for key in packet['observation']:
                    assert np.array_equal(goal_only['observation'][key], packet['observation'][key])
                cases.append(probe(brain, goal_only, name=route+'_goal_tone_only'))
                cases.append(probe(brain, rule_only, name=route+'_rule_tone_only'))
                cases.append(probe(brain, goal_only, prelude=rule_only,
                    name=route+'_rule_held_goal_tone'))
                cases.append(probe(brain, silent, prelude=rule_only,
                    name=route+'_rule_held_silence'))
        cases.append(probe(brain, packet, name=route+'_sensory_input_lesion', visual_lesion=True))
    assert fixed_hash(brain) == weight_sha
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    return {'path': str(path), 'checkpoint_sha256': sha, 'birth_hash': brain.birth_hash,
        'all_learned_connections_unchanged': True, 'probes': cases}


def main():
    result = {'scope': 'Existing navigation prototype, not the complete original brain architecture',
        'limitations': [
            'No rewardless recall in this prototype does not disprove the original shared-time architecture.',
            'An image containing red in recalled events is only a trace observation; it does not establish a goal image or spatial reasoning.',
            'Visual-input lesion removes body receptors too; ordinary blank vision still activates paired negative receptors.',
            'The exposure theater is a controlled sensory experiment, not autonomous route learning.'
            , 'The prototype internally cuts event links on goal changes: 12 presentations are not one unbroken stored timeline.'
            , 'The displayed connection is strongest only within the first returned event; it is not established as the dominant causal source of the final action.'
        ],
        'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((NAV/name).read_bytes()).hexdigest() for name in (
            'associative_controller.py', 'route_switch_controller.py', 'environment.py', 'textured_route_environment.py')},
        'coexposure': coexposure(), 'registered_checkpoint': registered_checkpoint()}
    output = HERE/'results'/'sound_beacon_route_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    simple = {'joint_exposure_frames': result['coexposure']['joint_exposure_frames'],
        'by_color': result['coexposure']['joint_exposure_by_color'],
        'stored_connections': result['coexposure']['stored_feature_event_connections'],
        'readout_connections': result['coexposure']['reward_gated_readout_connections'],
        'coexposure_probe_recalls': {r['name']: r['recall_count'] for r in result['coexposure']['probes']},
        'checkpoint_probes': [{k: r[k] for k in ('name', 'pfc_active_count', 'recall_count', 'source')} for r in result['registered_checkpoint']['probes']]}
    print(json.dumps(simple, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
