"""Unchanged physical navigation with timed auditory changes and real feedback.

Controller input stays exactly ``{'observation': six original fields,
'waveform': mono PCM}``. Scheduled route names, times, map coordinates and
scoring labels remain private environment data. No teacher/planner is provided.

Typical integration::
    env = IntegrationEnvironment(seed=3)
    packet = env.reset(route='bottom', start='left_middle',
                       switch_schedule=[(10.0, 'top')])
    muscles = brain.act(packet)
    transition = env.advance(muscles)
    brain.observe_outcome(transition['packet'], muscles,
                          **transition['feedback'])

The terminal packet must also be observed, without inventing another action.
The feedback reward is paid once, after a real first red arrival through the
currently required passage. The terminal Boolean remains true on later reads.
``step`` plus ``consume_feedback`` is equivalent to ``advance``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'色标导航'))
from environment import SENSOR_KEYS
from route_switch_environment import ROUTE_NAMES
from textured_route_environment import TexturedRouteEnv


class IntegrationEnvironment(TexturedRouteEnv):
    """Timings lie on a fixed control grid; physics substeps are unchanged.

    Cues are emitted after the preceding physical step and before the next
    packet. If that physical step already ended the trial, its first-arrival
    score is final and no later cue is emitted. No geometric predicate chooses
    cue timing or route. A repeated same-route cue is allowed as a control.
    """

    def __init__(self, *, seed=20260906, texture_seed=44017, control_dt=.1, **kwargs):
        if not np.isfinite(control_dt) or control_dt <= 0 or control_dt > 1:
            raise ValueError("control_dt must be finite and in (0,1]")
        self.control_dt = float(control_dt)
        self._constructor_seed = int(seed)
        self._explicit_reset_seed = None
        self._switch_schedule = []
        self._scheduled_cue_index = 0
        self._delivered_scheduled_cues = []
        self._pending_terminal_reward = 0.
        self._terminal_feedback_registered = False
        super().__init__(seed=seed, texture_seed=texture_seed, **kwargs)

    def _validate_schedule(self, schedule):
        result = []
        for item in (() if schedule is None else schedule):
            if isinstance(item, Mapping):
                if set(item) != {'time', 'route'}:
                    raise ValueError("A scheduled cue contains exactly time and route")
                timestamp, route = item['time'], item['route']
            else:
                timestamp, route = item
            if isinstance(timestamp, bool) or not np.isfinite(timestamp) or timestamp <= 0:
                raise ValueError("Cue times must be finite and strictly positive")
            frame = round(float(timestamp)/self.control_dt)
            if abs(frame*self.control_dt-float(timestamp)) > 1e-8:
                raise ValueError("Scheduled cue times must fall on the fixed control grid")
            if route not in ROUTE_NAMES:
                raise ValueError("Scheduled route must be bottom or top")
            if result and float(timestamp) <= result[-1]['time']:
                raise ValueError("Scheduled cues must have strictly increasing times")
            result.append({'time': float(timestamp), 'route': route, 'control_frame': int(frame)})
        return result

    def reset(self, *args, switch_schedule=None, seed=None, **kwargs):
        schedule = self._validate_schedule(switch_schedule)
        if seed is not None:
            # Repeated explicit seeds reproduce the same audio phase and start.
            self.rng = np.random.default_rng(int(seed))
        self._explicit_reset_seed = None if seed is None else int(seed)
        self._switch_schedule = schedule
        self._scheduled_cue_index = 0
        self._delivered_scheduled_cues = []
        self._pending_terminal_reward = 0.
        self._terminal_feedback_registered = False
        super().reset(*args, **kwargs)
        self._control_dt = self.control_dt
        self._register_terminal_feedback()
        return self.sensor_packet()

    def _register_terminal_feedback(self):
        if self._terminal and not self._terminal_feedback_registered:
            # _arrival_route_score is created only by the inherited real
            # <=10ms arrival scorer; the schedule never manufactures success.
            correct = bool(self._arrival_route_score and self._arrival_route_score['correct_arrival'])
            self._pending_terminal_reward = 1. if correct else 0.
            self._terminal_feedback_registered = True

    def _emit_due_schedule(self):
        packet = None
        while self._scheduled_cue_index < len(self._switch_schedule):
            cue = self._switch_schedule[self._scheduled_cue_index]
            if self.world.time < cue['time']-1e-8 or self._terminal:
                return packet
            # At fixed dt this matches the planned boundary, independent of
            # body position, actual route, reward, recall or model behavior.
            packet = super().emit_route_cue(cue['route'])
            self._delivered_scheduled_cues.append({**cue, 'emitted_time': float(self.world.time)})
            self._scheduled_cue_index += 1
        return packet

    def step(self, muscles, dt=None):
        duration = self.control_dt if dt is None else float(dt)
        if not np.isfinite(duration) or abs(duration-self.control_dt) > 1e-10:
            raise ValueError("Use the fixed control_dt; this preserves cue and contact timing")
        packet = super().step(muscles, duration)
        self._register_terminal_feedback()
        cue_packet = self._emit_due_schedule()
        return packet if cue_packet is None else cue_packet

    def consume_feedback(self):
        """Scalar post-action consequence, separately from the sensory packet.

        No route identity, distance, progress or coordinate is included. Rewards
        describe the environment consequence; a frozen evaluator should simply
        decline to pass them to a learner.
        """
        feedback = {'reward': float(self._pending_terminal_reward), 'terminal': bool(self._terminal)}
        self._pending_terminal_reward = 0.
        return feedback

    def advance(self, muscles, dt=None):
        packet = self.step(muscles, dt)
        return {'packet': packet, 'feedback': self.consume_feedback()}

    def private_metrics(self, **kwargs):
        metrics = super().private_metrics(**kwargs)
        metrics.update(integration_control_dt=self.control_dt,
            constructor_seed=self._constructor_seed, explicit_reset_seed=self._explicit_reset_seed,
            switch_schedule=copy.deepcopy(self._switch_schedule),
            delivered_scheduled_cues=copy.deepcopy(self._delivered_scheduled_cues),
            undelivered_scheduled_cues=len(self._switch_schedule)-self._scheduled_cue_index,
            terminal_feedback_registered=self._terminal_feedback_registered)
        return metrics


IntegrationEnv = IntegrationEnvironment


def _same_packet(a, b):
    assert set(a) == set(b) == {'observation', 'waveform'}
    assert set(a['observation']) == set(b['observation']) == SENSOR_KEYS
    for key in SENSOR_KEYS:
        assert np.array_equal(a['observation'][key], b['observation'][key])
    assert np.array_equal(a['waveform'], b['waveform'])


def self_check():
    result = {}
    plain = TexturedRouteEnv(seed=13)
    scheduled = IntegrationEnvironment(seed=13)
    a = plain.reset(route='bottom', start='left_middle')
    b = scheduled.reset(route='bottom', start='left_middle', switch_schedule=[(1., 'top'), (2., 'bottom')])
    _same_packet(a, b)
    for frame in range(26):
        action = np.array([.2, 0, .2, 0])
        a = plain.step(action, .1)
        b = scheduled.step(action)
        if frame+1 == 10:
            a = plain.emit_route_cue('top')
        if frame+1 == 20:
            a = plain.emit_route_cue('bottom')
        _same_packet(a, b)
        assert scheduled.world.get_state() == plain.world.get_state()
        actual, original = scheduled.private_metrics(), plain.private_metrics()
        assert all(actual[k] == value for k, value in original.items())
        assert scheduled.consume_feedback() == {'reward': 0., 'terminal': False}
    emitted = scheduled.private_metrics()['delivered_scheduled_cues']
    assert len(emitted) == 2 and all(abs(e['time']-e['emitted_time']) < 1e-8 for e in emitted)
    result['scheduled_cues_exactly_match_external_fixed_time_cues_and_physics'] = emitted
    assert np.count_nonzero(scheduled.sensor_packet()['waveform']) == 0
    initial = scheduled.reset(seed=19, route='bottom', switch_schedule=[(10., 'top')])
    repeated = scheduled.reset(seed=19, route='bottom', switch_schedule=[(10., 'bottom')])
    _same_packet(initial, repeated)
    alternate = copy.deepcopy(scheduled)
    alternate._switch_schedule[0]['route'] = 'top'  # evaluator-only paired fixture
    for _ in range(100):
        a, b = scheduled.step([0, 0, 0, 0]), alternate.step([0, 0, 0, 0])
        for key in SENSOR_KEYS:
            assert np.array_equal(a['observation'][key], b['observation'][key])
        assert scheduled.world.get_state() == alternate.world.get_state()
        if scheduled.world.time < 10.-1e-8:
            assert np.array_equal(a['waveform'], b['waveform'])
    assert not np.array_equal(a['waveform'], b['waveform'])
    result['same_seed_and_prefix_different_future_cues_do_not_leak'] = True

    # Fixed actuator fixtures exercise feedback after real physical passage and
    # first arrival. They are not model training or navigation demonstrations.
    actuator_fixture = [(45, [.45, 0, .45, 0]), (20, [0, 0, 0, 0]),
                        (13, [0, .3, .3, 0]), (10, [0, 0, 0, 0]),
                        (30, [.45, 0, .45, 0])]
    terminal_cases = []
    for required in ('bottom', 'top'):
        env = IntegrationEnvironment(seed=33)
        env.reset(route=required, start=(4., .55, 0.), switch_schedule=[(20., 'top')])
        rewards = []
        transition = None
        for count, muscles in actuator_fixture:
            for _ in range(count):
                transition = env.advance(muscles)
                assert set(transition['feedback']) == {'reward', 'terminal'}
                rewards.append(transition['feedback']['reward'])
                if transition['feedback']['terminal']:
                    break
            if transition['feedback']['terminal']:
                break
        assert transition['feedback']['terminal'], 'Fixed physical fixture did not reach red'
        metrics = env.private_metrics()
        assert metrics['used_passage'] == 'bottom'
        assert sum(rewards) == (1. if required == 'bottom' else 0.)
        assert env.consume_feedback() == {'reward': 0., 'terminal': True}
        assert not metrics['delivered_scheduled_cues']
        assert metrics['first_target_time'] <= metrics['elapsed_seconds']
        terminal_cases.append({'requested': required, 'used_passage': metrics['used_passage'],
            'first_target_time': metrics['first_target_time'], 'reward_sum': sum(rewards),
            'terminal_packet_returned': transition['packet']['observation']['ray_colors'].shape == (31, 3)})
    result['real_physics_terminal_reward_once_and_wrong_route_zero'] = terminal_cases
    for bad in ([(1.05, 'top')], [(1., 'top'), (1., 'bottom')], [(1., 'bad')]):
        try:
            IntegrationEnvironment().reset(switch_schedule=bad)
            raise AssertionError('Invalid schedule accepted')
        except ValueError:
            pass
    result['off_grid_duplicate_or_unknown_cues_rejected'] = True
    result['no_training_run'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results'/'integration_environment_audit.json')
    args = parser.parse_args()
    report = self_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
