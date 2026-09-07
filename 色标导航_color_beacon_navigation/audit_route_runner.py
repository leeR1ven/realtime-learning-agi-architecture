"""Independent runner boundary units, not learned-navigation evidence.

Uses transparent test doubles solely to exercise reward and intervention order.
Real physical passage/sensor checks are in route_environment_audit.json; final
paired outcomes are independently re-counted elsewhere after training finishes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from route_switch_experiment import run_segment


HERE = Path(__file__).resolve().parent


class BoundaryEnvironment:
    def __init__(self, correct=False, terminate_frame=2, initial_time=0.):
        self.world = SimpleNamespace(time=initial_time, position=np.zeros(2), heading=0.)
        self.correct = correct
        self.terminate_frame = terminate_frame
        self.frame = 0

    def private_metrics(self):
        ended = self.frame >= self.terminate_frame
        return dict(terminated=ended, correct_arrival=ended and self.correct,
                    target_reached=ended, control_steps=self.frame)

    def sensor_packet(self):
        # Times supplied only to the test observer, never production controller.
        return dict(test_observation_time=self.world.time,
                    waveform=np.zeros(800))

    def step(self, muscles, dt):
        self.frame += 1
        self.world.time += dt
        return self.sensor_packet()

    def private_trajectory(self):
        return []


class BoundaryBrain:
    def __init__(self, environment):
        self.environment = environment
        self.event_ids = np.array([-1])
        self.event_strength = np.array([0.])
        self.rewards = []
        self.observations = []
        self.interventions = []
        self.last_reward_info = None

    def observe_then_act(self, packet, *, learning):
        self.observations.append((self.environment.world.time, len(self.interventions)))
        return np.zeros(4), dict(teaching=False, goal_active=[0], source='explicit_boundary_fixture')

    def receive_reward(self, reward):
        self.rewards.append((self.environment.frame, reward))
        self.last_reward_info = dict(reward=reward, test_frame=self.environment.frame)

    def intervene_rule_state(self, state):
        self.interventions.append((self.environment.world.time, np.asarray(state).tolist()))


def main():
    checks = []
    for correct, learning, reward_enabled, expected in (
            (True, True, True, [(2, 1.)]), (False, True, True, []),
            (True, False, False, []), (True, True, False, [])):
        environment = BoundaryEnvironment(correct=correct)
        brain = BoundaryBrain(environment)
        row = run_segment(environment, brain, 20, learning=learning, reward_enabled=reward_enabled)
        assert brain.rewards == expected and environment.frame == 2
        checks.append(dict(correct=correct, learning=learning, reward_enabled=reward_enabled,
                           reward_calls=brain.rewards, terminated_frame=environment.frame))
    environment = BoundaryEnvironment(terminate_frame=99, initial_time=10.)
    brain = BoundaryBrain(environment)
    row = run_segment(environment, brain, 10, learning=False, clear_rule_time=10.6)
    assert len(brain.interventions) == 1
    assert abs(brain.interventions[0][0]-10.6) < 1e-8
    assert [n for _, n in brain.observations] == [0]*6+[1]*4

    # Original score accepts an early first crossing if a later same-route
    # crossing precedes arrival. The predeclared protocol excludes this case.
    crossings = [dict(direction='left_to_right', route='bottom', time=10.3),
                 dict(direction='right_to_left', route='bottom', time=11.),
                 dict(direction='left_to_right', route='bottom', time=13.)]
    original = bool(crossings[-1]['time'] >= 10.6 and crossings[0]['route'] == 'bottom')
    protocol = bool(original and crossings[0]['time'] >= 10.6)
    assert original and not protocol
    report = dict(
        boundary_fixture_units=checks,
        rule_clear_time=brain.interventions[0][0],
        clear_happens_before_first_post_pulse_observation=True,
        original_strict_counterexample=dict(crossings=crossings, original_pass=original, protocol_pass=protocol),
        finding='Runner must additionally require first forward crossing time >=10.6; root accepted deterministic recount without retraining.',
        source_sha256=hashlib.sha256((HERE/'route_switch_experiment.py').read_bytes()).hexdigest(),
        static_review=dict(
            reward_only_after_physical_step_and_first_correct_arrival=True,
            prefix_fixed_100_frames_rejects_any_prior_forward_midline_or_leaving_left_approach=True,
            clone_before_each_intervention_includes_environment_controller_and_rng=True,
            no_cue_scores_retained_original_route=True,
            swapped_mapping_pairs_old_acoustics_with_new_private_requirement=True,
            clear_rule_also_clears_replay_activity_but_preserves_goal=True,
            trial_starts_crossed_with_both_routes=True,
            goal_preserved_is_a_diagnostic_activity_check_not_semantic_goal_understanding=True),
        limits=['Synthetic boundary units do not establish physical navigation success.',
                'Stay/no-cue success measures retention; no-cue failure is not required.',
                'Clearing rule and replay tests that combined state, not an isolated anatomical PFC.',
                'Pair success is a behavior test for two absolute cues, not general language or subgoal reasoning.'])
    output = HERE/'results'/'route_runner_boundary_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
