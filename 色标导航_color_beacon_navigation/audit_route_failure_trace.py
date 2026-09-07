"""Read-only connection trace of the registered 1031/top -> bottom failure.

Exactly one 986-frame replay. Instrumentation records the original method
outputs; it never supplies teacher actions, rewards, coordinates or labels to
the learner. Offline training coordinates only annotate already recalled IDs.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from associative_controller import Controller as BaseController
from route_switch_controller import RouteSwitchController
from route_switch_environment import RouteSwitchEnv
from route_state_digest import digest


HERE = Path(__file__).resolve().parent


def turn_current(muscles):
    m = np.asarray(muscles)
    return float((m[2] - m[3]) - (m[0] - m[1]))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class TraceController(RouteSwitchController):
    capture = False

    def _activate(self, values, audio):
        if self.capture:
            self.activation_trace = dict(
                values=values.copy(), audio=audio.copy(),
                goal_before=self.context.copy(), rule_before=self.rule_context.copy(),
                pfc_before=np.flatnonzero(self.pfc_activity),
                replay_before=self.last_recalled_ids.copy(),
                replay_weights_before=self.last_recalled_weights.copy())
        result = super()._activate(values, audio)
        if self.capture:
            self.activation_trace.update(
                goal_after=self.context.copy(), rule_after=self.rule_context.copy(),
                pfc_after=result[1].copy(),
                replay_after_activation=self.last_recalled_ids.copy(),
                replay_weights_after_activation=self.last_recalled_weights.copy())
        return result

    def _activity_votes(self, active):
        result = super()._activity_votes(active)
        if self.capture:
            self.vote_outputs.append(result.copy())
        return result

    def _select_events(self, scores):
        result = super()._select_events(scores)
        if self.capture:
            self.final_scores = scores.copy()
        return result

    def _recall(self, active):
        if self.capture:
            self.vote_outputs = []
        result = super()._recall(active)
        if self.capture:
            self.recall_trace = result
        return result


def memory_hash(brain):
    names = ("context_source", "rule_source", "view_sources", "thresholds", "levels",
             "event_ids", "event_previous", "event_next", "event_frame", "event_context",
             "event_features", "event_views", "event_muscles", "event_strength", "event_kind",
             "event_active_count", "feature_to_events", "readout_feature_to_events")
    return digest({name: getattr(brain, name) for name in names})


def original_branch(report):
    pair = next(p for p in report["pairs"] if p["seed"] == 1031 and p["initial_route"] == "top")
    return pair, pair["branches"]["switch"]


def inspect_recall(brain, current_position, current_heading, training, ends, catalog):
    active = brain.activation_trace["pfc_after"]
    values = brain.activation_trace["values"]
    selected, weights, current, _, peak, feedback_active, sequence = brain.recall_trace
    primary = brain.vote_outputs[0]
    primary_peak = float(primary.max(initial=0.))
    feedback = brain.vote_outputs[1] if len(brain.vote_outputs) > 1 else np.zeros_like(primary)
    feedback_peak = float(feedback.max(initial=0.))
    entries = []
    signed = []
    for slot, weight in zip(selected, weights):
        event_id = int(brain.event_ids[slot])
        stored_frame = int(brain.event_frame[slot])
        episode = bisect.bisect_right(ends, stored_frame)
        begin = 0 if episode == 0 else ends[episode-1]
        local_frame = stored_frame-begin
        train = training[episode]
        truth = train["trajectory"][local_frame]
        assert int(brain.event_kind[slot]) == 1
        assert train["events_before"] <= event_id < train["events_after"]
        if str(event_id) not in catalog:
            catalog[str(event_id)] = dict(slot=int(slot), event_id=event_id,
                event_context=int(brain.event_context[slot]), event_frame=stored_frame,
                previous=int(brain.event_previous[slot]), next=int(brain.event_next[slot]),
                strength=float(brain.event_strength[slot]),
                event_muscles=brain.event_muscles[slot].tolist(),
                event_views=brain.event_views[slot].tolist(),
                event_pfc_ids=brain.event_features[slot].tolist(),
                offline_truth=dict(training_episode=episode, local_action_frame=local_frame,
                    route_request=train["requested_route"], rewarded=train["correct_arrival"],
                    training_start=train["start_name"], position_before_action=truth["position"],
                    heading_before_action=truth["heading"], time_before_action=truth["time"]))
        incoming = np.intersect1d(active, brain.event_features[slot], assume_unique=True)
        contributions = [1 / np.sqrt(len(brain.readout_feature_to_events[int(cell)])) /
                         np.sqrt(float(brain.event_active_count[slot])) for cell in incoming]
        fb = (.08 * primary_peak * float(feedback[slot]) / feedback_peak
              if feedback_peak > 0 and primary[slot] >= .8*primary_peak else 0.)
        diff = values-brain.event_views[slot]
        muscle = brain.event_muscles[slot]
        signed.append(float(weight)*turn_current(muscle))
        entries.append(dict(event_id=event_id, weight=float(weight),
            current_to_event_score=float(primary[slot]), feedback_score_addition=fb,
            sequence_score_addition=float(brain.final_scores[slot])-float(primary[slot])-fb,
            final_event_score=float(brain.final_scores[slot]),
            active_incoming_pfc_ids=incoming.tolist(),
            incoming_pfc_degree_normalized_contributions=contributions,
            incoming_sum_reconstructed=float(sum(contributions)),
            muscle_contribution=(float(weight)*muscle).tolist(),
            signed_turn_contribution=signed[-1],
            value_rms=float(np.sqrt(np.mean(diff**2))),
            depth_rms=float(np.sqrt(np.mean(diff[:31]**2))),
            rgb_rms=float(np.sqrt(np.mean(diff[31:124]**2))),
            body_rms=float(np.sqrt(np.mean(diff[124:]**2))),
            offline_position_distance=float(np.linalg.norm(np.asarray(truth["position"])-current_position)),
            offline_heading_difference=float((truth["heading"]-current_heading+np.pi)%(2*np.pi)-np.pi)))
    absolute = float(sum(abs(x) for x in signed))
    torque = turn_current(current)
    return dict(events=entries, recalled_muscle_current=np.asarray(current).tolist(),
        primary_peak=primary_peak, final_peak=float(peak),
        feedback_pfc_ids=feedback_active.tolist(), sequence_supported_event_ids=sequence,
        left_turn_memory_count=sum(x > 1e-9 for x in signed),
        right_turn_memory_count=sum(x < -1e-9 for x in signed),
        signed_turn_current=torque, weighted_absolute_turn_current=absolute,
        opposing_turn_cancellation_fraction=1-abs(torque)/absolute if absolute > 1e-9 else 0.,
        left_muscle_antagonism=2*float(min(current[0], current[1])),
        right_muscle_antagonism=2*float(min(current[2], current[3])))


def without_sequence_same_frame(brain):
    primary = brain.vote_outputs[0]
    peak = float(primary.max(initial=0.))
    scores = primary.copy()
    supported = primary >= .8*peak
    if len(brain.vote_outputs) > 1:
        feedback = brain.vote_outputs[1]
        maximum = float(feedback.max(initial=0.))
        if maximum > 0:
            scores[supported] += .08*peak*feedback[supported]/maximum
    scores[~supported] = 0
    selected, weights, muscles, _, _ = BaseController._select_events(brain, scores)
    original = brain.recall_trace
    return dict(intervention="Remove only the sequence score addition in this frame's readout.",
        no_extra_physical_rollout=True, no_state_or_weights_modified=True,
        original_event_ids=brain.event_ids[original[0]].tolist(),
        original_weights=np.asarray(original[1]).tolist(),
        original_muscles=np.asarray(original[2]).tolist(),
        original_turn_current=turn_current(original[2]),
        no_sequence_event_ids=brain.event_ids[selected].tolist(),
        no_sequence_weights=weights.tolist(), no_sequence_muscles=muscles.tolist(),
        no_sequence_turn_current=turn_current(muscles),
        muscle_l2_difference=float(np.linalg.norm(np.asarray(original[2])-muscles)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"results"/"route_failure_trace_1031_top_switch.json")
    args = parser.parse_args()
    report_path = HERE/"results"/"route_switch_evaluation.json"
    checkpoint = HERE/"results"/"route_switch_evaluation.npz"
    original = json.loads(report_path.read_text(encoding="utf-8"))
    pair, expected = original_branch(original)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == original["checkpoint_sha256"]
    brain = TraceController.load(checkpoint)
    brain.reset_activity()
    brain.rng = np.random.default_rng(1031+100000)
    memory_before = memory_hash(brain)
    env = RouteSwitchEnv(seed=1031)
    env.reset(start="left_middle", target="red", route="top")
    training = original["training"]
    ends = np.cumsum([row["control_steps"] for row in training]).tolist()
    assert ends[-1] == int(brain.frame)
    for row, end in zip(training, ends):
        assert row["events_after"] == end
    total = expected["control_steps"]
    crossing = expected["passages"][0]["time"]
    cross_frame = int(crossing/.1)
    sampling = {99, 150, 200, 225, 300, 400, 600, 800, 900}
    sampling.update(range(100, 131))
    sampling.update(range(cross_frame-15, cross_frame+10))
    sampling.update(range(total-6, total))
    catalog, cells, detailed, brief = {}, {}, [], []
    intervention = None
    started = time.perf_counter()
    state_names = ("escape_back_steps", "escape_turn_steps", "escape_action",
                   "exploration_left", "exploration_action", "exploration_burst")
    for frame in range(total):
        if frame == 100:
            assert env.private_trajectory() == pair["prefix"]["trajectory"]
            env.switch_route("bottom")
        brain.capture = frame in sampling
        position = env.world.position.copy()
        heading = float(env.world.heading)
        time_before = float(env.world.time)
        behavior_state = {key: int(getattr(brain, key)) for key in state_names}
        activation_before = env.world.activations.copy()
        muscles, info = brain.observe_then_act(env.sensor_packet(), learning=False)
        assert not info["teaching"]
        row = dict(frame=frame, time=time_before, position=position.tolist(), heading=heading,
            goal_active=info["goal_active"], rule_active=info["rule_active"],
            pfc_count=info["pfc_active_count"], recall_count=info["recall_count"],
            source=info["source"], muscles=muscles.tolist(),
            memory_turn_current=turn_current(info["muscle_currents"]),
            actual_command_turn_current=turn_current(muscles),
            sequence_supported_count=len(info["sequence_supported_events"]))
        brief.append(row)
        if brain.capture:
            detail = dict(row, activation=jsonable(brain.activation_trace),
                recall=inspect_recall(brain, position, heading, training, ends, catalog),
                behavior_state_before=behavior_state,
                behavior_state_after={key:int(getattr(brain,key)) for key in state_names},
                actual_actuators_before=activation_before.tolist(),
                actual_torque_before=env.world.config.max_muscle_force*env.world.config.lever_arm*turn_current(activation_before),
                replay_after_action_ids=brain.last_recalled_ids.tolist(),
                replay_after_action_weights=brain.last_recalled_weights.tolist())
            old = brain.activation_trace["pfc_before"]
            new = brain.activation_trace["pfc_after"]
            detail["pfc_old_new_intersection"] = np.intersect1d(old,new).tolist()
            for cell in new:
                cell = int(cell)
                if str(cell) not in cells:
                    cells[str(cell)] = dict(goal_source=int(brain.context_source[cell]),
                        rule_source=int(brain.rule_source[cell]),
                        receptor_sources=brain.view_sources[cell].tolist(),
                        threshold=float(brain.thresholds[cell]))
            if (intervention is None and frame >= 106 and info["source"] == "associative_memory"
                    and info["sequence_supported_events"]):
                intervention = dict(frame=frame,time=time_before,**without_sequence_same_frame(brain))
            detailed.append(detail)
        env.step(muscles,.1)
        if brain.capture:
            detailed[-1].update(position_after=env.world.position.tolist(),
                heading_after=float(env.world.heading), actual_actuators_after=env.world.activations.tolist(),
                actual_torque_after=env.world.config.max_muscle_force*env.world.config.lever_arm*turn_current(env.world.activations))
    replay_seconds = time.perf_counter()-started
    assert env.private_trajectory() == expected["trajectory"]
    assert env.private_metrics()["passages"] == expected["passages"]
    assert env.private_metrics()["arrival_route_score"] == expected["arrival_route_score"]
    assert memory_hash(brain) == memory_before
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    result = dict(schema_version=1, replay=dict(seed=1031,initial_route="top",new_route="bottom",
        switch_frame=100,brain_rng_seed=101031,total_control_frames=total,
        complete_trajectory_matches_formal_branch_exactly=True,all_fixed_and_learned_memory_unchanged=True,
        checkpoint_file_unchanged=True,replay_wall_seconds=replay_seconds),
        outcome={key:env.private_metrics()[key] for key in ("target_reached","correct_arrival","chosen_route","passages","arrival_route_score")},
        method=dict(sampled_frames=sorted(sampling), all_frame_summary_count=len(brief),
            teacher_calls=0,reward_calls=0, extra_rollouts=0,
            offline_position_annotation_only=True,
            receptor_index_decode="channel=id//32, level=(id//2)%16, side=id%2; side 0 is >= threshold, side 1 is < threshold",
            receptor_levels=brain.levels.tolist(),
            turn_sign="(right_forward-right_backward)-(left_forward-left_backward); positive turns left/CCW",
            same_frame_intervention_selection="First sampled frame >=10.6s with sequence support and actual associative-memory action."),
        frame_details=detailed, all_frames=brief, remembered_events=catalog,
        active_pfc_birth_connections=cells, same_frame_sequence_intervention=intervention,
        source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ("audit_route_failure_trace.py","route_switch_controller.py","associative_controller.py","route_switch_environment.py")},
        checkpoint_sha256=checkpoint_hash,
        limitations=["This is one registered failed trial, not a new success benchmark.",
            "One-frame sequence removal explains a local current change; it does not predict a complete counterfactual route.",
            "Offline true positions annotate memories after selection and never enter the learner."])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),replay_frames=total,
        replay_seconds=replay_seconds, sampled_frames=len(detailed),catalog_events=len(catalog),
        exact_original_trajectory=True,intervention_frame=None if intervention is None else intervention["frame"]),ensure_ascii=False))


if __name__ == "__main__":
    main()
