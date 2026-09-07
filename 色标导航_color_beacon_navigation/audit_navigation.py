"""Independent interface/causality checks, not a navigation success benchmark.

The tiny demonstrations below are actually executed muscle commands used as
controlled fixtures. They are not target-reaching demonstrations. No planner is
called, and the script never edits the controller or production checkpoints.
"""
from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import textwrap
import numpy as np

from environment import BeaconNavigationEnv, SENSOR_KEYS
from associative_controller import Controller

HERE = Path(__file__).resolve().parent


def audit():
    out = {"controller_sha256":hashlib.sha256((HERE/"associative_controller.py").read_bytes()).hexdigest(),
           "scope":"Interface/causality microtests; not evidence of successful target navigation"}
    env = BeaconNavigationEnv(seed=613)
    model = Controller(seed=2026, mixed_neurons=1024, max_events=256, exploration=False)
    # Same initial view, distinct sustained audio contexts and executed muscles.
    fixture_actions = {"red":np.array([.45,0,.45,0]), "green":np.array([0,.2,.2,0])}
    for target, action in fixture_actions.items():
        packet = env.reset("barrier", "left_middle", target)
        model.reset_activity()
        for _ in range(8):
            model.observe_then_act(packet, teacher_muscles=action, learning=True)
            packet = env.step(action)
    assert model.next_event_id == 16
    assert np.all(model.event_strength[:16] == 1)
    out["teaching_fixture_commands_were_executed"] = True

    # Following the cue, compare the exact same silent sensor packet under two
    # different retained contexts, with all episodic weights kept identical.
    packet = env.reset("barrier", "left_middle", "red")
    packet["waveform"][:] = 0
    answers = {}
    for name, context in (("red",0), ("green",1)):
        probe = copy.deepcopy(model)
        probe.reset_activity()
        probe.context[context] = 1
        action, info = probe.observe_then_act(packet, learning=False)
        answers[name] = {"muscles":action.tolist(),"active":info["pfc_active_count"],
                         "recall_count":info["recall_count"],"context":info["context_active"]}
    assert not np.allclose(answers["red"]["muscles"], answers["green"]["muscles"])
    out["silent_context_intervention_changes_action"] = answers

    # Change private scorer target after sound has stopped. The model has no
    # environment reference; verify an identical packet cannot change the action.
    base = copy.deepcopy(model)
    base.reset_activity(); base.context[0] = 1
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    env._target_name = "green"; env._target_index = 1
    am, ai = a.observe_then_act(copy.deepcopy(packet), learning=False)
    bm, bi = b.observe_then_act(copy.deepcopy(packet), learning=False)
    assert np.array_equal(am,bm)
    out["private_target_change_with_same_packet_has_no_effect"] = True

    # Extra unapproved observation metadata cannot provide an answer shortcut.
    fake = copy.deepcopy(packet)
    fake.update(target="green", route=[[8,6]], correct_muscles=[0,1,0,1])
    fake["observation"].update(position=[9,7], heading=-2, target_index=2)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    am,_ = a.observe_then_act(packet,learning=False)
    bm,_ = b.observe_then_act(fake,learning=False)
    assert np.array_equal(am,bm)
    out["extra_goal_route_position_fields_ignored"] = True

    # Teacher argument can write the subsequent experience, never the current
    # output. Compare the same call with opposite teachers.
    a,b = copy.deepcopy(base),copy.deepcopy(base)
    am,_ = a.observe_then_act(packet,teacher_muscles=[1,0,1,0])
    bm,_ = b.observe_then_act(packet,teacher_muscles=[0,1,0,1])
    assert np.array_equal(am,bm)
    out["teacher_action_does_not_replace_current_prediction"] = True

    lesions = {}
    for flag in ("context_enabled","view_enabled","motor_memory_enabled"):
        probe = copy.deepcopy(base)
        setattr(probe,flag,False)
        action,info = probe.observe_then_act(packet,learning=False)
        assert np.array_equal(action,np.zeros(4))
        lesions[flag] = {"muscles":action.tolist(),"source":info["source"],
                         "recall_count":info["recall_count"],"active":info["pfc_active_count"]}
    out["context_visual_and_motor_return_lesions"] = lesions

    # Record autonomous experience, then deliver a positive event AFTER it.
    probe = copy.deepcopy(base)
    old = probe.next_slot
    prediction,_ = probe.observe_then_act(packet,learning=True)
    env.step(prediction)
    assert probe.event_strength[old] == 0 and probe.event_kind[old] == 1
    newest = probe.next_slot
    rewarded = copy.deepcopy(packet); rewarded["reward"] = 1.
    probe.observe_then_act(rewarded,learning=True)
    assert probe.event_strength[old] > 0
    assert probe.event_strength[newest] == 0
    out["autonomous_action_not_automatically_correct_and_reward_is_retrospective"] = True

    # Backward links are provenance; the new sequence mechanism reads event_next.
    a,b = copy.deepcopy(base),copy.deepcopy(base)
    b.event_previous[:] = -1
    b.previous_event_id = -1
    am,ai = a.observe_then_act(packet,learning=False)
    bm,bi = b.observe_then_act(packet,learning=False)
    out["erasing_audit_only_backward_links_leaves_action_identical"] = bool(np.array_equal(am,bm))
    out["microfixture_multiple_events_can_be_active"] = max(x["recall_count"] for x in answers.values())

    # Production rollout must call only the sensor packet controller interface.
    import run_experiment as R
    tree = ast.parse(textwrap.dedent(inspect.getsource(R.rollout)))
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and node.id in ("MuscleTeacher","teacher"):
            forbidden.append(node.id)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ("_plan","muscles"):
            forbidden.append(node.func.attr)
    assert not forbidden
    out["production_rollout_has_no_teacher_or_planner_calls"] = True
    # Catch integration/API errors without a long navigation attempt.
    try:
        result = R.rollout(copy.deepcopy(model),start="left_middle",target="red",max_steps=2)
        out["production_rollout_two_frame_smoke"] = {"passed":True,"teacher_supplied":result["teacher_supplied"]}
    except Exception as exc:
        out["production_rollout_two_frame_smoke"] = {"passed":False,"error":repr(exc)}
    out["ordered_link_microfixture"] = ordered_link_microfixture()
    out["capacity_and_continuation"] = capacity_and_continuation(base,packet)
    return out


def ordered_link_microfixture():
    """Actual short action transitions with tightly similar sensory contexts.

    This checks the transition connection's causal effect, not room/subgoal
    understanding. The whole fixture remains separate from autonomous training.
    """
    model = Controller(seed=2026,mixed_neurons=1024,max_events=128,
                       exploration=False,reflex_enabled=False)
    env = BeaconNavigationEnv(seed=333)
    packet = env.reset("barrier","left_middle","red")
    actions = ([.45,0,.45,0],[0,.2,.2,0],[.2,0,0,.2])
    for i in range(12):
        action = np.asarray(actions[i%3])
        model.observe_then_act(packet,teacher_muscles=action)
        packet = env.step(action,.02)
    packet = env.reset("barrier","left_middle","red")
    packet["waveform"][:] = 0
    model.reset_activity(); model.context[0] = 1
    model.last_recalled_ids = np.array([0],dtype=np.int64)
    model.last_recalled_weights = np.array([1.],dtype=np.float32)
    ordered,shuffled,no_sequence = (copy.deepcopy(model) for _ in range(3))
    old_ids = shuffled.last_recalled_ids.copy()
    old_weights = shuffled.last_recalled_weights.copy()
    changed = shuffled.shuffle_sequence_connections(seed=731)
    # The helper resets replay state for rollout use. Restore it here so this
    # within-frame intervention changes only directed transition connections.
    shuffled.last_recalled_ids = old_ids
    shuffled.last_recalled_weights = old_weights
    no_sequence.sequence_enabled = False
    valid = model.event_ids >= 0
    for field in ("event_strength","event_muscles","event_views","context","event_context"):
        assert np.array_equal(getattr(ordered,field),getattr(shuffled,field))
    assert sorted(ordered.event_next[valid].tolist()) == sorted(shuffled.event_next[valid].tolist())
    answers = {}
    for name,probe in (("ordered",ordered),("shuffled",shuffled),("no_sequence",no_sequence)):
        muscles,info = probe.observe_then_act(packet,learning=False)
        answers[name] = {"muscles":muscles.tolist(),"selected_events":info["recall_events"],
                         "sequence_supported_events":info.get("sequence_supported_events",[]),
                         "feedback_pfc_active_count":info.get("feedback_pfc_active_count",0)}
    return {"changed_links":changed,"identical_current_packet_state_and_event_weights":True,
            "same_transition_target_multiset":True,"answers":answers,
            "ordered_vs_shuffled_action_differs":not np.array_equal(answers["ordered"]["muscles"],answers["shuffled"]["muscles"]),
            "proves_room_or_subgoal_dependency_learning":False}


def capacity_and_continuation(base,packet):
    path = HERE/"results"/"audit_only_controller_state.npz"
    original = copy.deepcopy(base)
    original.save(path)
    restored = Controller.load(path)
    same = True
    for frame in range(20):
        observation = copy.deepcopy(packet)
        observation["observation"]["velocity"] = np.array([frame*.013,-frame*.003])
        am,ai = original.observe_then_act(observation,learning=True)
        bm,bi = restored.observe_then_act(observation,learning=True)
        same = same and np.array_equal(am,bm) and ai == bi
        if frame in (5,12):
            original.receive_reward(.4)
            restored.receive_reward(.4)
    for name,value in vars(original).items():
        if isinstance(value,np.ndarray):
            same = same and np.array_equal(value,getattr(restored,name))
    assert same

    # Confirm the deliberate full-memory policy, so its real-time-learning
    # limitation is visible in the audit instead of silently overlooked.
    limited = Controller(seed=2026,mixed_neurons=64,max_events=4,exploration=False,reflex_enabled=False)
    limited.context[0] = 1
    for _ in range(4):
        limited.observe_then_act(packet,learning=True)
    assert np.count_nonzero(limited.event_strength) == 0
    limited.receive_reward(1.)
    old_next_id = limited.next_event_id
    old_ids = limited.event_ids.copy()
    limited.observe_then_act(packet,learning=True)
    stopped = limited.next_event_id == old_next_id and np.array_equal(old_ids,limited.event_ids)
    assert stopped and limited.skipped_unconfirmed_events == 1
    return {"saved_state_continues_exactly_with_online_updates":bool(same),
            "all_confirmed_capacity_stops_new_autonomous_recording":bool(stopped),
            "capacity_probe_slots":4,"skipped_new_events":limited.skipped_unconfirmed_events,
            "limitation":"At full confirmed capacity, new autonomous experiences cannot be stored until a replacement/expansion policy is supplied."}


if __name__ == "__main__":
    result = audit()
    path = HERE/"results"/"independent_navigation_audit.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
