"""Independent API, factual-causality and persistence units for integration.

All models are disposable small fixtures. No registered model is trained,
edited or replaced. Synthetic reward/motor fixtures are labelled as such.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from integrated_controller import IntegratedController
from integration_environment import IntegrationEnvironment
from route_state_digest import digest
from environment import SENSOR_KEYS


HERE=Path(__file__).resolve().parent


def fixture(capacity=32, schedule=None):
    brain=IntegratedController(seed=73,mixed_neurons=128,max_events=capacity,
                               exploration=False,reflex_enabled=False)
    env=IntegrationEnvironment(seed=79)
    env.reset(route="top",switch_schedule=schedule)
    return brain,env


def tick(brain,env,learning=True,reward=None):
    action,info=brain.observe_then_act(env.sensor_packet(),learning=learning)
    transition=env.advance(action)
    feedback=dict(transition["feedback"])
    if reward is not None:feedback["reward"]=reward
    outcome=brain.observe_outcome(transition["packet"],action,**feedback)
    return action,info,outcome


def synapses(brain):
    return digest((brain.core.excitatory.export("e"),brain.core.disinhibitory.export("d")))


def facts(brain):
    return digest({name:getattr(brain,name) for name in
        ("event_ids","event_previous","event_next","event_context","event_frame",
         "event_features","event_views","event_outcomes","event_terminal",
         "event_muscles","event_strength","feature_to_events")})


def reject_unchanged(brain,call):
    before=digest(brain)
    try:call()
    except (ValueError,RuntimeError,TypeError):pass
    else:raise AssertionError("Invalid call was accepted")
    assert digest(brain)==before,"Rejected call partially mutated the controller"


def pending_and_grounding():
    b,e=fixture()
    b.core.set_activity(np.ones(128),membrane=np.full(128,100.))
    before=synapses(b)
    action,info=b.observe_then_act(e.sensor_packet())
    assert b.pending is not None and len(b.event_slot_by_id)==0
    assert b.core.transition_count==0 and synapses(b)==before
    assert info["pfc_active_count"]>len(b.pending["features"])
    pre=b.pending["features"].copy()
    reject_unchanged(b,lambda:b.observe_then_act(e.sensor_packet()))
    transition=e.advance(action)
    wrong=action.copy();wrong[0]=1. if wrong[0]!=1. else 0.
    reject_unchanged(b,lambda:b.observe_outcome(transition["packet"],wrong))
    reject_unchanged(b,lambda:b.observe_outcome(transition["packet"],action,learning=False))
    out=b.observe_outcome(transition["packet"],action,**transition["feedback"])
    assert out["recorded_event"]==0 and len(b.event_slot_by_id)==1
    assert b.core.transition_count==1 and synapses(b)!=before
    post_values,_=b._observe(transition["packet"])
    _,post=b._mixed_activity(post_values)
    assert set(b.core.excitatory.rows)==set(map(int,pre))
    assert all(set(np.flatnonzero(row))==set(map(int,post)) for row in b.core.excitatory.rows.values())
    assert np.array_equal(b.event_outcomes[0],post_values)
    reject_unchanged(b,lambda:b.observe_outcome(transition["packet"],action))
    return dict(act_does_not_record_or_learn=True,pending_reentry_rejected=True,
        outcome_pairing_and_learning_mode_enforced=True,observed_transitions=1,
        strong_imagined_activity_excluded_from_learned_pre_post_edges=True,
        actual_pre_active=len(pre),imagined_pfc_active=info["pfc_active_count"],
        double_outcome_rejected=True)


def invalid_feedback_atomicity():
    cases=[]
    for name,feedback in (("one_dimensional_reward",{"reward":np.array([1.])}),
                          ("vector_terminal",{"terminal":np.array([True,False])}),
                          ("nonfinite_reward",{"reward":float("nan")})):
        b,e=fixture();a,_=b.observe_then_act(e.sensor_packet());t=e.advance(a)
        reject_unchanged(b,lambda:b.observe_outcome(t["packet"],a,**feedback))
        assert len(b.event_slot_by_id)==0 and b.core.transition_count==0 and b.pending is not None
        cases.append(name)
    return dict(rejected_without_partial_fact_or_core_update=cases)


def terminal_observation():
    b,e=fixture()
    e.reset(route="bottom",start=(9.1,1.5,np.pi))
    # Explicit physical initial momentum in a unit fixture, not a trained route.
    e.world.velocity[:]=[-1.,0.]
    a,_=b.observe_then_act(e.sensor_packet());t=e.advance(a)
    assert t["feedback"]["terminal"] and t["feedback"]["reward"]==0.
    result=b.observe_outcome(t["packet"],a,**t["feedback"])
    values,_=b._observe(t["packet"])
    assert b.event_terminal[0] and np.array_equal(values,b.event_outcomes[0])
    assert b.frame==1 and b.outcomes_observed==1 and b.core.transition_count==1
    assert b.core.advance_count==1 and b.pending is None
    return dict(real_first_arrival_terminal=True,post_terminal_sensation_saved=True,
                decisions=1,outcomes=1,phantom_decisions=0,reward=0.)


def fact_value_rule_separation():
    b,e=fixture()
    for _ in range(6):tick(b,e)
    a,info=b.observe_then_act(e.sensor_packet())
    assert info["fact_recall_events"] and info["fact_recall_unrewarded"]>0
    assert not info["recall_events"]
    t=e.advance(a);b.observe_outcome(t["packet"],a,**t["feedback"])
    b.receive_reward(1.) # disposable positive-value fixture, not environment success
    a,valued=b.observe_then_act(e.sensor_packet())
    assert valued["recall_events"]
    old=b.pending["features"].copy()
    t=e.advance(a);b.observe_outcome(t["packet"],a,**t["feedback"])
    before,_=b._observe(e.sensor_packet())
    e.switch_route("bottom")
    after,_=b._observe(e.sensor_packet());assert np.array_equal(before,after)
    a,new=b.observe_then_act(e.sensor_packet())
    common=np.intersect1d(old,b.pending["features"])
    assert len(common)>0 and b.shared_sensory[common].all()
    assert new["fact_recall_events"] and not new["recall_events"]
    assert new["goal_active"]==[0] and new["rule_active"]==[1]
    signatures={int(b.event_context[b.event_slot_by_id[i]]) for i in new["fact_recall_events"]}
    assert signatures=={33}
    return dict(unrewarded_facts_recalled=info["fact_recall_unrewarded"],
        same_rule_positive_value_recalled=len(valued["recall_events"]),
        cross_rule_shared_sensory_cells=len(common),
        cross_rule_facts_recalled=len(new["fact_recall_events"]),
        cross_rule_motor_value_events=0,synthetic_positive_value_fixture=True)


def check_links(b):
    ids=set(b.event_slot_by_id)
    for event_id,slot in b.event_slot_by_id.items():
        previous=int(b.event_previous[slot]);following=int(b.event_next[slot])
        assert previous==-1 or previous in ids
        assert following==-1 or following in ids
        if previous>=0:assert b.event_next[b.event_slot_by_id[previous]]==event_id
        if following>=0:assert b.event_previous[b.event_slot_by_id[following]]==event_id


def capacity_retirement():
    results=[]
    for capacity in (1,4):
        b,e=fixture(capacity)
        for _ in range(capacity):tick(b,e,reward=1.)
        assert np.all(b.event_strength>0)
        for _ in range(8):tick(b,e);check_links(b)
        assert b.next_event_id==capacity+8 and b.fact_capacity_retirements==8
        assert b.skipped_unconfirmed_events==0 and len(b.event_slot_by_id)==capacity
        assert b.core.excitatory.edge_count>0
        b.reset_activity();tick(b,e);check_links(b)
        newest=b.event_slot_by_id[b.next_event_id-1]
        assert b.event_previous[newest]==-1
        results.append(dict(capacity=capacity,new_facts_after_all_slots_rewarded=8,
            retirements_before_reset=8,dangling_or_asymmetric_links=0,
            reset_starts_new_chain=True,core_learned_connections_survive=True))
    return dict(fixtures=results,raw_fact_retirement_is_explicit_not_lossless_archiving=True)


def motor_competition():
    b,e=fixture(4)
    for _ in range(2):tick(b,e)
    b.event_strength[:2]=1.
    b.event_muscles[:2]=[[0.,.3,.3,0.],[.3,0.,0.,.3]]
    b.sequence_enabled=False;b.core_enabled=False
    values,_=b._observe(e.sensor_packet());_,external=b._mixed_activity(values)
    b.motor_competition_enabled=False
    mixed,available,before=b._cognitive_readout(values,external,np.zeros(b.mixed_neurons,bool))
    b.motor_competition_enabled=True
    competed,available,after=b._cognitive_readout(values,external,np.zeros(b.mixed_neurons,bool))
    assert np.allclose(mixed,[.15]*4)
    assert np.allclose(competed,[0.,.3,.3,0.]) or np.allclose(competed,[.3,0.,0.,.3])
    return dict(synthetic_equal_left_right_memory_fixture=True,
        before=mixed.tolist(),after=competed.tolist(),winner=after["motor_group"],
        group_mechanism="Five innate template groups with argmax competition; not an additional learned motor vocabulary.")


def pending_save_restore_and_freeze():
    b,e=fixture(schedule=[(.8,"bottom")])
    for _ in range(5):tick(b,e)
    a,info=b.observe_then_act(e.sensor_packet(),learning=True)
    with tempfile.TemporaryDirectory(prefix="integrated-audit-") as directory:
        path=Path(directory)/"pending.npz";b.save(path)
        with np.load(path,allow_pickle=False) as archive:
            assert all(not archive[n].dtype.hasobject for n in archive.files)
        restored=IntegratedController.load(path);other=copy.deepcopy(e)
        assert digest(b.pending)==digest(restored.pending)
        assert digest(b.core.snapshot())==digest(restored.core.snapshot())
        reject_unchanged(restored,lambda:restored.observe_then_act(other.sensor_packet()))
        for obj,world in ((b,e),(restored,other)):
            t=world.advance(a);obj.observe_outcome(t["packet"],a,**t["feedback"])
        for _ in range(10):
            aa,ia,oa=tick(b,e);ab,ib,ob=tick(restored,other)
            assert np.array_equal(aa,ab) and digest(ia)==digest(ib) and digest(oa)==digest(ob)
            assert digest(b.core.snapshot())==digest(restored.core.snapshot())
            assert facts(b)==facts(restored)
        before=(facts(b),synapses(b),b.core.maintenance_ticks,b.core.transition_count)
        for _ in range(6):tick(b,e,learning=False)
        after=(facts(b),synapses(b),b.core.maintenance_ticks,b.core.transition_count)
        assert before==after
    return dict(pending_decision_saved_and_resumed_once=True,
        online_continuation_frames=10,includes_real_midway_audio_change=True,
        complete_core_snapshots_and_facts_equal=True,frozen_physical_frames=6,
        frozen_facts_synapses_and_maintenance_unchanged=True,temporary_archive_removed=True)


def input_boundary():
    b,e=fixture();other=copy.deepcopy(b)
    packet=e.sensor_packet()
    assert set(packet)=={"observation","waveform"} and set(packet["observation"])==SENSOR_KEYS
    poisoned=copy.deepcopy(packet)
    poisoned.update(target="blue",route="bottom",goal_position=[0.,0.],reward=1.,trial_id=99)
    poisoned["observation"].update(position=[99.,99.],heading=99.,target="blue",route="bottom")
    a,ia=b.observe_then_act(packet);ab,ib=other.observe_then_act(poisoned)
    assert np.array_equal(a,ab) and digest(b)==digest(other)
    t=e.advance(a);poisoned=copy.deepcopy(t["packet"])
    poisoned.update(reward=1.,route="bottom",target="blue")
    b.observe_outcome(t["packet"],a,**t["feedback"])
    other.observe_outcome(poisoned,ab,**t["feedback"])
    assert digest(b)==digest(other)
    assert np.count_nonzero(b.event_strength)==0
    return dict(environment_emits_only_sensors_and_pcm=True,
        injected_private_labels_coordinates_and_packet_reward_ignored=True,
        explicit_outcome_reward_is_only_value_input=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results"/"integrated_controller_audit.json")
    args=parser.parse_args();started=time.perf_counter();checks={}
    functions=(pending_and_grounding,invalid_feedback_atomicity,terminal_observation,
        fact_value_rule_separation,capacity_retirement,motor_competition,
        pending_save_restore_and_freeze,input_boundary)
    for check in functions:
        try:checks[check.__name__]=dict(passed=True,evidence=check())
        except Exception as exc:checks[check.__name__]=dict(passed=False,error=type(exc).__name__+": "+str(exc))
    result=dict(schema_version=1,passed=all(x["passed"] for x in checks.values()),
        scope="Small disposable integration units, not navigation success or new training.",checks=checks,
        wall_seconds=time.perf_counter()-started,
        source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ("audit_integrated_controller.py","integrated_controller.py","recurrent_core.py","integration_environment.py")})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),passed=result["passed"],
        failed={n:v["error"] for n,v in checks.items() if not v["passed"]},
        wall_seconds=result["wall_seconds"]),ensure_ascii=False))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
