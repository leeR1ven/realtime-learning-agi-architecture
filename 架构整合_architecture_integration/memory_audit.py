"""Read-only audit of existing navigation memory plus small disposable units.

No navigation training is rerun. Synthetic positive signals below are confined
to newborn unit fixtures and never applied to either registered checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
NAV = HERE.parent/"色标导航_color_beacon_navigation"
sys.path.insert(0, str(NAV))
from route_switch_controller import RouteSwitchController
from route_switch_environment import RouteSwitchEnv
from route_state_digest import digest


def fixture(capacity=64):
    brain = RouteSwitchController(seed=53, mixed_neurons=512, max_events=capacity,
                                  exploration=False, reflex_enabled=False)
    env = RouteSwitchEnv(seed=59)
    env.reset(route="top")
    return brain, env


def record(brain, env, count):
    info = None
    for _ in range(count):
        muscles, info = brain.observe_then_act(env.sensor_packet(), learning=True)
        assert not info["teaching"]
        env.step(muscles,.1)
    return info


def links(brain):
    occupied = np.flatnonzero(brain.event_ids >= 0)
    ids = set(int(x) for x in brain.event_ids[occupied])
    dangling = []
    for slot in occupied:
        for field in ("event_previous", "event_next"):
            target = int(getattr(brain, field)[slot])
            if target >= 0 and target not in ids:
                dangling.append(dict(event_id=int(brain.event_ids[slot]),field=field,missing_target=target))
    return dict(stored_ids=sorted(ids),dangling_references=dangling)


def unit_checks():
    brain, env = fixture()
    last = record(brain,env,12)
    active = np.flatnonzero(brain.pfc_activity)
    factual_links = sum(len(brain.feature_to_events[int(i)]) for i in active)
    gated_links = sum(len(brain.readout_feature_to_events[int(i)]) for i in active)
    assert factual_links > 0 and gated_links == 0 and last["recall_count"] == 0
    before = dict(stored_facts=12,active_fact_links=factual_links,active_readout_links=gated_links,
                  repeated_input_recall_count=last["recall_count"],
                  exposed_novelty_fields=[k for k in last if "novel" in k.lower()])
    # This scalar is a controlled unit intervention, not a task success.
    brain.receive_reward(1.)
    _, rewarded = brain.observe_then_act(env.sensor_packet(),learning=False)
    assert rewarded["recall_count"] > 0
    values_before, _ = brain._observe(env.sensor_packet())
    pfc_before = np.flatnonzero(brain.pfc_activity)
    env.switch_route("bottom")
    values_after, _ = brain._observe(env.sensor_packet())
    _, switched = brain.observe_then_act(env.sensor_packet(),learning=False)
    assert np.array_equal(values_before,values_after)
    assert switched["recall_count"] == 0 and not np.intersect1d(pfc_before,np.flatnonzero(brain.pfc_activity)).size
    gating = dict(unrewarded=before,
        same_input_recall_after_synthetic_reward=rewarded["recall_count"],
        identical_view_new_rule_recall_count=switched["recall_count"],
        identical_view_new_rule_pfc_overlap=0,goal_preserved=switched["goal_active"] == [0],
        fact="Reward gates factual access; the fixed rule conjunction also prevents retrieval of the identical view learned under another rule.")

    small, small_env = fixture(4)
    record(small,small_env,1)
    small.receive_reward(1.)
    record(small,small_env,5)
    before_save = links(small)
    assert before_save["dangling_references"]
    with tempfile.TemporaryDirectory(prefix="memory-audit-") as folder:
        temp = Path(folder)/"unit.npz"
        small.save(temp)
        restored = RouteSwitchController.load(temp)
        assert links(restored) == before_save
        assert digest(small.rng) == digest(restored.rng)
        other_env = copy.deepcopy(small_env)
        for _ in range(8):
            a,ia=small.observe_then_act(small_env.sensor_packet(),learning=True)
            b,ib=restored.observe_then_act(other_env.sensor_packet(),learning=True)
            assert np.array_equal(a,b) and digest(ia)==digest(ib)
            small_env.step(a,.1);other_env.step(b,.1)
        assert np.array_equal(small.event_ids,restored.event_ids)
        assert np.array_equal(small.event_next,restored.event_next)
    full, full_env = fixture(4)
    record(full,full_env,4)
    full.receive_reward(1.)
    record(full,full_env,8)
    assert full.next_event_id == 4 and full.skipped_unconfirmed_events == 8
    capacity = dict(mixed_reward_and_unrewarded_capacity_four=before_save,
        save_load_preserves_the_existing_dangling_references=True,
        save_load_continuation_eight_frames_exact=True,
        fully_rewarded_capacity_four=dict(attempted_new_facts=8,new_facts_stored=0,
            skipped_unconfirmed_events=full.skipped_unconfirmed_events),
        attribution="Dangling edges originate in slot recycling, not in the tested save/load operation.")

    motor, _ = fixture(2)
    motor.event_strength[:] = 1.
    motor.event_muscles[:] = [[0.,.3,.3,0.],[.3,0.,0.,.3]]
    selected,weights,current,_,_ = motor._select_events(np.ones(2,dtype=np.float32))
    turn=float((current[2]-current[3])-(current[0]-current[1]))
    assert len(selected)==2 and np.allclose(current,[.15]*4) and turn==0
    return dict(synthetic_fixture_reward_only=True,registered_models_modified=False,
        fact_and_rule_gating=gating,capacity_and_restore=capacity,
        opposing_motor_fixture=dict(weights=weights.tolist(),muscles=current.tolist(),
            left_net_force_units=float(current[0]-current[1]),right_net_force_units=float(current[2]-current[3]),
            turn_current=turn,
            scope="Numerical readout fixture. Real body reflex/exploration can override it; this is not the main cause of the registered 25.43s wrong passage."))


def archive_audit(stem):
    rp=NAV/"results"/(stem+".json");cp=rp.with_suffix(".npz")
    report=json.loads(rp.read_text(encoding="utf-8"))
    training=report["training"]
    starts=np.r_[0,np.cumsum([r["control_steps"] for r in training])]
    poses=np.asarray([x["position"] for r in training for x in r["trajectory"][:-1]])
    headings=np.asarray([x["heading"] for r in training for x in r["trajectory"][:-1]])
    with np.load(cp,allow_pickle=False) as archive:
        meta=json.loads(archive["metadata"].tobytes().decode("utf-8"))
        occupied=archive["event_ids"]>=0
        event_ids=archive["event_ids"][occupied]
        frames=archive["event_frame"][occupied]
        strength=archive["event_strength"][occupied]
        contexts=archive["event_context"][occupied]
        previous=archive["event_previous"][occupied]
        following=archive["event_next"][occupied]
        assert len(poses)==starts[-1] and frames.max()<len(poses)
        assert np.array_equal(event_ids,frames)
        ids=set(int(i) for i in event_ids)
        dangling=sum(int(i)>=0 and int(i) not in ids for a in (previous,following) for i in a)
        coverage=[]
        for start in ("middle","lower","upper"):
            for route in ("bottom","top"):
                episodes=[r for r in training if r["start_name"]==start and r["requested_route"]==route]
                mask=np.zeros(len(frames),dtype=bool)
                for row in episodes:
                    i=row["episode"];mask|=(frames>=starts[i])&(frames<starts[i+1])
                coverage.append(dict(start=start,route=route,episodes=len(episodes),
                    successes=sum(r["correct_arrival"] for r in episodes),
                    recorded_facts=int(mask.sum()),reward_accessible_facts=int(np.count_nonzero(mask&(strength>0)))))
        branch_coverage=[]
        for pair in report["pairs"]:
            if not pair["valid_prefix"]:continue
            target="top" if pair["initial_route"]=="bottom" else "bottom"
            signature=33 if target=="top" else 17
            pose=pair["prefix"]["trajectory"][-1]
            distance=np.linalg.norm(poses[frames]-pose["position"],axis=1)
            angle=np.abs((headings[frames]-pose["heading"]+np.pi)%(2*np.pi)-np.pi)
            route=contexts==signature
            row=dict(seed=pair["seed"],initial_route=pair["initial_route"],new_route=target,
                switch_success=pair["branches"]["switch"]["strict_success"],
                prefix_position=pose["position"],prefix_heading=pose["heading"])
            for radius,degrees in ((.5,30),(1.,60)):
                near=route&(distance<=radius)&(angle<=np.deg2rad(degrees))
                row[f"within_{radius}m_{degrees}deg"]=dict(recorded_facts=int(near.sum()),
                    reward_accessible_facts=int(np.count_nonzero(near&(strength>0))))
            branch_coverage.append(row)
        inventory=dict(max_events=meta["max_events"],stored_events=len(event_ids),
            reward_accessible_events=int(np.count_nonzero(strength>0)),
            stored_but_reward_blocked_events=int(np.count_nonzero(strength<=0)),
            teacher_events=int(np.count_nonzero(archive["event_kind"][occupied]==2)),
            dangling_time_references=int(dangling),
            skipped_unconfirmed_events=int(meta["skipped_unconfirmed_events"]))
    rewarded=[r for r in training if r["correct_arrival"]]
    credited=[dict(episode=r["episode"],route=r["requested_route"],steps=r["control_steps"],
        strengthened=r["reward_details"]["strengthened_events"],passages=r["passages"],
        source_counts=r["source_counts"]) for r in rewarded]
    return dict(summary=report["summary"],inventory=inventory,
        training_start_route_coverage=coverage,
        all_successful_episodes_entire_recorded_episode_reinforced=all(r["steps"]==r["strengthened"] for r in credited),
        rewarded_backtracking_episodes=[r for r in credited if len(r["passages"])>1],
        correct_arrivals_after_wrong_first_passage=[r["episode"] for r in training
            if r["correct_arrival"] and r["chosen_route"]!=r["requested_route"]],
        offline_switch_state_coverage=branch_coverage,
        coverage_scope="Position and heading thresholds are offline descriptive diagnostics, not model inputs or a proof of exact visual equivalence.",
        checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(rp.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results"/"memory_audit.json")
    args=parser.parse_args()
    result=dict(schema_version=1,scope="No full retraining; existing registered archives are read-only.",
        units=unit_checks(),gray=archive_audit("route_switch_evaluation"),
        texture=archive_audit("route_texture_evaluation"),
        priorities=[
            dict(priority=1,kind="实现偏差与容量缺陷",action="分离事实可回忆强度与任务价值；提供不被当前规则硬屏蔽的感觉事实通路；容量扩展或显式归档，回收须修复边界引用且不能停止新增事实。"),
            dict(priority=2,kind="建模缺口",action="把原PFC动态联想/抑制接到事实回忆和下一感官预测，利用真实前后状态识别阶段不一致；奖励只调价值与经过验证的可塑资格，不再使整段游走等同可执行路线。"),
            dict(priority=3,kind="建模缺口，较低优先级",action="在原PFC/运动竞争中抑制互斥行动，保留多记忆供比较，但只让相容运动协同进入肌肉合成；用左右互斥记忆夹具及实际身体动作检验。")],
        limitations=["A gray-wall failure has a direct recorded wrong-phase memory chain; texture coverage counts alone do not prove its exact failed event chain.",
            "No dangling links or capacity exhaustion were found in either completed registered archive; capacity failures are reproduced in disposable small fixtures.",
            "The 80% visual-support gate and sequence gain are parameters, but increasing width or retuning those values alone does not restore missing factual access or phase reasoning."])
    result["source_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),NAV/"associative_controller.py",NAV/"route_switch_controller.py")}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),gray=result["gray"]["inventory"],texture=result["texture"]["inventory"]),ensure_ascii=False))


if __name__=="__main__":
    main()
