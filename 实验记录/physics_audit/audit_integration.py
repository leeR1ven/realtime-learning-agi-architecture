"""Independent causal and persistence probes; reports failures without training scans.

This audit deliberately changes private state in isolated Brain instances to
exercise otherwise rare checkpoint boundaries. It never edits brain.py.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "闭环仿真"))
from brain import Brain, tone, sound_features
from world import World
from checkpoint import merge_checkpoints, load_checkpoint

OUT = Path(__file__).resolve().parent


def step_info(brain, obs, waveform=(), **kw):
    muscles, emitted, info = brain.step(obs, waveform, **kw)
    return np.asarray(muscles), np.asarray(emitted), info


def table_difference(a, b):
    keys = set((i, j) for i, row in a.表.items() for j in row)
    keys.update((i, j) for i, row in b.表.items() for j in row)
    return max((abs(a.查(i).get(j, 0) - b.查(i).get(j, 0)) for i, j in keys), default=0.)


def run_audit():
    result = {"brain_sha256": hashlib.sha256((ROOT / "闭环仿真" / "brain.py").read_bytes()).hexdigest()}
    world = World()
    obs = world.reset(position=(1, 3), objects=[{"position": (4, 3), "color": (1, 0, 0)}])

    # Privileged labels/coordinates must have no effect when actual receptors agree.
    a, b = Brain(), Brain()
    altered = copy.deepcopy(obs)
    altered.update(target_action=4, target_name="右转", object_name="left", target_position=[-100, 99],
                   position=np.array([7, 5]), heading=-2.)
    am, _, ai = step_info(a, obs, plasticity=False)
    bm, _, bi = step_info(b, altered, plasticity=False)
    result["no_privileged_label_or_position_action_bypass"] = bool(np.array_equal(am, bm))

    # Disconnect the complete receptor encoder, while leaving the raw PCM nonzero.
    b = Brain()
    b.audio_weights[0, 3] = 1.
    b.encoder.encode = lambda values: np.zeros(b.n, dtype=bool)
    _, _, bi = step_info(b, obs, tone(0), plasticity=False)
    result["disconnected_encoder_audio_action"] = bi["action"]
    result["encoder_disconnection_blocks_raw_audio_drive"] = bool(bi["action"] != 3)

    # Stimulation is only the teacher. Test the pairing after removing every cue
    # except the PCM waveform; a shuffled pairing must learn the shuffled action.
    pairing = ((0, 4), (1, 2))
    brains = []
    for tone_id, action_id in pairing:
        learner = Brain()
        for _ in range(12):
            learner.step(obs, tone(tone_id), stimulation=np.eye(5)[action_id], pleasant=1,
                         mouth_stimulation=np.eye(4)[tone_id])
        learner.auditory_trace[:] = 0
        learner.pain_trace = 0
        learner.last_action = 0
        prediction = learner.step(obs, tone(tone_id), plasticity=False)
        result[f"PCM_only_after_pairing_{tone_id}"] = {"expected_action": action_id,
            "actual_action": prediction[2]["action"], "correct": prediction[2]["action"] == action_id}
        path = OUT / f"audit_parent_{tone_id}.npz"
        learner.save(path)
        brains.append((learner, path))
    fused_path = OUT / "audit_fused.npz"
    merge_checkpoints([x[1] for x in brains], fused_path)
    fused = Brain.load(fused_path)
    fused_answers = []
    for tone_id, action_id in pairing:
        candidate = copy.deepcopy(fused)
        _, _, info = candidate.step(obs, tone(tone_id), plasticity=False)
        fused_answers.append({"tone": tone_id, "expected": action_id, "actual": info["action"]})
    result["two_parent_disjoint_sound_association_merge"] = fused_answers

    # Memories and the old PFC must be causally connected before interpreting
    # their growing edge counts as the reason behavior succeeded.
    healthy = copy.deepcopy(brains[0][0])
    severed = copy.deepcopy(healthy)
    severed.pfc.微步 = lambda previous, external: np.zeros(severed.n, dtype=bool)
    severed.pfc.驱动 = lambda previous: np.zeros(severed.n)
    severed.memory.recall_signal = lambda pattern: np.zeros(severed.n)
    same = True
    for waveform in (tone(0), (), tone(1), (), tone(3)):
        h, _, hi = healthy.step(obs, waveform, plasticity=False)
        s, _, si = severed.step(obs, waveform, plasticity=False)
        same = same and np.array_equal(h, s) and np.array_equal(hi["scores"], si["scores"])
    result["legacy_pfc_and_memory_severed_scores_identical_on_probes"] = bool(same)

    # Remove the added direct audio matrix to test each inherited memory path's
    # ability to change an actual action, not merely a logged activity count.
    causal = {}
    for name, pfc_on, memory_on in (("pfc_only", True, False), ("episodic_only", False, True),
                                   ("both", True, True), ("both_disconnected", False, False)):
        probe = copy.deepcopy(brains[0][0])
        probe.audio_weights[:] = 0
        probe.reset_activity()
        _, _, info = probe.step(obs, tone(0), plasticity=False,
                               pfc_enabled=pfc_on, memory_enabled=memory_on)
        causal[name] = {"action": info["action"],
                       "temporal_motor": info.get("temporal_motor"),
                       "episodic_motor": info.get("episodic_motor")}
    result["inherited_memory_path_action_causality_without_audio_matrix"] = causal

    # Standard save/load continuation should preserve plasticity, not just the
    # action from a read-only snapshot.
    a, path = brains[0]
    a = copy.deepcopy(a)
    a.save(OUT / "continuation.npz")
    b = Brain.load(OUT / "continuation.npz")
    same_actions = True
    for t in range(8):
        waveform = tone(t % 4)
        am, ae, ai = a.step(obs, waveform, pleasant=.2)
        bm, be, bi = b.step(obs, waveform, pleasant=.2)
        same_actions = same_actions and np.array_equal(am, bm) and np.array_equal(ae, be)
    result["normal_checkpoint_continuation"] = {
        "same_actions_and_emissions": bool(same_actions),
        "pfc_max_difference": table_difference(a.pfc.联想, b.pfc.联想),
        "memory_max_difference": table_difference(a.memory.特征到时间, b.memory.特征到时间)}

    # Decay scheduling is live state. Fill a valid-width PFC table to the default
    # warning threshold; restoring must not make its next decay happen early.
    a = Brain()
    table = a.pfc.联想
    for source in range(152):
        table.表[source] = dict.fromkeys(range(a.n), .1)
    table.连接条数 = sum(map(len, table.表.values()))
    table.维护次数 = 200
    table.上次衰减维护 = 199
    a.save(OUT / "decay_boundary.npz")
    b = Brain.load(OUT / "decay_boundary.npz")
    counters = {"before": [table.维护次数, table.上次衰减维护],
                "restored": [b.pfc.联想.维护次数, b.pfc.联想.上次衰减维护]}
    # No preceding thought: PFC learning adds no edges on this frame, isolating
    # the maintenance step from associative learning.
    a.step(obs)
    b.step(obs)
    delta = table_difference(a.pfc.联想, b.pfc.联想)
    result["checkpoint_decay_boundary"] = {**counters, "same_weights_after_next_frame": delta == 0,
                                           "max_difference": delta}

    # These five birth-trained reverse templates are now explicitly fixed.
    # Corruption must be rejected instead of silently rebuilding the checkpoint.
    a = Brain()
    a.motor.reverse.矩阵[:] = False
    before = a.motor.decode(1).copy()
    a.save(OUT / "motor_plastic.npz")
    try:
        b = Brain.load(OUT / "motor_plastic.npz")
        result["checkpoint_fixed_motor_corruption_rejected"] = False
        result["checkpoint_motor_changed_after_load"] = b.motor.decode(1).tolist()
    except ValueError as exc:
        result["checkpoint_fixed_motor_corruption_rejected"] = "fixed_motor_reverse" in str(exc)

    result["pure_unknown_330Hz_feature_leak"] = sound_features(
        .7*np.sin(2*np.pi*330*np.arange(800)/8000)).tolist()
    return result


def long_continuation():
    import time
    start = time.perf_counter()
    a = Brain()
    w = World()
    obs = w.reset(position=(2, 3))
    def observation(t):
        o = copy.deepcopy(obs)
        o["ray_distances"] = .2 + np.mod(np.arange(5)*.31 + t*.017, 3.5)
        o["ray_colors"] = np.mod(np.arange(15).reshape(5, 3)*.23 + (t//17)*.1, 1.)
        o["pain"] = .6 if t % 137 == 0 else 0.
        return o
    for t in range(1560):
        a.step(observation(t), tone((t//13) % 4), pleasant=.05)
    path = OUT / "long_continuation.npz"
    a.save(path)
    b = Brain.load(path)
    matching = True
    first_difference = None
    for t in range(1560, 1840):
        o, waveform = observation(t), tone((t//13) % 4)
        am, ae, ai = a.step(o, waveform, pleasant=.05)
        bm, be, bi = b.step(o, waveform, pleasant=.05)
        this_match = np.array_equal(am, bm) and np.array_equal(ae, be)
        matching = matching and this_match
        if not this_match and first_difference is None:
            first_difference = t
    return {"frames_before_save":1560, "frames_after_save":280,
            "all_actions_and_audio_exact":bool(matching), "first_difference":first_difference,
            "pfc_max_difference":table_difference(a.pfc.联想,b.pfc.联想),
            "memory_in_max_difference":table_difference(a.memory.特征到时间,b.memory.特征到时间),
            "memory_out_max_difference":table_difference(a.memory.时间到特征,b.memory.时间到特征),
            "maintenance_original":[[t.维护次数,t.上次衰减维护] for t in
                (a.pfc.联想,a.pfc.去抑,a.memory.特征到时间,a.memory.时间到特征)],
            "maintenance_restored":[[t.维护次数,t.上次衰减维护] for t in
                (b.pfc.联想,b.pfc.去抑,b.memory.特征到时间,b.memory.时间到特征)],
            "wall_seconds":time.perf_counter()-start}


if __name__ == "__main__":
    result = run_audit()
    if "--long" in sys.argv:
        result["long_continuation_crossing_decay"] = long_continuation()
    (OUT / "integration_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
