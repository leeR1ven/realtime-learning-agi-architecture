"""Independent read-only audit of pfc_probe.py protocol and frozen checkpoints.

Writes only this audit's JSON. No hyperparameter selection or checkpoint update.
"""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import textwrap
import numpy as np
import pfc_probe as P

HERE = Path(__file__).resolve().parent


def strict_pairs(training, seed, count=240):
    assert count % 4 == 0
    known = set(training)
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(count // 2):
        cue = i % 2
        for _ in range(100000):
            delay = int(rng.integers(2, 6))
            ds = tuple(int(rng.integers(4, 8)) if rng.random() < .5 else -1 for _ in range(delay))
            pair = [P.Trial(0, cue, ds), P.Trial(1, cue, ds)]
            if all(t not in known for t in pair) and pair[0] not in trials:
                trials.extend(pair)
                break
        else:
            raise RuntimeError("Could not generate unseen paired trials")
    return trials


class Recorder(P.CandidatePFC):
    def __init__(self, config):
        super().__init__(config)
        self.inputs_seen = []
        self.events = []

    def step(self, sensory):
        old = self.plastic_readout.copy()
        answer = super().step(sensory)
        assert np.array_equal(old, self.plastic_readout)
        self.inputs_seen.append(np.asarray(sensory).copy())
        self.events.append("step")
        return answer

    def feedback(self, teacher):
        assert self.events and self.events[-1] == "step"
        self.events.append("feedback")
        return super().feedback(teacher)


def audit():
    out = {"pfc_probe_sha256": hashlib.sha256((HERE / "pfc_probe.py").read_bytes()).hexdigest()}
    env = P.Environment(3)
    model = Recorder(P.Config(width=64, seed=3))
    trials = P.make_trials(40, 71621)
    fixed = model.birth_hash()
    for trial in trials:
        event_start = len(model.events)
        decision, _ = env.present(model, trial)
        assert model.events[event_start:] == ["step"] * (len(trial.distractors) + 4)
        target = env.target(trial)
        model.feedback(target)
    assert model.birth_hash() == fixed
    out["sensory_updates_do_not_change_readout"] = True
    out["feedback_arrives_after_all_trial_frames"] = True
    out["teacher_updates_leave_fixed_network_unchanged"] = True

    # Inspect actual training call order, not only the manually driven loop.
    tree = ast.parse(textwrap.dedent(inspect.getsource(P.train)))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For))
    present_line = next(n.lineno for n in ast.walk(loop) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == "present")
    feedback_line = next(n.lineno for n in ast.walk(loop) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "feedback")
    assert present_line < feedback_line
    out["actual_train_calls_present_before_feedback"] = True

    before = model.plastic_readout.copy()
    events = len(model.events)
    P.evaluate(model, env, trials)
    P.evaluate(model, env, trials, clear_before_cue=True)
    P.evaluate(model, env, trials, clear_each_frame=True)
    assert np.array_equal(before, model.plastic_readout)
    assert "feedback" not in model.events[events:]
    out["normal_and_lesion_evaluation_supply_no_feedback_or_weight_updates"] = True

    # Same cue and full distraction history, opposite context and label.
    pair = [P.Trial(0, 1, (-1, 4, 6, -1)), P.Trial(1, 1, (-1, 4, 6, -1))]
    streams, cleared_scores = [], []
    for trial in pair:
        model.inputs_seen = []
        env.present(model, trial)
        streams.append(np.asarray(model.inputs_seen))
        _, score = env.present(model, trial, clear_before_cue=True)
        cleared_scores.append(score)
    assert not np.array_equal(streams[0][:2], streams[1][:2])
    assert np.array_equal(streams[0][2:], streams[1][2:])
    assert env.target(pair[0]) != env.target(pair[1])
    assert np.array_equal(cleared_scores[0], cleared_scores[1])
    out["counterfactual_pair_has_identical_full_suffix_and_opposite_answer"] = True
    out["cleared_state_pair_scores_are_identical"] = True

    overlaps = []
    for seed in (3, 11, 29):
        training = P.make_trials(1200, 10000 + seed)
        testing = P.make_trials(240, 20000 + seed)
        known = set(training)
        overlaps.append({"seed": seed, "train_unique": len(known),
                         "iid_test_exact_seen": sum(t in known for t in testing), "iid_test_count": len(testing)})
    out["original_iid_episode_overlap"] = overlaps

    # A cleared-state model has the same sensory channels and readout capacity;
    # show it can learn the immediate mapping before interpreting XOR failure.
    no_state, train_info = P.train(P.Config(width=64, seed=3), env,
                                   P.make_trials(1200, 10003), task="current_cue", clear_each_frame=True)
    out["cleared_state_current_cue_positive_control"] = P.evaluate(
        no_state, env, P.make_trials(240, 20003), task="current_cue", clear_each_frame=True)

    paired_results = []
    checkpoint_dir = HERE / "results" / "pfc_probe_quick_checkpoints"
    for seed in (3, 11, 29):
        training = P.make_trials(1200, 10000 + seed)
        paired = strict_pairs(training, 91800 + seed)
        assert not (set(paired) & set(training))
        assert len(set(paired)) == len(paired)
        environment = P.Environment(seed)
        for width in (64, 256, 1024):
            path = checkpoint_dir / f"delayed_xor_n{width}_seed{seed}.npz"
            if not path.exists():
                continue
            frozen = P.CandidatePFC.load(path)
            old_weights = frozen.plastic_readout.copy()
            full = P.evaluate(frozen, environment, paired)
            cleared = P.evaluate(frozen, environment, paired, clear_before_cue=True)
            assert cleared["accuracy"] == .5
            assert np.array_equal(old_weights, frozen.plastic_readout)
            paired_results.append({"seed":seed, "width":width, "trials":len(paired),
                                   "exact_train_overlap":0, "full_accuracy":full["accuracy"],
                                   "clear_before_cue_accuracy":cleared["accuracy"]})
    out["independent_strict_unseen_paired_checkpoint_results"] = paired_results
    return out


if __name__ == "__main__":
    result = audit()
    path = HERE / "results" / "independent_probe_audit.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
