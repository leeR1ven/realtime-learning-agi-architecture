"""New candidate PFC working-memory probe, NOT the original PFC or AGI.

The brain below receives only an eight-channel sensory vector. Its fixed sparse
random input/recurrent synapses update one leaky tanh membrane vector. This is a
signed rate approximation, not a claim about biological firing or the original
neuron model. A two-output readout learns online only after a decision. There is
no backpropagation through time, symbolic task solver, task-specific combination
feature, or past-input buffer in CandidatePFC.

The environment owns context/cue identities, the XOR scoring rule, trial reset,
and teacher feedback. Shuffled-feedback and memory-lesion conditions are applied
by the experiment driver, not selected by the model. Evaluation supplies no
teacher signal; the same neural forward/update API remains in use.

Run with the existing environment:
  ..\\.venv\\Scripts\\python.exe -X utf8 pfc_probe.py --quick
Only this experiment's JSON and NPZ files are written.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


VERSION = "candidate-pfc-rate-reservoir-v1"
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    width: int = 256
    inputs: int = 8
    input_fanin: int = 3
    recurrent_fanin: int = 8
    input_gain: float = 1.5
    recurrent_gain: float = 1.1
    leak: float = 0.4
    bias_gain: float = 0.1
    learning_rate: float = 0.6
    seed: int = 3
    readout_rule: str = "delta"


class CandidatePFC:
    """Task-agnostic fixed reservoir plus an online local plastic readout."""

    def __init__(self, config: Config):
        self.config = config
        if config.width < 1 or config.inputs < 1 or config.readout_rule not in ("delta", "hebb"):
            raise ValueError("invalid width, input count or readout rule")
        rng = np.random.default_rng(config.seed)
        self.fixed_input_source = rng.integers(config.inputs, size=(config.width, config.input_fanin))
        self.fixed_input_weight = rng.normal(size=self.fixed_input_source.shape) * config.input_gain / np.sqrt(config.input_fanin)
        self.fixed_recurrent_source = rng.integers(config.width, size=(config.width, config.recurrent_fanin))
        self.fixed_recurrent_weight = rng.normal(size=self.fixed_recurrent_source.shape) * config.recurrent_gain / np.sqrt(config.recurrent_fanin)
        self.fixed_bias = rng.uniform(-config.bias_gain, config.bias_gain, config.width)
        self.plastic_readout = np.zeros((2, config.width + 1), dtype=np.float64)
        self.state_membrane = np.zeros(config.width, dtype=np.float64)
        self.steps = 0

    def reset_state(self):
        self.state_membrane[:] = 0

    def _features(self):
        # One constant bias plus CURRENT membrane activity; no historical input.
        return np.append(self.state_membrane, 1.0)

    def scores(self):
        weights = self.plastic_readout
        if self.config.readout_rule == "hebb":
            # Generic row norm controls accumulated synaptic magnitude. No label
            # counts, class means, context identities or XOR feature are used.
            weights = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
        return weights @ self._features()

    def step(self, sensory):
        sensory = np.asarray(sensory, dtype=np.float64)
        if sensory.shape != (self.config.inputs,) or not np.isfinite(sensory).all():
            raise ValueError("expected a finite sensory vector matching the fixed interface")
        current = (sensory[self.fixed_input_source] * self.fixed_input_weight).sum(axis=1)
        current += (self.state_membrane[self.fixed_recurrent_source] * self.fixed_recurrent_weight).sum(axis=1)
        current += self.fixed_bias
        self.state_membrane *= 1.0 - self.config.leak
        self.state_membrane += self.config.leak * np.tanh(current)
        self.steps += 1
        return self.scores()

    def feedback(self, teacher_output):
        """External teacher arrives AFTER action selection, never as an input."""
        if teacher_output not in (0, 1):
            raise ValueError("teacher must activate one of two output neurons")
        features = self._features()
        if self.config.readout_rule == "delta":
            target = np.zeros(2)
            target[teacher_output] = 1.0
            error = target - self.plastic_readout @ features
            # Normalized local delta: presynaptic activity x postsynaptic error.
            self.plastic_readout += self.config.learning_rate * error[:, None] * features / (1e-9 + features @ features)
        else:
            # Pure teacher correlation; no prediction or error enters the rule.
            self.plastic_readout[teacher_output] += self.config.learning_rate * features / max(np.linalg.norm(features), 1e-9)

    def arrays(self):
        return {name: value for name, value in vars(self).items() if isinstance(value, np.ndarray)}

    def birth_hash(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.arrays().items()):
            if name.startswith("fixed_"):
                digest.update(name.encode())
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": VERSION, "config": asdict(self.config), "steps": self.steps,
                "birth_hash": self.birth_hash(), "readout_bias": True}
        encoded = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
        arrays = self.arrays()
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("nonfinite checkpoint state")
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, **arrays, metadata_json=encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata_json"].tobytes().decode())
            if meta["version"] != VERSION:
                raise ValueError("checkpoint version mismatch")
            model = cls(Config(**meta["config"]))
            expected = model.arrays()
            if set(data.files) != set(expected) | {"metadata_json"}:
                raise ValueError("checkpoint keys differ")
            for name, template in expected.items():
                value = data[name]
                if value.shape != template.shape or value.dtype != template.dtype or not np.isfinite(value).all():
                    raise ValueError(f"invalid checkpoint array: {name}")
                setattr(model, name, value.copy())
        if model.birth_hash() != meta["birth_hash"]:
            raise ValueError("fixed birth synapse hash mismatch")
        model.steps = int(meta["steps"])
        return model


@dataclass(frozen=True)
class Trial:
    context: int
    cue: int
    distractors: tuple[int, ...]  # -1 is blank; 4..7 are unrelated sensory tokens.


class Environment:
    """Owns the scoring rule; the model never receives context/cue metadata."""

    def __init__(self, seed):
        rng = np.random.default_rng(70_000 + seed)
        self.permutation = rng.permutation(8)
        self.signs = rng.choice([-1.0, 1.0], 8)
        # Output identity is randomized independently of reservoir and encoding.
        self.label_flip = int(np.random.default_rng(90_000 + seed).integers(2))

    def sensory(self, token, amplitude=1.0):
        vector = np.zeros(8)
        if token >= 0:
            vector[self.permutation[token]] = amplitude * self.signs[token]
        return vector

    def target(self, trial, task="delayed_xor"):
        value = trial.context ^ trial.cue if task == "delayed_xor" else trial.cue
        return value ^ self.label_flip

    def present(self, model, trial, *, clear_each_frame=False, clear_before_cue=False):
        # Identical externally marked episode boundaries for EVERY condition.
        model.reset_state()

        def frame(token, amplitude=1.0):
            if clear_each_frame:
                model.reset_state()
            return model.step(self.sensory(token, amplitude))

        frame(trial.context)
        frame(trial.context)
        for token in trial.distractors:
            frame(token, .35)
        if clear_before_cue:
            model.reset_state()
        frame(2 + trial.cue)
        scores = frame(2 + trial.cue)
        return int(np.argmax(scores)), scores.copy()


def make_trials(count, seed, *, long_delay=False):
    if count % 4:
        raise ValueError("trial count must be divisible by four for exact balance")
    rng = np.random.default_rng(seed)
    combinations = np.tile(np.arange(4), count // 4)
    rng.shuffle(combinations)
    trials = []
    for combination in combinations:
        delay = int(rng.integers(8, 13) if long_delay else rng.integers(2, 6))
        distractions = tuple(int(rng.integers(4, 8)) if rng.random() < .5 else -1 for _ in range(delay))
        trials.append(Trial(int(combination // 2), int(combination % 2), distractions))
    return trials


def make_strict_paired_trials(count, seed, forbidden):
    """Novel full sequences; matched pairs differ ONLY in earlier context.

    Rejection changes the short-sequence frequency relative to the IID test, so
    the resulting delay distribution is reported separately. This is a split
    audit, not an unreported replacement for the first IID evaluation.
    """
    if count % 4:
        raise ValueError("paired count must be divisible by four")
    forbidden = set(forbidden)
    rng = np.random.default_rng(seed)
    used, result = set(), []
    attempts = 0
    for pair_index in range(count // 2):
        cue = pair_index % 2
        while True:
            attempts += 1
            if attempts > 1_000_000:
                raise RuntimeError("strict paired test space exhausted")
            delay = int(rng.integers(2, 6))
            distractions = tuple(int(rng.integers(4, 8)) if rng.random() < .5 else -1 for _ in range(delay))
            pair = (Trial(0, cue, distractions), Trial(1, cue, distractions))
            key = (cue, distractions)
            if key not in used and not any(trial in forbidden for trial in pair):
                used.add(key)
                result.extend(pair)
                break
    rng.shuffle(result)
    return result


def split_audit(training, testing, paired):
    train_set = set(training)
    return {"iid_full_sequence_overlap": sum(trial in train_set for trial in testing),
            "iid_test_count": len(testing),
            "strict_full_sequence_overlap": sum(trial in train_set for trial in paired),
            "strict_test_count": len(paired), "strict_context_pairs": len(paired) // 2,
            "strict_delay_counts": {str(delay): sum(len(t.distractors) == delay for t in paired) for delay in range(2, 6)}}


def evaluate(model, environment, trials, *, task="delayed_xor", clear_each_frame=False, clear_before_cue=False):
    correct = 0
    confusion = np.zeros((2, 2), dtype=int)
    per_combination = {f"{context}{cue}": [0, 0] for context in range(2) for cue in range(2)}
    margins = []
    pairs = {}
    started = time.perf_counter()
    for trial in trials:
        predicted, scores = environment.present(model, trial, clear_each_frame=clear_each_frame,
                                               clear_before_cue=clear_before_cue)
        expected = environment.target(trial, task)
        correct += predicted == expected
        confusion[expected, predicted] += 1
        group = per_combination[f"{trial.context}{trial.cue}"]
        group[0] += int(predicted == expected)
        group[1] += 1
        margins.append(float(scores[expected] - scores[1 - expected]))
        pairs.setdefault((trial.cue, trial.distractors), {})[trial.context] = predicted == expected
    complete_pairs = [value for value in pairs.values() if set(value) == {0, 1}]
    return {"accuracy": correct / len(trials), "correct": int(correct), "trials": len(trials),
            "confusion_expected_by_predicted": confusion.tolist(),
            "per_context_cue_correct_total": per_combination,
            "mean_true_margin": float(np.mean(margins)),
            "matched_context_pairs": len(complete_pairs),
            "both_contexts_correct_fraction": (float(np.mean([all(value.values()) for value in complete_pairs]))
                                                if complete_pairs else None),
            "wall_seconds": time.perf_counter() - started,
            "feedback_supplied": False}


def train(config, environment, trials, *, task="delayed_xor", clear_each_frame=False, shuffled_feedback=False):
    model = CandidatePFC(config)
    teachers = np.array([environment.target(trial, task) for trial in trials])
    if shuffled_feedback:
        np.random.default_rng(120_000 + config.seed).shuffle(teachers)
    correct = []
    started = time.perf_counter()
    for trial, teacher in zip(trials, teachers):
        decision, _ = environment.present(model, trial, clear_each_frame=clear_each_frame)
        correct.append(decision == environment.target(trial, task))
        model.feedback(int(teacher))  # AFTER the model's decision.
    return model, {"trials": len(trials), "wall_seconds": time.perf_counter() - started,
                   "online_accuracy": float(np.mean(correct)),
                   "last_200_online_accuracy": float(np.mean(correct[-200:])),
                   "shuffled_feedback": shuffled_feedback}


def verify_resume(model, path, seed):
    model.save(path)
    restored = CandidatePFC.load(path)
    rng = np.random.default_rng(seed)
    equality = True
    before_fixed = model.birth_hash()
    for _ in range(12):
        sensory = rng.uniform(-1, 1, 8)
        first, second = model.step(sensory), restored.step(sensory)
        equality &= bool(np.array_equal(first, second))
        teacher = int(rng.integers(2))
        model.feedback(teacher)
        restored.feedback(teacher)
    equality &= all(np.array_equal(value, restored.arrays()[key]) for key, value in model.arrays().items())
    equality &= model.birth_hash() == restored.birth_hash() == before_fixed
    return bool(equality)


def aggregate(rows):
    summary = []
    for condition, width, rule in sorted({(row["condition"], row["width"], row["rule"]) for row in rows}):
        subset = [row for row in rows if (row["condition"], row["width"], row["rule"]) == (condition, width, rule)]
        values = [row["evaluation"]["accuracy"] for row in subset]
        summary.append({"condition": condition, "width": width, "rule": rule,
                        "mean_accuracy": float(np.mean(values)), "std_across_seeds": float(np.std(values)),
                        "min_accuracy": float(min(values)), "max_accuracy": float(max(values)),
                        "seeds": [row["seed"] for row in subset], "accuracies": values})
    return summary


def augment_heldout(source, output=None):
    """Add an audited split using existing trained NPZs, with no retraining."""
    source = Path(source)
    report = json.loads(source.read_text(encoding="utf-8"))
    new_rows, audits = [], {}
    started = time.perf_counter()
    for item in report["checkpoint_checks"]:
        model = CandidatePFC.load(item["path"])
        seed = item["seed"]
        env = Environment(seed)
        training = make_trials(report["training_trials_per_independent_model"], 10_000 + seed)
        testing = make_trials(report["evaluation_trials"], 20_000 + seed)
        paired = make_strict_paired_trials(report["evaluation_trials"], 50_000 + seed, training)
        audits[str(seed)] = split_audit(training, testing, paired)
        for name, lesion in (("strict_novel_paired_context", False), ("strict_paired_context_cleared", True)):
            result = evaluate(model, env, paired, clear_before_cue=lesion)
            new_rows.append({"condition": name, "width": item["width"], "seed": seed,
                             "rule": model.config.readout_rule, "trained_as": item["condition"], "evaluation": result})
    report["rows"].extend(new_rows)
    report["summary"] = aggregate(report["rows"])
    report["split_audit"] = audits
    report["strict_heldout_augmentation"] = {"source": str(source.resolve()), "retrained": False,
                                           "wall_seconds": time.perf_counter() - started}
    report["limitations"].append("The original IID test repeats some complete training sequences. Strict paired results exclude all such sequences but rejection changes the delay distribution.")
    output = Path(output) if output else source.with_name(source.stem + "_strict.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "split_audit": audits,
                      "added_summary": aggregate(new_rows)}, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="three seeds, 1200 teaching trials")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--augment-heldout", type=Path, help="evaluate saved trained models on strict novel context pairs, without retraining")
    args = parser.parse_args()
    if args.augment_heldout:
        return augment_heldout(args.augment_heldout, args.output)
    seeds = [3, 11, 29] if args.quick else [3, 11, 29, 47, 83]
    training_count = 1200 if args.quick else 3600
    test_count = 240 if args.quick else 800
    output = args.output or HERE / "results" / ("pfc_probe_quick.json" if args.quick else "pfc_probe_full.json")
    checkpoint_directory = output.parent / (output.stem + "_checkpoints")
    rows, resumes, audits = [], [], {}
    started = time.perf_counter()
    for width in (64, 256, 1024):
        for seed in seeds:
            environment = Environment(seed)
            training = make_trials(training_count, 10_000 + seed)
            testing = make_trials(test_count, 20_000 + seed)
            paired = make_strict_paired_trials(test_count, 50_000 + seed, training)
            audits[str(seed)] = split_audit(training, testing, paired)
            extended = make_trials(test_count, 30_000 + seed, long_delay=True)
            config = Config(width=width, seed=seed)
            conditions = [
                ("delayed_xor", "delayed_xor", False, False, "delta"),
                ("no_memory_trained", "delayed_xor", True, False, "delta"),
                ("shuffled_feedback", "delayed_xor", False, True, "delta"),
                ("current_cue_control", "current_cue", False, False, "delta"),
            ]
            if width == 256:
                conditions.append(("delayed_xor_hebb", "delayed_xor", False, False, "hebb"))
            for condition, task, clear_each, shuffled, rule in conditions:
                config = Config(width=width, seed=seed, readout_rule=rule)
                model, train_stats = train(config, environment, training, task=task,
                                          clear_each_frame=clear_each, shuffled_feedback=shuffled)
                result = evaluate(model, environment, testing, task=task, clear_each_frame=clear_each)
                base = {"condition": condition, "width": width, "seed": seed, "rule": rule,
                        "config": asdict(config), "train": train_stats, "evaluation": result,
                        "birth_hash": model.birth_hash(), "trainable_parameter_count": int(model.plastic_readout.size),
                        "fixed_synapse_count": int(model.fixed_input_weight.size + model.fixed_recurrent_weight.size)}
                rows.append(base)
                if condition in ("delayed_xor", "delayed_xor_hebb"):
                    for label, lesion, trials in (("context_cleared_before_cue", True, testing),
                                                 ("long_delay_8_to_12", False, extended),
                                                 ("strict_novel_paired_context", False, paired),
                                                 ("strict_paired_context_cleared", True, paired)):
                        lesion_result = evaluate(model, environment, trials, clear_before_cue=lesion)
                        rows.append({"condition": label, "width": width, "seed": seed, "rule": rule,
                                     "evaluation": lesion_result, "trained_as": condition})
                    save_path = checkpoint_directory / f"{condition}_n{width}_seed{seed}.npz"
                    resumes.append({"path": str(save_path.resolve()), "condition": condition, "width": width,
                                    "seed": seed, "exact_continuation": verify_resume(model, save_path, 40_000 + seed)})
                print(f"n={width:4d} seed={seed:2d} {condition:25s} {rule:5s} accuracy={result['accuracy']:.3f}", flush=True)
    report = {
        "version": VERSION, "candidate_not_original_pfc": True, "quick": args.quick,
        "wall_seconds": time.perf_counter() - started, "widths": [64, 256, 1024], "seeds": seeds,
        "training_trials_per_independent_model": training_count, "evaluation_trials": test_count,
        "chance_accuracy": .5, "input_channels": 8,
        "protocol": {"context_frames": 2, "training_delay_frames": [2, 5], "cue_frames": 2,
                     "distractor_probability_per_delay_frame": .5, "distractor_amplitude": .35,
                     "long_delay_frames": [8, 12], "trial_boundary_resets_state": True,
                     "teaching_after_decision_only": True, "test_feedback": False,
                     "per_seed_random_signed_token_permutation": True, "per_seed_random_output_label_flip": True,
                     "task_scorer_in_environment_only": True, "parameters_selected_before_first_run": True},
        "rules": {"delta": "normalized presynaptic activity times postsynaptic teacher-minus-output error",
                  "hebb": "additive normalized presynaptic activity times teacher one-hot; L2-normalized output rows at decision"},
        "rows": rows, "summary": aggregate(rows), "checkpoint_checks": resumes,
        "split_audit": audits,
        "all_checkpoint_checks_passed": all(row["exact_continuation"] for row in resumes),
        "limitations": [
            "New fixed-random recurrent leaky-tanh candidate; not an implementation result of the original PFC.",
            "Delta uses an explicit output error signal, while the separately reported Hebb condition does not.",
            "Four abstract sensory combinations and externally marked episode boundaries; no natural language or physical embodiment.",
            "Short training with no hyperparameter search; failure does not prove all local-learning recurrent systems cannot solve the task.",
            "Only readout weights learn; fixed recurrent circuits do not acquire task-specific working memory.",
            "A successful delayed-XOR score is limited evidence of state-dependent nonlinear readout, not AGI or broad cognitive ability.",
            "Width changes random feature diversity and readout parameter count together; width alone is not an isolated causal explanation.",
            "Seed standard deviation is descriptive and not a confidence interval; examples within each seed share a trained model.",
            "The IID test can repeat complete training sequences. Strict paired tests exclude them, at the cost of changing the short-sequence distribution.",
            "State lesions do not isolate recurrent synapses from leaky membrane persistence; both are included in this candidate.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "wall_seconds": report["wall_seconds"],
                      "summary": report["summary"], "checkpoint_checks": report["all_checkpoint_checks_passed"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if report["all_checkpoint_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
