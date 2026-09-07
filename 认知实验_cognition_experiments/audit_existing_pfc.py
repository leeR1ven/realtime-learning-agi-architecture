"""Read-only causal audit of the existing embodied prototype, not a new brain.

Perturb ONLY its persistent PFC activity in isolated checkpoint copies. Hold
observations, weights, sensory traces, motor state and memory constant. Report
whether internal activity can change the actual motor decision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "闭环仿真_closed_loop_sim"))
from brain import Brain, tone
from world import World


def audit():
    checkpoint = ROOT / "闭环仿真_closed_loop_sim" / "results" / "demo_brain.npz"
    base = Brain.load(checkpoint)
    base.reset_activity()
    observation = World().observe()
    rng = np.random.default_rng(260906)
    patterns = {
        "empty": np.zeros(base.n, dtype=bool),
        "all_active": np.ones(base.n, dtype=bool),
    }
    for i in range(8):
        pattern = np.zeros(base.n, dtype=bool)
        pattern[rng.choice(base.n, 48, replace=False)] = True
        patterns[f"random_{i}"] = pattern
    rows = []
    for cue in range(-1, 4):
        wave = () if cue < 0 else tone(cue)
        predictions = []
        for name, pattern in patterns.items():
            brain = copy.deepcopy(base)
            brain.previous = pattern.copy()
            brain.pfc.念头 = pattern.copy()
            muscles, _, info = brain.step(observation, wave, plasticity=False)
            predictions.append({
                "state": name,
                "action": info["action"],
                "muscles": muscles.tolist(),
                "scores": info["scores"],
                "next_thought_active": info["thought_active"],
            })
        reference = predictions[0]
        rows.append({
            "cue": cue,
            "all_scores_identical": all(
                np.array_equal(p["scores"], reference["scores"])
                for p in predictions[1:]),
            "all_muscles_identical": all(
                p["muscles"] == reference["muscles"] for p in predictions[1:]),
            "predictions": predictions,
        })
    # The old flag is not a selective lesion: the PFC still advances, while
    # direct audio weights are disabled alongside PFC association readout.
    disabled = copy.deepcopy(base)
    disabled.previous[:] = False
    disabled.step(observation, tone(0), pfc_enabled=False, plasticity=False)
    result = {
        "scope": "Existing prototype persistent PFC state, single-step causal intervention",
        "brain_sha256": hashlib.sha256((ROOT / "闭环仿真_closed_loop_sim" / "brain.py").read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rows": rows,
        "pfc_disabled_still_advances_state": bool(disabled.previous.any()),
        "persistent_state_perturbation_changes_motor_scores": any(
            not row["all_scores_identical"] for row in rows),
        "interpretation": (
            "This tests the prototype's persistent activity, not the causal role "
            "of all learned PFC weights. Cue-to-motor PFC associations remain "
            "connected. It is not a test of the user's full proposed architecture."
        ),
    }
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    path = out / "existing_pfc_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path),
                      "interventions": len(patterns) * len(rows),
                      "state_changes_scores": result["persistent_state_perturbation_changes_motor_scores"],
                      "disabled_still_advances": result["pfc_disabled_still_advances_state"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    audit()
