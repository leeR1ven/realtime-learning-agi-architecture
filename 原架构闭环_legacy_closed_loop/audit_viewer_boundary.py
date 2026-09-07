"""Independent, disposable viewer-state audit; no GUI process or model training."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_brain_boundary import equal_tree
from brain import OriginalBrain, packet_digest
from experience_environment import ExperienceEnvironment

HERE = Path(__file__).resolve().parent


class FrozenBrain:
    """Audit-only wrapper: run real decisions but never learn new connections."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def observe_then_act(self, packet, **kwargs):
        kwargs["learning"] = False
        return self.original.observe_then_act(packet, **kwargs)


def run():
    spec = importlib.util.spec_from_file_location("original_viewer_boundary_target", HERE / "viewer.py")
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    env = ExperienceEnvironment(seed=982, texture_seed=44017)
    session = viewer.LifeSession(FrozenBrain(OriginalBrain()), env)
    session.pair("red", 220.)
    for _ in range(3):
        session.tick()
        assert np.array_equal(session.executed, env.world.last_action)
        assert np.array_equal(session.executed, session.decision["muscles"])
        assert np.array_equal(session.executed, session.info["actual_motor_in_memory"])
    # Save immediately after external stimulus changes. The brain still has an
    # old prepared action; the restored new packet must remain just as unobserved.
    session.hide()
    session.tone(777., .8)
    assert session.brain.prepared["packet_digest"] != packet_digest(session.packet)
    with tempfile.TemporaryDirectory(prefix="viewer-boundary-") as directory:
        path = Path(directory) / "disposable_life.npz"
        session.save(path)
        restored = viewer.LifeSession.load(path)
        restored.brain = FrozenBrain(restored.brain)
        equal_tree(session.brain.snapshot(), restored.brain.snapshot(), "brain_after_load")
        equal_tree(viewer.environment_snapshot(session.environment),
                   viewer.environment_snapshot(restored.environment), "world_after_load")
        assert packet_digest(session.packet) == packet_digest(restored.packet)
        for step in range(3):
            session.tick()
            restored.tick()
            equal_tree(session.brain.snapshot(), restored.brain.snapshot(), "brain_continuation")
            equal_tree(viewer.environment_snapshot(session.environment),
                       viewer.environment_snapshot(restored.environment), "world_continuation")
            a, b = session.snapshot(), restored.snapshot()
            # Origin is intentionally replaced by an explanatory load-status UI
            # string; it never enters brain input or any physical computation.
            a.pop("origin")
            b.pop("origin")
            equal_tree(a, b, "life_continuation")
    assert session.control_steps == 6
    assert session.brain.adapter.motor_learning_steps == 0
    # --warmup2 and its helper are checked without running 48 learning frames.
    tree = ast.parse((HERE / "viewer.py").read_text(encoding="utf-8"))
    declarations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and any(isinstance(arg, ast.Constant) and arg.value == "--warmup2" for arg in node.args)]
    assert len(declarations) == 1
    assert next(k.value.value for k in declarations[0].keywords if k.arg == "const") == 48

    class Counter:
        def __init__(self):
            self.pairs = []
            self.ticks = 0

        def pair(self, *args):
            self.pairs.append((self.ticks, args))

        def tick(self, next_presentation=None):
            self.ticks += 1
            if next_presentation is not None:
                self.pair(*next_presentation)

    count = Counter()
    viewer.warmup(count, 48)
    assert count.ticks == 48
    assert [frame for frame, _ in count.pairs] == [0, 8, 16, 24, 32, 40]
    assert [args[0] for _, args in count.pairs] == ["red", "green", "blue", "green", "blue", "red"]
    return {"passed": True, "actual_initial_frames": 3, "actual_continuation_frames_per_branch": 3,
        "model_learning": False, "gui_launched": False, "formal_or_living_archives_written": False,
        "complete_brain_world_and_session_restore_exact": True,
        "new_unobserved_pcm_with_old_prepared_action_restored_exact": True,
        "texture_rng_and_all_world_fields_restored_exact": True,
        "display_executed_equals_actual_physical_action": True,
        "warmup2_declares_48_actions": True, "warmup_control_flow_ticks": count.ticks,
        "warmup_presentations": count.pairs,
        "warmup_check_scope": "Independent helper/control-flow count. This audit did not run 48 learning frames; viewer author's separate smoke test compares them against run_closed_loop.train.",
        "static_ui_review": "Muscle bars use session.executed and physical activations; executed source uses the preceding decision. Current neural counts use the observed outcome. Offline presentation/color catalog is not decoded as thought content or passed to the brain.",
        "viewer_source_sha256": hashlib.sha256((HERE / "viewer.py").read_bytes()).hexdigest()}


if __name__ == "__main__":
    report = run()
    path = HERE / "results" / "viewer_boundary_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
