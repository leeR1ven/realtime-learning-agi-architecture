"""Independent invariant checks for the new planar body; no policy training."""
from pathlib import Path
import sys
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "闭环仿真_closed_loop_sim"))
from world import World, WorldConfig


def check_world():
    results = {}
    c = WorldConfig(width=100, height=100, linear_drag=0, angular_drag=0,
                    muscle_time_constant=0)
    w = World(c)
    w.reset(position=(20, 20))
    o = w.step([1, 0, 1, 0], 0.2)
    # F=4 N, m=1 kg, t=.2 s; 40 semi-implicit substeps.
    expected_x = 20 + 4 * c.physics_dt**2 * 40 * 41 / 2
    assert np.allclose(o["velocity"], [0.8, 0], atol=1e-12)
    assert abs(o["position"][0] - expected_x) < 1e-12
    assert abs(o["angular_velocity"]) < 1e-12
    results["force_acceleration"] = {"vx": o["velocity"][0], "x": o["position"][0]}

    w.reset(position=(20, 20))
    o = w.step([0, 1, 1, 0], 0.1)
    expected_omega = 2 * c.max_muscle_force * c.lever_arm / w.inertia * .1
    assert abs(o["angular_velocity"] - expected_omega) < 1e-12
    assert np.linalg.norm(o["position"] - [20, 20]) < 1e-12
    results["pure_force_couple"] = {"omega": o["angular_velocity"], "expected": expected_omega}

    w.reset(position=(20, 20))
    o = w.step([1, 0, 0, 1], 0.1)
    assert abs(o["angular_velocity"] + expected_omega) < 1e-12
    results["antagonist_turn_symmetry"] = True

    w.reset(position=(20, 20))
    o = w.step([1, 1, 1, 1], .2)
    assert np.linalg.norm(o["velocity"]) == 0 and o["angular_velocity"] == 0
    results["coactivation_cancels"] = True

    w = World(WorldConfig(width=100, height=100))
    w.reset(position=(20, 20), velocity=(2, 1), angular_velocity=3)
    def energy():
        return .5 * w.config.mass * (w.velocity @ w.velocity) + .5 * w.inertia * w.angular_velocity**2
    e0 = energy()
    es = [e0]
    for _ in range(50):
        w.step([0, 0, 0, 0], .02)
        es.append(energy())
    assert np.all(np.diff(es) < 0)
    results["passive_drag_dissipates"] = {"before": e0, "after": es[-1]}

    w = World()
    w.reset(position=(2, 3), walls=[(3, 1, 3.01, 5)], velocity=(100, 0))
    o = w.step([0, 0, 0, 0], .02)
    assert o["position"][0] <= 3 - w.config.radius + 1e-8
    assert o["velocity"][0] <= 1e-8 and o["pain"] > 0 and o["contact"]
    assert o["touch"][0] > .9
    results["thin_wall_no_tunneling"] = {"x": o["position"][0], "vx": o["velocity"][0],
                                         "pain": o["pain"]}

    # From rest, force into wall yields an opposing physical impulse.
    w.reset(position=(3-w.config.radius-.001, 3), walls=[(3, 1, 3.01, 5)])
    for _ in range(10):
        o = w.step([1, 0, 1, 0])
    assert o["contact"] and o["contact_impulse"] > 0
    assert w.position[0] <= 3 - w.config.radius + 1e-8
    results["sustained_wall_contact"] = {"impulse": o["contact_impulse"], "pain": o["pain"]}

    w = World(WorldConfig(friction=.4, restitution=0))
    w.reset(position=(.181, 2), velocity=(-2, 1), angular_velocity=4)
    before = .5 * (w.velocity @ w.velocity) + .5 * w.inertia * w.angular_velocity**2
    o = w.step([0, 0, 0, 0], .02)
    after = .5 * (w.velocity @ w.velocity) + .5 * w.inertia * w.angular_velocity**2
    assert after < before and w.position[0] >= w.config.radius - 1e-8
    results["contact_energy_does_not_increase"] = {"before": before, "after": after}

    w = World()
    w.reset(position=(1, 3), objects=[{"position": (4, 3), "radius": .3, "color": (1, 0, 0)}])
    o = w.observe()
    assert abs(o["ray_distances"][2] - 2.7) < 1e-12
    assert np.array_equal(o["ray_colors"][2], [1, 0, 0])
    w.reset(position=(1, 3), walls=[(2, 2, 2.2, 4)])
    o = w.observe()
    assert abs(o["ray_distances"][2] - 1) < 1e-12
    assert np.array_equal(o["ray_colors"][2], w.config.wall_color)
    results["actual_ray_geometry_and_occlusion"] = True

    for bad in ([np.nan, 0, 0, 0], [2, 0, 0, 0], [-1, 0, 0, 0], [0, 0]):
        try:
            w.step(bad)
            raise AssertionError("invalid action accepted")
        except ValueError:
            pass
    try:
        w.reset(position=(2.1, 3))
        raise AssertionError("overlapping reset accepted")
    except ValueError:
        pass
    results["invalid_action_and_initial_overlap_rejected"] = True

    # Deterministic replay covers many collision and thrust combinations.
    rng = np.random.default_rng(417)
    actions = rng.uniform(size=(500, 4))
    traces = []
    for _ in range(2):
        w = World()
        w.reset(position=(1, 1), heading=.4, walls=[(3, 1, 3.2, 4)])
        for action in actions:
            o = w.step(action)
            assert np.isfinite(np.r_[o["position"], o["velocity"], o["angular_velocity"]]).all()
        traces.append(w.get_state())
    assert traces[0] == traces[1]
    results["ten_second_deterministic_finite_replay"] = True
    return results


if __name__ == "__main__":
    result = check_world()
    path = Path(__file__).with_name("world_checks.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PASS: {len(result)} independent check groups")
