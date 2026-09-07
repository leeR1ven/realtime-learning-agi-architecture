"""Read-only numerical audit of the old three-link simulator."""
from pathlib import Path
import sys
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "虚拟身体_virtual_body"))
import 身体 as B
import 验证 as V

G = np.array([[1., -1, 0], [0, 1, -1], [0, 0, 1]])
q = np.array([.1, -.2, .3])
M = B.质量矩阵(q)
K = B.自由动能矩阵(q)[2:, 2:]
results = {"M_matches_absolute_angle_kinetic_metric_error": float(np.max(np.abs(M-K))),
           "M_symmetry_error": float(np.max(np.abs(M-M.T))),
           "GM_asymmetry": float(np.max(np.abs(G@M-(G@M).T)))}

# Independent virtual-work derivative of torso centre position.
force = 1.0
q = np.zeros(3)
expected = np.zeros(3)
eps = 1e-6
for i in range(3):
    plus, minus = q.copy(), q.copy()
    plus[i] += eps
    minus[i] -= eps
    expected[i] = force*(B.关节位置质心(plus)[1][2,0]-B.关节位置质心(minus)[1][2,0])/(2*eps)
body = B.建身体(角度=q)
B.加推力(body, 1, 持续秒=1)
M = B.质量矩阵(q)
B.物理一步(body, np.zeros(3), dt=eps)
actual = M @ (body["角速"]/eps)
results["unit_force_virtual_work"] = {"expected_generalized_force": expected.tolist(),
                                      "actual_generalized_force": actual.tolist()}

def angular_momentum_about_com(body):
    q, qd = body["角度"], body["角速"]
    base = body["基位"] if body["模式"] == "空" else np.array([body["基x"], B.踝高])
    _, centres = B.关节位置质心(q, *base)
    vjoint = body["基速"].copy() if body["模式"] == "空" else np.zeros(2)
    velocities = []
    for i, link in enumerate(B.节表):
        velocities.append(vjoint + B.叉(qd[i], link[3]*B.方向(q[i])))
        vjoint = vjoint + B.叉(qd[i], link[1]*B.方向(q[i]))
    masses = np.array([s[2] for s in B.节表])
    com = (masses[:,None]*centres).sum(axis=0)/masses.sum()
    vcom = (masses[:,None]*velocities).sum(axis=0)/masses.sum()
    return sum(B.叉积y(centres[i]-com, masses[i]*(velocities[i]-vcom))
               + B.节表[i][4]*qd[i] for i in range(3))

body = B.建身体(角度=[.1,-.2,.3])
body["模式"] = "空"
body["基位"] = np.array([0., 3.])
before = angular_momentum_about_com(body)
B.悬空一步(body, [0,10,0], dt=eps)
after = angular_momentum_about_com(body)
results["internal_knee_torque_in_free_flight"] = {"applied_internal_torque": 10.,
                                                "Hdot_expected": 0.,
                                                "Hdot_actual": float((after-before)/eps)}

body = B.建身体()
mass = sum(x[2] for x in B.节表)
B.起跳(body, 1.8)
st = B.读状态(body)
results["jump_injected_from_rest"] = {"com_velocity": st["质心速度"].tolist(),
                                    "linear_momentum_added": (mass*st["质心速度"]).tolist(),
                                    "kinetic_energy_added_joules": float(.5*mass*(st["质心速度"]@st["质心速度"]))}

body = B.建身体()
B.迈步(body, .5)
results["legacy_step_support_mismatch"] = {"reported_foot_x": body["脚x"],
                                          "actual_chain_base_x": B.读状态(body)["关节点"][0,0]}

# Short baselines only, deliberately no gain scan or long training.
baselines = []
start = time.perf_counter()
for label, impulse, allow_jump in (("quiet",0,False),("push_8",8,False),("push_32_jump",32,True)):
    state, _, fallen, reason, lean, intents, jumps, displacement = V.跑回合(
        [0,0,0], 冲量=impulse, 最长秒=1.0, 允许跳=allow_jump)
    baselines.append({"scenario":label,"sim_seconds":state["时间"],"fallen":fallen,
                      "reason":reason,"max_lean":lean,"step_intents":intents,
                      "jumps":jumps,"displacement":displacement})
results["one_second_old_baselines"] = baselines
results["baseline_wall_seconds"] = time.perf_counter()-start

Path(__file__).with_name("legacy_checks.json").write_text(
    json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(results,ensure_ascii=False,indent=2))
