"""仅验证出生固定组合检测 + 原类运动反向赫布能忠实解码 4 肌肉；不改核心模块。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
np.random.seed(2026)
import 运动输出区_motor_output as M
muscles, levels = 4, 10
pool=M.特征神经元池(muscles*levels)
net=M.神经网络(pool,层数=2,连接半径=0,权重范围=0.1,阈值初值=1.1)
# [左推进,右推进,左后退,右后退]；每个组合完全由固定权重的共同输入检测。
forces=np.array([[0,0,0,0],[1,1,0,0],[0,0,1,1],[0,1,1,0],[1,0,0,1]],dtype=float)
def encode(values):
    active=np.arange(levels)[None,:] < np.rint(values[:,None]*levels)
    result=np.zeros((muscles,levels,2));result[:,:,0]=active;result[:,:,1]=~active
    return result.ravel()
net.来源[0]=np.tile(np.arange(pool.总数),(pool.总数,1))
net.权重[0]=np.zeros((pool.总数,pool.总数))
for i,force in enumerate(forces):
    net.权重[0][i]=encode(force)/pool.对数
    net.阈值[0][i]=0.999
reverse=M.反向赫布网络(M.特征神经元池(muscles*levels),本底范围=0)
checks=[]
for force in forces:
    command=net.前向传播(encode(force)).astype(bool)
    reverse.学习一帧(pool.激活.copy(),command)
for force in forces:
    command=net.前向传播(encode(force)).astype(bool)
    recovered=reverse.前向(command)[0::2].reshape(muscles,levels).sum(1)/levels
    assert np.array_equal(force,recovered)
    checks.append({'force':force.tolist(),'command':np.flatnonzero(command).tolist(),'decoded':recovered.tolist()})
print(json.dumps({'checks':checks,'learned_reverse_connections':reverse.条数()},indent=2))
