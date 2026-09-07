"""任务.py —— 本能期·站立任务

模型：平面火柴人（骨盆 + 躯干 + 两条腿，髋/膝/踝各一对肌肉）。
感觉输入（17 维）：骨盆高度/俯仰、6 个关节角 + 全部速度。
输出（6 路）：左右髋/膝/踝的肌肉发力。

这里只定义"任务本身"（人是怎么算倒下的、奖励是什么、一回合怎么跑），
连接(权重)的训练在 训练.py —— 进化式搜索，不碰你的赫布架构。

本能学好后，冻结这组连接 = 天生会站稳；
以后日常赫布学习在这之上精调，不重新学站立。
"""
import os
import numpy as np
import mujoco

目录 = os.path.dirname(os.path.abspath(__file__))
模型路径 = os.path.join(目录, "人形.xml")

最大秒数 = 3.0        # 一回合最长站多久
dt = 0.02             # 物理步长（秒）＝ 高频本能层的工作周期
最大步数 = int(最大秒数 / dt)

隐层数 = 24           # 本能区隐藏神经元个数
输入数 = 17           # 感觉输入维数
输出数 = 6            # 肌肉路数：左髋 左膝 左踝 右髋 右膝 右踝
力矩上限 = np.array([150.0, 150.0, 70.0, 150.0, 150.0, 70.0])

# qpos 顺序：0 平移X  1 平移Z  2 俯仰  3 左髋 4 左膝 5 左踝 6 右髋 7 右膝 8 右踝
# 特征用 qpos[1:]（去掉绝对位置 X），中心取"站直"姿态附近
特征中心 = np.array([0.89, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0])


def 建模型():
    """每个进程各自建一份模型（训练时多个工人并行）"""
    with open(模型路径, "r", encoding="utf-8") as f:
        文本 = f.read()
    m = mujoco.MjModel.from_xml_string(文本)
    d = mujoco.MjData(m)
    return m, d


def 参数个数():
    """本能区连接总数（权重+偏置），进化式搜索只搜这么多参数"""
    return 隐层数 * 输入数 + 隐层数 + 输出数 * 隐层数 + 输出数


def 网络前向(theta, 特征):
    """感觉特征 -> 6 路肌肉发力。单隐层 tanh 网络（连接就存在 theta 里）"""
    n1 = 隐层数 * 输入数
    W1 = theta[:n1].reshape(隐层数, 输入数)
    b1 = theta[n1:n1 + 隐层数]
    n2 = 输出数 * 隐层数
    W2 = theta[n1 + 隐层数:n1 + 隐层数 + n2].reshape(输出数, 隐层数)
    b2 = theta[n1 + 隐层数 + n2:]
    h = np.tanh(W1 @ 特征 + b1)
    发力 = np.tanh(W2 @ h + b2) * 力矩上限
    return 发力


def 提取特征(d):
    """从仿真数据里取 17 维感觉特征"""
    qpos = d.qpos.copy()
    qvel = d.qvel.copy()
    角 = (qpos[1:] - 特征中心)          # 8 维：高度、俯仰、6 关节
    速 = qvel * 0.2                     # 9 维：速度（含前进速度，阻尼用）
    return np.concatenate([角, 速])


def 倒下(d):
    """摔倒判定：骨盆太低 或 躯干俯仰太大"""
    return d.qpos[1] < 0.55 or abs(d.qpos[2]) > 0.6


def 随机初始(m, d, 随机):
    """每回合从稍微不同的姿势开始（微蹲、微前倾、带点速度），逼模型学会真平衡"""
    mujoco.mj_resetData(m, d)
    d.qpos[0] = 0.0                                   # X
    d.qpos[1] = 随机.uniform(0.86, 0.92)              # Z：略高于站直高度，落地一下
    d.qpos[2] = 随机.uniform(-0.08, 0.08)             # 俯仰
    d.qpos[3] = 随机.uniform(-0.06, 0.06)             # 左髋
    d.qpos[4] = 随机.uniform(0.0, 0.10)               # 左膝（允许微蹲）
    d.qpos[5] = 随机.uniform(-0.04, 0.04)             # 左踝
    d.qpos[6] = 随机.uniform(-0.06, 0.06)             # 右髋
    d.qpos[7] = 随机.uniform(0.0, 0.10)               # 右膝
    d.qpos[8] = 随机.uniform(-0.04, 0.04)             # 右踝
    d.qvel[0] = 随机.uniform(-0.15, 0.15)             # 前进速度
    d.qvel[1] = 随机.uniform(-0.15, 0.15)             # 上下速度
    d.qvel[2] = 随机.uniform(-0.3, 0.3)               # 俯仰速度
    mujoco.mj_forward(m, d)


def 跑一回合(m, d, theta, 种子=None, 记录=False):
    """用一组连接跑一回合：返回(存活秒数, 累计奖励, 每步俯仰, 每步高度)"""
    随机 = np.random.RandomState(0 if 种子 is None else 种子)
    随机初始(m, d, 随机)
    累计 = 0.0
    俯仰史 = []
    高度史 = []
    for 步 in range(最大步数):
        特征 = 提取特征(d)
        d.ctrl[:] = 网络前向(theta, 特征)
        mujoco.mj_step(m, d)
        俯仰史.append(d.qpos[2])
        高度史.append(d.qpos[1])
        if 倒下(d):
            break
        累计 += 1.0 + 0.3 * (1.0 - min(1.0, abs(d.qpos[2]) / 0.4)) \
                 - 0.8 * abs(d.qpos[1] - 0.89)
    存活秒 = len(俯仰史) * dt
    if 记录:
        return 存活秒, 累计, np.array(俯仰史), np.array(高度史)
    return 存活秒, 累计


def 考察(theta, 回合数=20):
    """训练后考核：多回合随机起始，报平均能站几秒"""
    m, d = 建模型()
    秒们 = []
    for 种子 in range(回合数):
        秒, _ = 跑一回合(m, d, theta, 种子=种子 + 1000)
        秒们.append(秒)
    秒们 = np.array(秒们)
    return 秒们.mean(), 秒们.min(), 秒们.max()


if __name__ == "__main__":
    print("模型文件:", 模型路径)
    m, d = 建模型()
    print(f"自由度: {m.nq} | 肌肉(执行器): {m.nu} | 本能区连接数: {参数个数()}")
    print(f"特征维度: {输入数} | 每回合最长: {最大秒数} 秒 ({最大步数} 步 x {dt * 1000:.0f}ms)")

    # 基线：完全不发力 + 随机发力各 10 回合，看"天生"能站多久
    theta0 = np.zeros(参数个数())
    m0, d0 = 建模型()
    秒0 = [跑一回合(m0, d0, theta0, 种子=s)[0] for s in range(10)]
    print(f"基线·肌肉完全不发力: 平均 {np.mean(秒0):.2f} 秒 / 最长 {np.max(秒0):.2f} 秒")

    thetaR = np.random.RandomState(1).uniform(-0.2, 0.2, 参数个数())
    mR, dR = 建模型()
    秒R = [跑一回合(mR, dR, thetaR, 种子=s)[0] for s in range(10)]
    print(f"基线·随机乱发力:     平均 {np.mean(秒R):.2f} 秒 / 最长 {np.max(秒R):.2f} 秒")
    print("（两条基线都远小于 3 秒 = 任务确实需要学习才能站稳）")
