"""可视化.py —— 看火柴人按当前"本能连接"怎么站、怎么摔

用法：
  python 可视化.py          打开实时窗口，一直循环随机起始的回合
  python 可视化.py 录屏     不弹窗，把站得最长的一回合存成图片到 预览\
"""
import os
import sys
import time
import numpy as np

import 任务 as T

目录 = os.path.dirname(os.path.abspath(__file__))
权重路径 = os.path.join(目录, "本能_站立.npz")
预览目录 = os.path.join(目录, "预览")


def 载入连接():
    if os.path.exists(权重路径):
        z = np.load(权重路径)
        return z["theta"]
    return np.zeros(T.参数个数())


def 跑一回合选种(theta, 找种子=range(50)):
    """快速扫一批随机起始，返回(种子, 存活秒)，用来挑一段好看的来录"""
    m, d = T.建模型()
    最佳 = (0, 0.0)
    for 种子 in 找种子:
        秒, _ = T.跑一回合(m, d, theta, 种子=种子)
        if 秒 > 最佳[1]:
            最佳 = (种子, 秒)
    return 最佳


def 打开实时窗口(theta):
    """MuJoCo 自带的 3D 窗口，能看到小人实时站立/摔倒"""
    import mujoco
    import mujoco.viewer  # 新版本需显式导入才有实时窗口
    m, d = T.建模型()
    viewer = mujoco.viewer.launch_passive(m, d)
    print("窗口已打开（3D 视角，可按住鼠标拖动旋转）。关掉窗口即退出。", flush=True)
    回合 = 0
    while viewer.is_running():
        回合 += 1
        随机 = np.random.RandomState(回合 * 7 + 3)
        T.随机初始(m, d, 随机)
        步 = 0
        for 步 in range(T.最大步数):
            _t0 = time.perf_counter()
            特征 = T.提取特征(d)
            d.ctrl[:] = T.网络前向(theta, 特征)
            mujoco.mj_step(m, d)
            viewer.sync()
            if T.倒下(d):
                break
            time.sleep(max(0.0, T.dt - (time.perf_counter() - _t0)))
        秒 = (步 + 1) * T.dt
        print(f"第 {回合} 回合: 存活 {秒:.2f} 秒"
              f"{'（站满 ' + str(T.最大秒数) + ' 秒，完美！）' if 秒 >= T.最大秒数 else ''}",
              flush=True)
    viewer.close()


def 录屏(theta):
    """选一段站得最久的回合，逐帧渲染成 PNG"""
    import mujoco
    from PIL import Image
    种子, 秒 = 跑一回合选种(theta)
    print(f"选中最长回合: 种子 {种子}，存活 {秒:.2f} 秒", flush=True)

    m, d = T.建模型()
    T.随机初始(m, d, np.random.RandomState(种子))
    渲染 = mujoco.Renderer(m, 480, 640)
    os.makedirs(预览目录, exist_ok=True)
    for 旧 in os.listdir(预览目录):
        os.remove(os.path.join(预览目录, 旧))

    相机 = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(相机)
    相机.lookat[:] = [0.0, 0.0, 0.85]
    相机.distance = 2.6
    相机.azimuth = 130
    相机.elevation = -15

    文件们 = []
    步 = 0
    for 步 in range(T.最大步数):
        特征 = T.提取特征(d)
        d.ctrl[:] = T.网络前向(theta, 特征)
        if 步 % 3 == 0:                      # 每 0.06 秒存一帧
            渲染.update_scene(d, camera=相机)
            图像 = Image.fromarray(渲染.render())
            名 = os.path.join(预览目录, f"帧_{步 // 3:03d}.png")
            图像.save(名)
            文件们.append(名)
        mujoco.mj_step(m, d)
        if T.倒下(d):
            break
    print(f"已保存 {len(文件们)} 帧到: {预览目录}")
    return 文件们


if __name__ == "__main__":
    theta = 载入连接()
    print(f"本能区连接: {len(theta)} 个（来自 {os.path.basename(权重路径) if os.path.exists(权重路径) else '全零'}）")
    if len(sys.argv) > 1 and sys.argv[1] == "录屏":
        录屏(theta)
    else:
        打开实时窗口(theta)

