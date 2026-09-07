"""训练.py —— 本能期训练：进化式搜索（不依赖梯度，直接搜本能区连接）

原理（和你说的"强化学习只改可实时学习的连接"一致）：
  1. 给本能区连接(theta)加一点随机扰动 → 生成一群"变异婴儿"
  2. 每个婴儿用同一套站立任务跑一回合，看能站几秒（存活秒 = 奖励）
  3. 站得久的婴儿留下，它们的连接均值成为下一代起点
  4. 反复几百代后，连接收敛成"天生会站稳" → 冻结 = 本能

训练结束输出：
  本能_站立.npz   —— 冻结用的一组连接（以后接进模型当固定本能）
  训练报告.txt    —— 训练过程与考核结果
"""
import os
import time
import numpy as np
from multiprocessing import Pool

import 任务 as T

环境数 = 96            # 每代生成多少个"变异婴儿"
精英数 = 12            # 每代留下几个最好的
每体回合数 = 2         # 每个婴儿测几回合取平均（压制运气噪声）
最大代数 = 3000        # 最多进化多少代
起步扰动 = 0.35        # 一开始变异幅度（探索大）
最小扰动 = 0.02        # 后期变异幅度（精调）
成功标准 = 2.6         # 精英平均存活 >= 2.6 秒 = 学会站稳，提前停止
停滞代数 = 120         # 连续多少代没进步就围绕历史最佳重启探索
报告路径 = os.path.join(T.目录, "训练报告.txt")
权重路径 = os.path.join(T.目录, "本能_站立.npz")

# ---- 工人进程：每个工人自己建一份仿真模型，反复跑回合 ----
_m = None
_d = None


def 工人初始化():
    global _m, _d
    _m, _d = T.建模型()


def 评估一个(任务参数):
    theta, 种子们 = 任务参数
    秒们 = []
    for 种子 in 种子们:
        秒, _ = T.跑一回合(_m, _d, theta, 种子=种子)
        秒们.append(秒)
    return float(np.mean(秒们))


def 主训练():
    t0 = time.perf_counter()
    P = T.参数个数()
    随机 = np.random.RandomState(2026)
    mu = np.zeros(P)                    # 从"完全不会发力"开始
    sigma = 起步扰动
    最优theta = mu.copy()
    最优秒 = 0.0
    无进展代数 = 0

    报告 = []
    报告.append("本能期训练报告 —— 站立任务")
    报告.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    报告.append(f"本能区连接数: {P} | 每代婴儿: {环境数} x {每体回合数}回合 | 最长回合: {T.最大秒数} 秒")
    报告.append(f"参数: 起步扰动 {起步扰动} -> 最小 {最小扰动} | 成功标准 {成功标准} 秒")
    报告.append("")

    print(f"本能区连接数: {P} | 每代 {环境数} 个婴儿 x {每体回合数} 回合 | 目标: 站稳 {T.最大秒数} 秒")
    print("开始进化...", flush=True)

    with Pool(initializer=工人初始化) as 池:
        for 代 in range(1, 最大代数 + 1):
            婴儿群 = mu + sigma * 随机.standard_normal((环境数, P))
            任务们 = []
            for i in range(环境数):
                种子们 = 随机.randint(0, 2 ** 31 - 1, size=每体回合数)
                任务们.append((婴儿群[i], 种子们))
            秒们 = np.array(池.map(评估一个, 任务们))

            排序 = np.argsort(秒们)[::-1]
            精英秒 = 秒们[排序[:精英数]]
            mu = 婴儿群[排序[:精英数]].mean(axis=0)   # 精英连接的均值 = 下一代起点

            if 秒们[排序[0]] > 最优秒:
                最优秒 = 秒们[排序[0]]
                最优theta = 婴儿群[排序[0]].copy()
                无进展代数 = 0
            else:
                无进展代数 += 1

            sigma = max(最小扰动, sigma * 0.997)

            if 代 % 10 == 0 or 代 == 1:
                msg = (f"第 {代:>3} 代 | 本代最佳 {秒们[排序[0]]:.2f} 秒"
                       f" | 精英平均 {精英秒.mean():.2f} 秒"
                       f" | 历史最佳 {最优秒:.2f} 秒"
                       f" | 扰动 {sigma:.3f}"
                       f" | 用时 {time.perf_counter() - t0:.0f} 秒")
                print(msg, flush=True)
                报告.append(msg)

            # 一直没进展就加大探索（可能是困在只会摔倒的区域）
            if 无进展代数 >= 停滞代数:
                mu = 最优theta + 0.35 * 随机.standard_normal(P)
                sigma = max(sigma, 0.5)
                无进展代数 = 0
                报告.append(f"第 {代} 代：停滞 {停滞代数} 代，围绕历史最佳({最优秒:.2f}秒)重启探索")
                print(f"第 {代} 代：停滞，围绕历史最佳重启探索", flush=True)

            if 精英秒.mean() >= 成功标准:
                报告.append(f"第 {代} 代：精英平均 {精英秒.mean():.2f} 秒 >= {成功标准}，提前停止")
                print("学会站稳了，提前停止。", flush=True)
                break
        else:
            报告.append(f"跑满 {最大代数} 代，历史最佳 {最优秒:.2f} 秒")

    # ---- 考核：拿最终连接，用全新随机起始跑 30 回合 ----
    考核theta = mu if 最优秒 < 2.5 else 最优theta
    平均, 最小, 最大 = T.考察(考核theta, 回合数=30)
    msg2 = (f"最终考核(30 回合随机起始): 平均 {平均:.2f} 秒"
            f" | 最差 {最小:.2f} 秒 | 最好 {最大:.2f} 秒")
    print(msg2, flush=True)
    报告.append("")
    报告.append(msg2)
    报告.append(f"总用时: {time.perf_counter() - t0:.0f} 秒")

    np.savez(权重路径,
             theta=考核theta,
             隐层数=T.隐层数,
             输入数=T.输入数,
             输出数=T.输出数,
             平均存活秒=平均)
    with open(报告路径, "w", encoding="utf-8") as f:
        f.write("\n".join(报告) + "\n")
    print("已保存:", 权重路径)
    print("已保存:", 报告路径)


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)
    主训练()
