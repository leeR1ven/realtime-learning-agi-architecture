"""_调试13b.py —— 细化 踝 0.30~0.40 之间找上限≈16-20"""
import numpy as np
import 身体 as B
import 感官本能 as S

秒数 = 3.5
量表 = [10, 12, 14, 16, 18, 20, 24, 28]
原踝Kp, 原踝Kd = S.Kp踝, S.Kd踝
for 踝比 in (0.30, 0.32, 0.34, 0.36, 0.38):
    S.Kp踝, S.Kd踝 = 原踝Kp*踝比, 原踝Kd*踝比
    行 = []
    for 冲 in 量表:
        for 方 in (1.0, -1.0):
            S.重置本能()
            身 = B.建身体()
            B.加推力(身, 冲*方)
            while 身["时间"] < 秒数:
                st = B.读状态(身)
                _, 扭矩, _ = S.本能连接(st)
                for _ in range(10):
                    B.物理一步(身, 扭矩)
                st = B.读状态(身)
                倒, 因 = B.摔倒了吗(st)
                if 倒:
                    break
            else:
                continue
            break
        else:
            行.append(f"{冲:+.0f}:过")
            continue
        break
    print(f"踝x{踝比}: " + " ".join(行) if 行 else f"踝x{踝比}: 全过(到28)")
S.Kp踝, S.Kd踝 = 原踝Kp, 原踝Kd