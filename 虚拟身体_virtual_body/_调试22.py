"""_调试22.py —— 向后 -28 的跳步变体"""
import numpy as np
import 身体 as B
import 感官本能 as S

S.踝上限 = 45.0
S.关节肌肉表[0] = (0, 1, -45.0, 45.0)
跳步冷却 = 0.20
秒数 = 4.0

def 跑(冲, 变体):
    S.重置本能()
    身 = B.建身体()
    B.加推力(身, 冲)
    上次跳 = -9.0
    跳数 = 0
    最大 = 0.0
    while 身["时间"] < 秒数:
        st = B.读状态(身)
        最大 = max(最大, abs(st["前倾"]))
        if 身["模式"] == "空":
            扭矩 = S.空中本能(st)
        else:
            _, 扭矩, 迈 = S.本能连接(st)
            if (迈 and 身["时间"] - 上次跳 > 跳步冷却
                    and (abs(st["前倾速度"]) > 1.8 or abs(st["前倾"]) > 0.25)):
                扭矩 = S.空中本能(st)
                B.起跳(身, S.跳升速)
                上次跳 = 身["时间"]
                跳数 += 1
        for _ in range(10):
            if 身["模式"] == "空":
                if B.悬空一步(身, 扭矩):
                    break
            else:
                B.物理一步(身, 扭矩)
        st = B.读状态(身)
        if 身["模式"] == "站":
            倒, 因 = B.摔倒了吗(st)
            if 倒:
                return f"摔({因})", 最大, 跳数
    return "站住", 最大, 跳数

for 名, 改 in (
    ("基准(升速2.0)", lambda: setattr(S, "跳升速", 2.0)),
    ("升速2.4", lambda: setattr(S, "跳升速", 2.4)),
    ("膝空0.45", lambda: (setattr(S, "跳升速", 2.4), setattr(S, "膝空目标", 0.45))),
    ("空中踝Kp220", lambda: (setattr(S, "跳升速", 2.4), setattr(S, "空中踝Kp", 220.0), setattr(S, "空中踝Kd", 30.0))),
    ("空中髋Kd80", lambda: (setattr(S, "跳升速", 2.4), setattr(S, "空中髋Kd", 80.0))),
):
    改()
    r = 跑(-28.0, 名)
    print(f"{名}: {r[0]} 最大{np.degrees(r[1]):.1f}° 跳{r[2]}")