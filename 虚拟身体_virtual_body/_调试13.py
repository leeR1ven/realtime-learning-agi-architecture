"""_调试13.py —— 肌肉强度缩放扫描(不跳): 找"踝髋单扛"上限≈16-20 的比例
用法: python _调试13.py
"""
import numpy as np
import 身体 as B
import 感官本能 as S

秒数 = 3.5
量表 = [8, 12, 16, 20, 24, 28, 32]

def 跑(冲):
    S.重置本能()
    身 = B.建身体()
    B.加推力(身, 冲)
    while 身["时间"] < 秒数:
        st = B.读状态(身)
        _, 扭矩, _ = S.本能连接(st)
        for _ in range(10):
            B.物理一步(身, 扭矩)
        st = B.读状态(身)
        倒, 因 = B.摔倒了吗(st)
        if 倒:
            return False, 因
    return True, ""

原踝Kp, 原踝Kd, 原髋Kp, 原髋Kd = S.Kp踝, S.Kd踝, S.Kp髋直, S.Kd髋直
for 踝比 in (0.20, 0.30, 0.40, 0.55):
    for 髋比 in (0.6, 1.0):
        S.Kp踝, S.Kd踝 = 原踝Kp*踝比, 原踝Kd*踝比
        S.Kp髋直, S.Kd髋直 = 原髋Kp*髋比, 原髋Kd*髋比
        行 = []
        for 冲 in 量表:
            for 方 in (1.0, -1.0):
                ok, 因 = 跑(冲*方)
                if not ok:
                    行.append(f"{冲*方:+.0f}:{因[:4]}")
                    break
            else:
                行.append(f"{冲:+.0f}:过")
        print(f"踝x{踝比} 髋x{髋比}: " + " ".join(行))
S.Kp踝, S.Kd踝, S.Kp髋直, S.Kd髋直 = 原踝Kp, 原踝Kd, 原髋Kp, 原髋Kd