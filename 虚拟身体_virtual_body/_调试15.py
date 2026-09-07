import numpy as np
import 身体 as B
import 感官本能 as S

for 比 in (0.33, 0.35, 0.37, 0.40):
    S.Kp踝, S.Kd踝 = 200.0*比, 90.0*比
    行 = []
    for 冲 in (12.0, 16.0, 20.0, 24.0, 28.0):
        for 方 in (1.0, -1.0):
            S.重置本能()
            身 = B.建身体()
            B.加推力(身, 冲*方)
            倒 = False
            while 身["时间"] < 3.5:
                st = B.读状态(身)
                _, 扭矩, _ = S.本能连接(st)
                for _ in range(10):
                    B.物理一步(身, 扭矩)
                st = B.读状态(身)
                倒, 因 = B.摔倒了吗(st)
                if 倒:
                    break
            if 倒:
                行.append(f"{冲*方:+.0f}:{因[:3]}")
                break
        else:
            行.append(f"{冲:+.0f}:过")
    print(f"踝x{比}: " + " ".join(行))