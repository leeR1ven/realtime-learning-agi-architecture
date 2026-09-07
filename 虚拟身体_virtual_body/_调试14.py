import numpy as np
import 身体 as B
import 感官本能 as S

S.Kp踝, S.Kd踝 = 200.0*0.30, 90.0*0.30   # 踝 60 / 27
for 冲 in (12.0, -12.0, 20.0, -20.0, 28.0, -28.0):
    S.重置本能()
    身 = B.建身体()
    B.加推力(身, 冲)
    最大 = 0.0
    while 身["时间"] < 3.5:
        st = B.读状态(身)
        最大 = max(最大, abs(st["前倾"]))
        _, 扭矩, _ = S.本能连接(st)
        for _ in range(10):
            B.物理一步(身, 扭矩)
        st = B.读状态(身)
        倒, 因 = B.摔倒了吗(st)
        if 倒:
            print(f"冲={冲:+6.1f}: 摔({因}) 最大前倾={np.degrees(最大):.1f}°")
            break
    else:
        print(f"冲={冲:+6.1f}: 站住 最大前倾={np.degrees(最大):.1f}°")