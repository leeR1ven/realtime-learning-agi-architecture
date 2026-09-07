"""_调试17.py —— Kp踝=60 下 静站/起身 稳定性"""
import numpy as np
import 身体 as B
import 感官本能 as S

S.Kp踝, S.Kd踝 = 60.0, 27.0
角度转度 = 57.2958

def 跑(初始角度, 秒):
    S.重置本能()
    身 = B.建身体(角度=初始角度)
    最大 = 0.0
    while 身["时间"] < 秒:
        st = B.读状态(身)
        最大 = max(最大, abs(st["前倾"]))
        _, 扭矩, _ = S.本能连接(st)
        for _ in range(10):
            B.物理一步(身, 扭矩)
        st = B.读状态(身)
        倒, 因 = B.摔倒了吗(st)
        if 倒:
            return False, 因, st, 最大
    return True, "", st, 最大

ok1, 因1, st1, m1 = 跑([0,0,0], 8.0)
print(f"静站8s: {'站住' if ok1 else 因1} 最大前倾={np.degrees(m1):.2f}° 结束前倾={np.degrees(st1['前倾']):.2f}°")
ok2, 因2, st2, m2 = 跑([0.35, -0.64, 0.10], 12.0)
print(f"深蹲起身12s: {'站住' if ok2 else 因2} 结束膝弯={np.degrees(st2['膝弯']):.1f}° 髋屈={np.degrees(st2['髋屈']):.1f}°")