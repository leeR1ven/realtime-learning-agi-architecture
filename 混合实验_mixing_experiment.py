import numpy as np
import 听觉前处理_auditory_preprocess as 听

池 = 听.池
网 = 听.网
R = 池.总数

def 听一次(信号):
    return 网.前向传播(信号).astype(bool)

def 简报(名, 码):
    idx = np.nonzero(码)[0]
    if len(idx) == 0:
        print(f"{名}: 亮 0 个"); return
    hist = np.bincount(idx // (R // 10), minlength=10)
    簇 = ' '.join(f"[{i * (R // 10)}~{(i + 1) * (R // 10)}):{hist[i]}" for i in range(10) if hist[i])
    print(f"{名}: 亮 {len(idx)} 个 | 位置分布 {簇}")

print("======== 实验1：两个'局部声音'同时进网络（你现有的听觉网络）========")
def 造音(中心, 半径, 种子):
    rng = np.random.default_rng(种子)
    s = np.zeros(R)
    for p in range(-半径, 半径 + 1):
        s[(中心 + p) % R] = rng.uniform(0.0, 1.0)
    return s

for 名, c1, c2 in [("两音很近(间隔200)", 3900, 4100), ("两音中等(间隔1500)", 3000, 4500), ("两音很远(间隔4000)", 2000, 6000)]:
    print("----", 名, "----")
    A = 造音(c1, 300, 1); B = 造音(c2, 300, 2)
    混 = np.maximum(A, B)   # 两路声音同时进同一个池 = 信号叠加
    OA = 听一次(A); OB = 听一次(B); OM = 听一次(混)
    简报("只听声音1", OA); 简报("只听声音2", OB); 简报("两路同时(混合)", OM)
    if OM.sum():
        print(f"  混合码里: 含声音1的细胞 {int((OM & OA).sum())} | 含声音2的细胞 {int((OM & OB).sum())} | 两簇都没有的新细胞 {int((OM & ~OA & ~OB).sum())}")

print()
print("======== 实验2：'中间簇'需要什么？（纯神经元：求和+平滑+门槛）========")
def 环卷积(x, sigma):
    g = np.exp(-0.5 * (np.minimum(np.abs(np.arange(R) - 0), R - np.abs(np.arange(R) - 0))) ** 2 / sigma ** 2)
    g = g / g.sum()
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(g)).real

def 峰位置(x):
    # 找显著局部峰（平滑后）
    xs = 环卷积(x, 20)
    峰 = []
    for p in np.nonzero((xs[1:-1] > xs[:-2]) & (xs[1:-1] > xs[2:]))[0]:
        p += 1
        if xs[p] > 0.3 * xs.max():
            峰.append((p, xs[p]))
    return 峰

for c2 in (4200, 4700, 7000):
    A = np.exp(-0.5 * (np.minimum(np.abs(np.arange(R) - 4000), R - np.abs(np.arange(R) - 4000))) ** 2 / (60 ** 2))
    B = np.exp(-0.5 * (np.minimum(np.abs(np.arange(R) - c2), R - np.abs(np.arange(R) - c2))) ** 2 / (60 ** 2))
    混 = 环卷积(A + B, 300)
    ps = 峰位置(混)
    print(f"声音1在4000, 声音2在{c2}: 混合后显著峰位置 = {[p for p, v in ps]} (中间4000~{c2}之间)")