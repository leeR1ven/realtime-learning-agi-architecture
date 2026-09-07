import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
np.random.seed(2026)
import 视觉前处理_visual_preprocess as V
import 听觉前处理_auditory_preprocess as A
import 运动输出区_motor_output as M
import 前额叶区_prefrontal as P

def code(m, x): return m.网.前向传播(x).astype(bool).copy()
def stats(codes):
    baseline=codes[0]
    return {'active':[int(c.sum()) for c in codes], 'xor_from_baseline':[int(np.count_nonzero(c != baseline)) for c in codes], 'unique':len({c.tobytes() for c in codes})}
start=time.perf_counter()
images=[np.zeros((96,96,3),dtype=np.uint8)]
for x in (0,24,48,72):
    im=images[0].copy();im[32:64,x:x+24]=(255,40,20);images.append(im)
vcodes=[code(V,V.图像转信号(im)) for im in images]
spectra=[np.zeros(1500)]
for i in (100,300,800,1200):
    s=np.zeros(1500); s[i]=1;spectra.append(s)
acodes=[code(A,A.频谱转信号(s)) for s in spectra]
forces=[np.zeros(200)]
for m in range(6):
    f=np.zeros(200);f[m]=1;forces.append(f)
for i in range(20):
    f=np.zeros(200);f[:6]=np.random.uniform(0,1,6);forces.append(f)
mcodes=[code(M,M.发力转信号(f)) for f in forces]
pvcodes=[P.前额网.前向传播(c,acodes[0],mcodes[0]).astype(bool).copy() for c in vcodes]
pacodes=[P.前额网.前向传播(vcodes[0],c,mcodes[0]).astype(bool).copy() for c in acodes]
reverse=M.反向赫布网络(M.特征神经元池(M.对数))
for f,c in zip(forces,mcodes):
    code(M,M.发力转信号(f));reverse.学习一帧(M.池.激活.copy(),c)
recalled=[]
for f,c in zip(forces[:7],mcodes[:7]):
    out=reverse.前向(c)
    recalled.append({'expected':f[:6].tolist(),'decoded':(M.解码发力(out)[:6]/M.每档对数).tolist(),'all200_mae':float(np.abs(M.解码发力(out)/M.每档对数-f).mean())})
labels=np.repeat(np.arange(3),[P.视宽,P.听宽,P.运宽]); cross=(labels[:,None]!=labels[P.前额网.来源[0]])
report={'seed':2026,'visual_black_then_red_blocks':stats(vcodes),'audio_silence_then_single_bins':stats(acodes),'motor_zero_then_6_single_muscles_then20_random':stats(mcodes),'pfc_visual_probe':stats(pvcodes),'pfc_audio_probe':stats(pacodes),'pfc_layer1_cross_edges':int(cross.sum()),'pfc_layer1_edges':int(cross.size),'pfc_layer1_cross_targets':int(cross.any(axis=1).sum()),'motor_reverse_after27_first6muscle_patterns':recalled,'null_reverse_first6':(M.解码发力(reverse.前向(np.zeros(4000,dtype=bool)))[:6]/M.每档对数).tolist(),'elapsed_seconds':time.perf_counter()-start}
print(json.dumps(report,indent=2,ensure_ascii=False))
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
