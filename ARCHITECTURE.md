# Architecture Overview (English)

An overview of the model from neurons up to the closed brain-body loop.
All source code uses Chinese identifiers on purpose; keep the corresponding files at hand
while reading. Source files live at the repository root; each experiment lives in its own folder.

---

## 0. One-sentence summary

This is a **pure-spiking/connectionist brain model**: sensory signals enter feed-forward
networks and become sparse neural firing; the prefrontal area (PFC) receives all streams and
acts as the "thinking hub" (association, attention, gating); the PFC can **mirror its output
back** to the sensory areas (imagining something) and can drive the motor area; the motor area
expands a sparse command code into muscle forces through a **reverse network**; body actions
produce new sensory input. A **shared hippocampal time ring** provides one global clock, and
the memory areas hang every frame of firing onto "the time it happened", producing replayable
continuous memory.

Learning happens only in **connection weights**. Thinking, recall and movement are just spikes
travelling along existing connections.

---

## 1. Three building blocks shared by all areas

### 1.1 Paired feature neurons (`class 特征神经元池`)
- Every feature is represented by a **positive/negative pair**; the two neurons compete and
  **exactly one fires**.
- Signals accumulate into the pool; `更新所有对()` lets the winner of each pair fire and then
  clears the signal.
- No extra rule is needed to keep the model at "half active": pair competition guarantees it.
- Strength is expressed by thermometer coding: a feature is split into `每档对数` pairs;
  level L means the first L positive neurons fire and the rest fire negative, giving N+1 levels.

### 1.2 Sparse locally-connected network (`class 神经网络`)
- Every hidden layer has the same width as the input pool.
- A neuron connects only to the previous layer within a **ring neighbourhood** (default ±10,
  first and last positions adjacent). Weights are drawn once at birth from `[0, 权重范围)`
  and **never change**; a neuron fires when its weighted sum `>=` its threshold.
- These networks *process signals*; they do **not** use Hebbian learning. Predictable and fast.
- Each file's header centralizes `层数 / 连接半径 / 权重范围 / 阈值初值`.

### 1.3 Connection tables (`权重连接管理_weight_manager.py`, `class 连接表`)
- Used by the **plastic** connections only (three memory areas, PFC association).
- Rules: each co-activation adds `强化量`; once the table reaches `警戒线` (default 100k)
  entries it periodically multiplies all weights by `衰减率` (0.9) every `衰减间隔` frames;
  weights below `消失下限` (0.05) are deleted.
- Net effect: frequently used connections survive; rarely used ones fade to zero, so the number
  of connections does not grow without bound.

---

## 2. Areas

### 2.1 Visual preprocessing (`视觉前处理_visual_preprocess.py`)
- The image is downsampled to a 24×16 mosaic; each "big pixel" has R/G/B channels, and each
  channel uses `每档对数=10` pairs as a 0-10 brightness thermometer.
- Feature pairs = 24×16×3×10 = 11,520 (23,040 neurons); 3 layers, ring radius 10.
- The output layer is calibrated to a target activity (≈240 cells/frame): a per-layer gain α
  keeps firing counts stable regardless of input strength (no silence, no seizure).
  `解码亮度()` reads the picture back for verification.

### 2.2 Auditory preprocessing (`听觉前处理_auditory_preprocess.py`)
- Mono input; the spectrum is compressed to 1,500 bands, each with `每档对数=10` pairs
  as a 0-10 loudness thermometer.
- Feature pairs = 1,500×10 = 15,000 (30,000 neurons); 3 layers.
- `压缩频谱()` maps arbitrary-length spectra onto the fixed 1,500 bands.

### 2.3 Motor output area (`运动输出区_motor_output.py`)
- Muscles use force thermometers: `肌肉数=200`, each with `每档对数=10` pairs,
  0-10 levels (11 force levels), 2,000 pairs total (4,000 neurons).
- The forward network is like the sensory areas: sparse command codes fire on the last layer;
  `解码发力()` turns firing back into 0-1 force per muscle.
- The **reverse Hebbian network** (`class 反向赫布网络`) solves "command code -> muscle
  features":
  - ① Monotonic: a firing command neuron wires to every muscle feature active that frame;
    `一次学会=True` saturates a synapse at 1, so once learned it is never forgotten;
  - ② Gated growth: before adding new synapses it tries recall; if recall already matches the
    real muscles, no new synapses are added, so repeating the same actions does not flood memory;
  - ③ Innate mirror baseline: recall always adds a weak fixed mirror-line vote, so even a novel
    command gives a rough muscle direction (no learning, no memory cost).
- `镜像投票()` is the birth-fixed mirror return line. It is only applied when the reverse
  network width equals the module-wide motor net (miniature simulators use their own small
  reverse net and skip it automatically).

### 2.4 Three memory areas (`视觉记忆区_visual_memory.py` / `听觉记忆区_auditory_memory.py` / `运动记忆区_motor_memory.py`)
- Identical structure; each plugs into its own output layer. Each memory area's width
  **automatically equals** `网.输出层数量` of its source, so resizing a preprocessing network
  resizes memory areas without manual synchronization.
- Two directions per area:
  - `特征到时间`: which time neurons were active when a feature pattern fired
    ("when did this happen");
  - `时间到特征`: which features fired at a moment ("what happened then").
- `学习(旧时间, 新时间, 新特征)` stores a frame given the previous/next time index;
  `回忆(特征模式, 步数, 阈值)` replays a chain step by step from a cue.

### 2.5 Hippocampal time ring (`海马体时间区_hippocampal_time.py`)
- One **shared** timeline: 1,728,000 time neurons connected head-to-tail in a ring; each neuron
  covers 0.05 s; one external frame = 0.1 s = two consecutive neurons; one full lap =
  1,728,000 × 0.05 s = 24 hours.
- Two consecutive time neurons per frame keep consecutive frames tightly chained
  (no skipped or crossed frames).
- Memory areas use it to order events; replay re-lights features in time order.

### 2.6 Prefrontal area (`前额叶区_prefrontal.py`)
The PFC is the thinking hub, built in four layers:

1. **PFC network** (`class 前额叶神经网络`): input = visual output layer + auditory output
   layer + the brain-side motor command code, concatenated. Each stream is first "diffused"
   (every spike lights ±`扩散宽度` neighbours) so that sparse codes can drive the local
   network. Most cells are single-stream "identity cells" (local receptive field only); a
   fraction of "convergence cells" (`汇合比例`, default 0.25) additionally receive
   `汇合每路线数` (64) long-range lines from each of the other two streams and have a raised
   threshold (`汇合阈值倍数` 1.2), so they fire only when **several streams agree** — the
   mechanism for fusing multiple inputs into one thought cluster. The output threshold (1.2) is
   above the hidden threshold (0.55), keeping output sparse.

2. **Innate reciprocal mirror lines** (`class 天生互惠返回线`): PFC output is projected back
   to the three input segments along birth-fixed mirror connections; `返回保留比例=0.3` keeps
   cells above 30% of the peak. Not learned, never forgotten — "thinking of X" re-activates the
   sensory area that would produce X (innate reciprocal wiring).

3. **PFC association area** (`class 前额叶联想区`): temporal Hebbian learning inside the PFC
   (previous thought -> next thought association plus disinhibition connections). `微步()`
   applies **dynamic global inhibition**: it tracks the "typical thought size" and keeps
   activity inside 0.5-1.3× that size. Too many cells -> keep only the strongest and raise
   inhibition; too few -> relax inhibition and keep cells that actually have signal.

4. **Focus gate** (`class 专注闸门`): inhibition vs. disinhibition pools gate three outward
   pathways: input (external signals cannot interrupt deep thought), recall (no recall when not
   wanted), and movement (think without moving until the gate opens). Focus is not an instant
   switch: inhibition takes a few micro-steps to overcome the spontaneous disinhibitory
   activity, and recovers gradually when the thought ends.

A **hippocampal index** (`class 海马索引`) locks PFC thought patterns onto time neurons.
`锁定胜出()` decides "this frame corresponds to a real time event" by requiring the winner's
vote to clearly beat the runner-up (base votes 80, margin 40), so noise is not stored as events.

### 2.7 Closed-loop flow (`闭环流程_closed_loop.py`)
A "parts checklist + integration run" that plugs everything (hippocampal index, memory recall,
association area, focus gate, reciprocal return lines) into one closed loop. It reads but never
edits the other files. Run `python 闭环流程_closed_loop.py [frames]`.

---

## 3. Fixed vs. plastic weights (important)

| Component | Weights | Why |
| --- | --- | --- |
| Visual/auditory/motor/PFC forward networks | fixed at birth, never change | stable signal-processing circuits |
| Reciprocal mirror lines / `镜像投票` | fixed at birth, never change | innate return wiring |
| Three memory areas | Hebbian + periodic decay | "what happened when" |
| PFC association area | Hebbian + dynamic inhibition | order relations between thoughts |
| Motor reverse Hebbian network | one-shot, monotonic | command->muscle paths, never forgotten |

---

## 4. Where to change numbers (linkage)

- Every file centralizes sizes in a "global creation" block at the top.
- `视觉前处理_visual_preprocess.py`: `大像素行数/列数`, `通道数`, `每档对数`, `层数`, `连接半径`.
- `听觉前处理_auditory_preprocess.py`: `频带数`, `每档对数`, `层数`, `连接半径`.
- `运动输出区_motor_output.py`: `肌肉数`, `每档对数`, `层数`, `连接半径`.
- Memory areas and the PFC read `网.输出层数量` of their sources, so they follow
  automatically; there is no second place to edit.
- PFC tuning (layers/radius/diffusion width/convergence ratio/thresholds) is also centralized
  at the top of `前额叶区_prefrontal.py`.
- Decay parameters live at the top of `权重连接管理_weight_manager.py`.

---

## 5. Running and self-checks

```powershell
# inside the repository, using the bundled virtual environment
.\.venv\Scripts\python.exe -X utf8 视觉前处理_visual_preprocess.py
.\.venv\Scripts\python.exe -X utf8 听觉前处理_auditory_preprocess.py
.\.venv\Scripts\python.exe -X utf8 运动输出区_motor_output.py
.\.venv\Scripts\python.exe -X utf8 前额叶区_prefrontal.py
.\.venv\Scripts\python.exe -X utf8 闭环流程_closed_loop.py
```

The visual closed-loop simulation (2D physics body) is documented in `闭环仿真_closed_loop_sim/README.md`:

```powershell
Set-Location -LiteralPath '闭环仿真_closed_loop_sim'
& '..\.venv\Scripts\python.exe' -X utf8 viewer.py
& '..\.venv\Scripts\python.exe' -X utf8 evaluate.py --soak-frames 10000
```

---

## 6. Known limits (honest statement)

- Anchor-free free association drifts (like mind-wandering); replay along a real closed loop
  does not drift.
- Time-memory connections grow over time (~375k after 10k frames) while the frame budget stays
  real-time (~20 ms/frame today); long runs rely on the decay mechanism. Larger scales need
  real hardware profiling.
- The closed-loop simulation is a 2D propelling body for now; bipedal movement, natural
  language and full AGI are out of scope. This project validates the architecture, not those
  end results.