# An AGI Architecture That Learns in Real Time

**Relational Neuron Brain** — a pure-connectionist, brain-inspired model for continuous,
online cognition. It is built from paired spiking neurons, sparse local circuits, a shared
hippocampal time ring, Hebbian memory areas and a prefrontal "thinking hub", all running in a
closed perception→thought→action loop.

> Chinese README: [README.md](README.md) · Chinese architecture notes: [架构说明.md](架构说明.md)
> · English architecture notes: [ARCHITECTURE.md](ARCHITECTURE.md) · Closed-loop simulation:
> [闭环仿真_closed_loop_sim/README.md](闭环仿真_closed_loop_sim/README.md)

**License:** [MIT](LICENSE) · **Status:** active research prototype · **Runtime:** Python 3 +
NumPy on CPU (64-bit Windows tested) · **No external ML framework required.**

---

## Why this project exists

Most neural-network research treats learning as an offline training phase on a frozen dataset.
A brain does not work that way:

- it learns **while it lives** — every frame changes synapses for the next frame;
- memory is not a snapshot store — it is **connection weights** grown by firing order;
- thinking and recall are **spikes travelling along existing connections**, not queries;
- perception (outside-in) and motor output (inside-out) run in opposite directions and need
  **reciprocal wiring** — exactly what the brain has.

This repository tries to build that machinery from first principles, in a small but complete
form, so the architecture can be studied, ablated and extended one piece at a time.

## What is inside

| Idea | Implementation |
| --- | --- |
| Winner-take-all features | every feature is a **positive/negative pair**; exactly one fires per pair (see `特征神经元池`) |
| Signal processing, not learned | **sparse ring networks** with birth-fixed random weights and thresholds; each layer has the width of its input pool; never Hebbian |
| Strength coding | **thermometer codes**: a channel is N pairs = N+1 levels (brightness / loudness / muscle force) |
| Time | one shared **hippocampal time ring**: 1,728,000 neurons, 0.05 s each, two per 0.1 s frame, one lap = 24 h |
| Memory | three **temporal-Hebbian memory areas** (visual/auditory/motor) with bidirectional feature↔time tables and automatic decay of unused connections |
| Thinking | **prefrontal hub**: multi-stream identity + convergence cells, association area with dynamic global inhibition, focus gate (inhibition vs. disinhibition), innate mirror return lines back to the sensory areas |
| Movement | **motor reverse-Hebbian network**: commands are learned to muscles once (monotonic, never forgotten) plus an innate mirror baseline for novel commands |
| Growth control | shared **weight connection manager**: frequent connections survive, rare ones fade below a threshold and are removed |

High-level data flow:

```text
World ──▶ Visual preprocess ──┐
World ──▶ Auditory preprocess ─┤──▶ Prefrontal hub ──▶ Motor output ──▶ Muscles ──▶ World
                                │       │    ▲                              ▲
           Memory areas ◀───────┘       │    └── innate mirror return lines ─┘
                 ▲                      │
                 └── shared hippocampal time ring (0.1 s/frame, 24 h lap) ──┘
```

## Core ideas in detail

Below are the eight mechanisms this architecture relies on, each written as **what it is / how it works / what problem it solves**. For a deeper dive, see [ARCHITECTURE.md](ARCHITECTURE.md) and the Chinese [架构说明.md](架构说明.md).

1. **Paired feature neurons** — Every feature is represented by **one positive and one negative neuron that compete**, and **exactly one of the pair fires** at any moment. Incoming signals are first added into the pool's `信号` (signal) array; when `更新所有对()` ("update all pairs") runs, the positive neuron fires if its signal ≥ the negative neuron's, otherwise the negative one fires, and the buffer is then cleared, ready for the next frame. Competition alone therefore guarantees "one fires per pair, roughly half of all pairs active overall" — no external rule is needed. Splitting one feature across `每档对数` pairs turns it into a 0–N "thermometer code", which can encode graded strength such as brightness, loudness or muscle force (implementation: `特征神经元池` in each sensory/motor module).

2. **Sparse local connection networks** — Signal-processing networks keep the same width as their input pool on every layer. Each neuron connects only to a **ring-shaped neighborhood** of the layer below (default ±10, wrapped around so the two ends are neighbors). Weights are drawn randomly once at birth from `[0, 权重范围)` and are then **never changed**; a neuron fires when its weighted input sum ≥ threshold. These networks are "signal-processing circuits" and are deliberately **never Hebbian**, which keeps them fast, stable and predictable. Layer count, ring radius, weight range and initial threshold are set once in the "全局创建" (global setup) block at the top of each file — change a number there and every downstream width follows automatically.

3. **Temporal Hebbian learning** — the timing-sensitive version of Hebb's rule: when a neuron fires in the **previous frame** and another fires in the **next frame**, the connection from the earlier neuron to the later one is strengthened once (direction: previous frame → next frame). Neurons that fire together in the *same* frame are not learned, so simultaneity is never mistaken for sequence. A repeatedly observed order grows into a **replayable memory chain**: after frame A, the A→B connection makes "B is next" easier to fire. The three memory areas and the prefrontal association area all grow their connections this way.

4. **Hippocampal time ring** — One timeline is shared across the whole system: **1,728,000 time neurons** wired head-to-tail into a ring (like beads on a bracelet), each occupying 0.05 s; one external frame = 0.1 s = **two consecutive time neurons light up**; one full lap takes exactly 24 hours. Making it a ring recycles time slots instead of adding new neurons forever, and bridging each frame with two consecutive time neurons ensures no frame is skipped or misplaced. Memory events attach to the time neuron active "then", so recall can replay them in time order.

5. **Inhibition / disinhibition neurons (dynamic inhibition)** — The prefrontal hub contains ordinary thought neurons, **inhibitory neurons** that suppress nearby normal neurons, and **disinhibitory neurons** that suppress the inhibitors (i.e. "release the brake"). The rule is dynamic: when too many thoughts light up at once, global inhibition is strengthened on the spot so only the strongest-signal batch remains; when too few light up, inhibition is relaxed so weakly-supported cells can also fire. The target band tracks the typical thought-cluster size and expands/contracts with it (about 0.5–1.3×), keeping thinking **always present but neither crowded nor silent** — it neither lights thousands of cells at once nor leaves a genuine idea unable to wake.

6. **Innate reciprocal return lines (mirror network)** — Perception runs outside-in, while imagination and recall run inside-out — the opposite direction. To cover that, the prefrontal cortex is given **fixed mirror return paths** back to every sensory area at birth: prefrontal output is traced backward, layer by layer, along the forward connections, and `返回保留比例` (default 0.3) decides how much of it is kept. These lines **never learn and never forget**, so "thinking of X" can reactivate the very neurons that would fire when X is seen or heard — an innate reciprocal wiring.

7. **Motor reverse network (reverse-Hebbian)** — Movement sends commands outward ("command code → muscles"), and at birth no parameters exist for that direction. The model therefore first **thrashes randomly, like a baby**: each frame it watches which command-code neurons fired and which muscle features were active at that moment, then adds 1 to that "command → muscle" connection in the reverse table; with `一次学会=True` ("learn once"), a connection saturates at 1 the first time it is seen. Learned routes are **kept monotonically and never forgotten**; repeating a known action is gated so no new connections appear; a genuinely novel command code falls back to the **innate mirror baseline** for an approximate muscle target.

8. **Weight connection manager** — Hebbian learning only ever *adds*; without management, memory connections would grow without bound. When the `连接表` (connection table) reaches the `警戒线` (default 100,000 entries), the manager periodically decays the whole table: every weight is multiplied by `衰减率` (0.9), and connections that fall below the `消失下限` (0.05) are removed. Frequently used connections are decayed too, but every use adds them back (reinforcement 1), so they settle near an equilibrium; rarely used connections decay again and again until they hit zero and disappear — memory prunes itself, and the total connection count stays bounded in the long run.

## Repository layout

| Path | Purpose |
| --- | --- |
| `视觉前处理_visual_preprocess.py` / `听觉前处理_auditory_preprocess.py` | visual / auditory preprocessing networks |
| `运动输出区_motor_output.py` | muscle thermometer coding, forward net, reverse-Hebbian net, mirror baseline |
| `视觉记忆区_visual_memory.py` / `听觉记忆区_auditory_memory.py` / `运动记忆区_motor_memory.py` | temporal-Hebbian memory per channel |
| `海马体时间区_hippocampal_time.py` | the shared 24-hour time ring |
| `前额叶区_prefrontal.py` | PFC: network, association, dynamic inhibition, focus gate, reciprocal lines |
| `权重连接管理_weight_manager.py` | generic growth + periodic decay for plastic connections |
| `闭环流程_closed_loop.py` | integration harness running the whole chain as one closed loop |
| `闭环仿真_closed_loop_sim/` | runnable 2D physics-body closed loop (its own README, evaluation suite) |
| `认知实验_cognition_experiments/` `色标导航_color_beacon_navigation/` `原架构闭环_legacy_closed_loop/` `虚拟身体_virtual_body/` `运动实验_motor_experiments/` | earlier-phase experiments |
| `对外联络_outreach/` | outreach kit: one-page brief, email/community templates, and pre-send checklists |
| `架构说明.md` / `ARCHITECTURE.md` | full architecture notes (Chinese / English) |

## Get started

```powershell
# self-checks of the core modules (identifiers are Chinese on purpose)
.\.venv\Scripts\python.exe -X utf8 视觉前处理_visual_preprocess.py
.\.venv\Scripts\python.exe -X utf8 听觉前处理_auditory_preprocess.py
.\.venv\Scripts\python.exe -X utf8 运动输出区_motor_output.py
.\.venv\Scripts\python.exe -X utf8 前额叶区_prefrontal.py
.\.venv\Scripts\python.exe -X utf8 闭环流程_closed_loop.py
```

The **closed-loop simulation** is the most convincing demo: a small body in a 2D world sees,
hears, feels pain, acts and learns online, in one window.

```powershell
Set-Location -LiteralPath '闭环仿真_closed_loop_sim'
& '..\.venv\Scripts\python.exe' -X utf8 viewer.py                 # interactive window
& '..\.venv\Scripts\python.exe' -X utf8 viewer.py --smoke-test    # headless smoke test
& '..\.venv\Scripts\python.exe' -X utf8 evaluate.py --soak-frames 10000
```

The evaluation suite runs **22 independent checks**: collision learning transfers to unseen
walls, arbitrary sound→action mappings are learned and causally depend on the PFC and the
temporal memory, two independently-trained branches can be **fused** into one brain without
losing either skill, checkpoints resume deterministically, an innate reflex area verifiably withdraws on touch and cries on pain with zero learned
connections, lesioning the motor reverse connections stops all movement, and **think-before-act**
freeze rehearsal appears in the closed loop: facing learned danger the brain shuts the action gate
while the PFC advances internally (no movement, no collision, no experience write), then resumes
and turns around the obstacle once the rehearsal resolves. A 10,000-frame soak (~1,000 simulated
seconds) runs at roughly 16 ms/frame average and never exceeds 100 ms/frame on the development machine.

## Where numbers live (edit once, everything follows)

Each module centralizes its size at the top:

- `视觉前处理_visual_preprocess.py`: mosaic rows/cols, channels, levels per channel, layer count, ring radius
  (defaults: 24×16 mosaic, RGB, 10 levels ⇒ 11,520 feature pairs / 23,040 neurons).
- `听觉前处理_auditory_preprocess.py`: frequency bands, levels (defaults: mono, 1,500 bands × 10 levels ⇒ 15,000
  pairs / 30,000 neurons).
- `运动输出区_motor_output.py`: muscle count, levels (defaults: 200 muscles × 10 levels ⇒ 2,000 pairs /
  4,000 neurons).
- Memory areas and the PFC read `网.输出层数量` from their sources, so they resize
  **automatically** — there is no second place to edit.

Read [架构说明.md](架构说明.md) / [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## What is verified — and what is honestly still open

Verified in the closed loop: perception → PFC → action → feedback works end-to-end; online
learning measurably changes behaviour; the PFC has a causal path to decisions; memory fusion
across branches works; movement depends on the learned motor connections.

Open problems we are actively circling (great places to contribute):

1. **Long-run memory growth** — time-memory connections still grow over time (~375k after
   10k frames). The decay manager bounds it, but the steady-state behaviour needs longer
   soak tests and smarter forgetting policies.
2. **Anchor-free association drift** — free recall drifts like mind-wandering; replay along a
   real closed loop does not. Controlling the drift (attention, gating) is an open question.
3. **Scaling** — everything is CPU/NumPy today. Profiling at 10×–100× scale and GPU or sparse
   backends are unexplored.
4. **Conflicting experiences & consolidation** — two branches with complementary memories fuse
   cleanly; conflicting teachings and sleep-like consolidation are not yet studied.
5. **Richer embodiment** — the body is a 2D propelling agent. Sensors (touch already exists in
   the sim), a first-person 3D world and more actuators are next steps.

## How to contribute

Anyone is welcome. The fastest ways to help:

- **Read and critique.** Open an issue about anything that looks wrong — naming, maths,
  biology or engineering.
- **Run the soak tests** on your hardware and report frame-time numbers and memory growth.
- **Pick an open problem above**, design a small ablation, and add it under a new experiment
  folder (see `认知实验_cognition_experiments/` for the house style: state your changes vs. the original rules,
  measure causal effects, never overclaim).
- **Port or polish docs** — English docs exist but Chinese-first identifiers still need
  friendly glossaries and diagrams.

House rules observed so far (please keep them):

1. Edit **one module at a time** so per-area ablations stay clean.
2. Fixed signal-processing networks must stay fixed; Hebbian learning only touches the plastic
   tables listed in `权重连接管理_weight_manager.py`.
3. Prefer pure neuron/connection mechanisms over lookup tables or code shortcuts — that is the
   whole point of the architecture.
4. Keep dependencies minimal (NumPy only for the core; Tkinter for the sim window).
5. Add or extend a self-check in the file you touch and in `闭环仿真_closed_loop_sim/evaluate.py` when the
   closed loop is affected.

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 李秩文 (Li Zhiwen). Use it, learn from it,
build on it; we only ask that you keep the attribution.
## Contact

- Email: rivenlee94@gmail.com
- Phone / WeChat: 17798645729
