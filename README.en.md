# An AGI Architecture That Learns in Real Time — Relational Neuron Brain

A pure-connectionist, brain-inspired model for continuous cognition.
The goal is a "digital brain" that perceives, remembers, associates, controls a body and learns
using **only connections between neurons and their firing** — no lookup tables, no searches,
no conventional gradient training shortcuts.

Implemented in Python + NumPy on CPU. Source identifiers are Chinese; docs are bilingual.

- Chinese README: [README.md](README.md)
- Full Chinese architecture notes: [架构说明.md](架构说明.md)
- Full English architecture notes: [ARCHITECTURE.md](ARCHITECTURE.md)

## What this project tries to prove

- The brain is not an input->hidden->output black box but many parallel, interconnected,
  continuously running areas in a closed loop;
- Memory is not a snapshot store but **connection weights** that grow with firing order
  over time;
- Thinking and recall are just spikes travelling along existing connections;
- Perception (outside-in) and motor/imagery (inside-out) run in opposite directions and need
  dedicated reciprocal wiring.

## Core ideas

1. **Paired feature neurons** — each feature is a competing positive/negative pair; exactly
   one fires, so "half active" and "one per pair" come for free.
2. **Sparse locally-connected networks** — fixed birth weights, ring neighbourhoods, one
   threshold per neuron; used for signal processing only (never Hebbian).
3. **Temporal Hebbian learning** — earlier firing connects to later firing (previous frame ->
   next frame), letting memory areas grow continuous chains.
4. **Hippocampal time ring** — 1,728,000 time neurons in a ring; 0.05 s per neuron, two per
   0.1 s frame, one full lap = 24 h; events are hung onto the time neurons "when they happened".
5. **Inhibition / disinhibition** — inhibition silences neighbours, disinhibition silences
   inhibition; global inhibition is dynamic so the PFC always keeps a healthy number of active
   thoughts (not too few, not too many).
6. **Innate reciprocal mirror lines** — PFC-to-sensory return paths are wired at birth and
   never change, so "imagining X" re-activates the sensory representation of X without learning.
7. **Motor reverse Hebbian network** — movement is inside-out. Random infant babbling teaches
   "command -> muscle feature" once (monotonic, never forgotten); afterwards a command code
   drives the muscles. Includes an innate mirror baseline for novel commands.
8. **Weight connection management** — rarely-used connections decay away while frequently-used
   ones are maintained, preventing unbounded memory growth.

## Repository layout

| Path | What it is |
| --- | --- |
| `视觉前处理.py` `听觉前处理.py` | Sensory preprocessing networks (external signals -> sparse firing) |
| `运动输出区.py` | Muscle force encoding, forward network, reverse Hebbian network, mirror baseline |
| `视觉记忆区.py` `听觉记忆区.py` `运动记忆区.py` | Temporal Hebbian memory per channel |
| `海马体时间区.py` | Shared time ring (0.1 s/frame, 24 h lap) |
| `前额叶区.py` | PFC: association, dynamic inhibition, focus gate, reciprocal return lines |
| `权重连接管理.py` | Generic growth + decay rule for plastic connections |
| `闭环流程.py` | Integration harness that runs the whole chain in one closed loop |
| `闭环仿真/` | Runnable 2D physics-body closed loop (see its README) |
| `认知实验/` `色标导航/` `原架构闭环/` `虚拟身体/` `运动实验/` | Experiments from earlier phases |
| `实验记录/` | Process records and audits |

## Getting started

```powershell
# self-checks of the core files
.\.venv\Scripts\python.exe -X utf8 视觉前处理.py
.\.venv\Scripts\python.exe -X utf8 听觉前处理.py
.\.venv\Scripts\python.exe -X utf8 运动输出区.py
.\.venv\Scripts\python.exe -X utf8 前额叶区.py

# visual closed loop (opens a window; the body acts and learns on its own)
Set-Location -LiteralPath '闭环仿真'
& '..\.venv\Scripts\python.exe' -X utf8 viewer.py

# headless smoke / full evaluation (16 checks + 10,000-frame soak)
& '..\.venv\Scripts\python.exe' -X utf8 viewer.py --smoke-test
& '..\.venv\Scripts\python.exe' -X utf8 evaluate.py --soak-frames 10000
```

Requirements: Python 3 + NumPy. The included virtual environment (`.venv/`) is not tracked.

## Verified results (summary)

- Closed-loop `evaluate.py`: 16/16 checks pass — collision learning transfers, arbitrary
  sound->action pairing, memory fusion across branches, checkpoint restore determinism, motor
  reverse connections cause movement.
- 10,000-frame soak (~1,000 simulated seconds): ~16 ms/frame average, p95 ≈ 22 ms, max 54 ms,
  no frame over 100 ms — real-time capable on the development machine.
- PFC: replay along a real experience covers the next thought 100%; dynamic inhibition keeps
  activity around the typical thought size.

## Honest limits

- Free association without an anchor drifts (mind-wandering); closed-loop replay does not.
- Time-memory connections grow over time; frame time is still ~20 ms today, and the decay
  mechanism bounds long runs.
- The closed loop currently uses a 2D propelling body: no bipedal walking, natural language,
  or full AGI yet. CPU-scale results do not extrapolate to arbitrary scale.

## Contribute

Contributions and architecture discussions are welcome. The docs and code are in Chinese
first; English docs live alongside. When extending, please keep the habit of editing one
module at a time so per-area ablations stay clean.