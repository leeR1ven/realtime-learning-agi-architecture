# Outreach Kit (对外联络)

> Goal: help the research community and AI companies understand the "Realtime-Learning AGI Architecture" in a professional, concise way.
> Principles: no hype, no impersonation, no mass unsolicited mail; add one personal reason per recipient; all source and failure records are public for reproduction and improvement.

---

## 0. Before you send (~10 min checklist)

- [ ] **Commit and push the 22/22 baseline first.** Local code is one version ahead of GitHub (PFC causal-path fix + think-before-act freeze-rehearsal acceptance). From the repo root:
  `git add -A && git commit -m "先想后动冻结推演验收 22/22 + 对外文档本地路径改相对" && git push origin main`
- [ ] Add **topics** under Settings → General (suggested): `agi`, `brain-inspired`, `cognitive-architecture`, `hebbian-learning`, `continual-learning`, `embodied-ai`, `closed-loop-simulation`.
- [ ] Enable **Discussions** (Settings → General) so people can ask questions.
- [ ] Suggested two-line repo description:
  `纯神经元连接的仿脑持续认知模型原型，不依赖梯度/查表捷径；成对特征神经元、海马时间环、时序赫布记忆、前额叶联想与镜像返回、2D 身体闭环仿真全部开源。`
  `Pure-connectionist brain-inspired cognition prototype (no gradient/shortcuts): paired neurons, hippocampal time ring, temporal Hebbian memory, PFC association + mirror return, open 2D closed-loop body simulation.`
- [ ] Local absolute paths like `F:\一种AGI架构` in the docs are already converted to repo-relative links (done this round); `.venv` is not committed.
- ⚠️ Never hand your email password or GitHub token to anyone (including AI). **You must send from your own accounts.**

---

## 1. One-page brief (copy/paste ready)

**English**
Realtime-Learning AGI Architecture is a pure-connectionist, brain-inspired prototype that tries to do perception, memory, association, motor control and online learning using only connections and firing between neurons — no gradient training, no lookup tables, no shortcuts. Paired feature neurons guarantee one-winner-per-pair by competition; fixed sparse local networks handle signals only; three memory areas plus the PFC grow connections with a frame-to-frame temporal Hebbian rule; a hippocampal ring of 1.728M time neurons (one turn per 24 h) lets memories replay in time order; inhibition/disinhibition pools keep PFC thought sparse but alive; innate mirror return lines let the brain re-activate sensory areas from the inside; a reverse-Hebbian network learns command-to-muscle mappings in one exposure and never forgets them.
Runnable evidence is a closed-loop 2D body: after colliding once it generalizes avoidance to unseen walls; four tones can be paired to arbitrary actions and causally depend on PFC/memory; two trained branches fuse losslessly; checkpoints resume deterministically; and the model shows think-before-act — freezing a few frames in front of learned danger (no movement, no collision), then detouring once internal replay reaches a safe successor. `evaluate.py` passes 22/22; a 1,000-frame soak runs at ~7 ms/frame p95 on CPU. All code is Python + NumPy, MIT-licensed.
Honest scope: a small embodied prototype that validates mechanisms — not AGI, and not evidence of bipedal movement, language, or open-ended planning.

**中文**
“一种可实时学习的 AGI 架构”是一个纯神经元连接的仿脑持续认知模型原型。它试图只靠神经元之间的连接与放电完成感知、记忆、联想、运动控制与在线学习：成对特征神经元用竞争保证“每对恰一激活”；稀疏局部网络固定布线、只做信号处理；三个记忆区与前额叶用“上一帧→下一帧”的时序赫布规则长连接；海马体 172.8 万个时间神经元绕成 24 小时一圈的时间环，让记忆按时间顺序回放；抑制/去抑制神经元把前额叶思考稳定在“有念头但不多不少”；天生镜像返回线让大脑能从内部反向激活感觉区；反向赫布网络让“命令→肌肉”一次学会、单调保留。
可运行证据是一个 2D 物理身体闭环：撞墙后学会换场景规避、四音调可任意配对到动作并因果依赖前额叶/记忆、两个分支记忆可无损融合、存档可断点恢复、且出现了“先想后动”——危险前冻结推演数帧（不动、不撞），回放到安全后继后再绕行。评估 `evaluate.py` 22/22 通过，1000 帧压测 p95 约 7 ms/帧。全部代码 Python + NumPy、CPU 可跑、MIT 许可。
诚实边界：这是验证机制的小型具身原型，**不是 AGI，也尚未证明双足、语言与开放规划**。

---

## 2. Email templates

### 2a) English — research lab / professor / open-source organization

> **Subject:** A pure-connectionist, continually learning brain architecture — small but fully reproducible prototype
>
> Dear [Name/Team],
>
> I have built and open-sourced a small but complete, brain-inspired cognitive architecture that runs end-to-end in a closed 2D body loop using only neuron connections and firing — no backprop, no lookup tables. It combines paired winner-take-all feature neurons, a 24-hour hippocampal time ring, frame-to-frame temporal Hebbian learning in three memory areas and the PFC, dynamic inhibition/disinhibition for sparse-but-alive thought, innate mirror return lines for inside-out recall, and a reverse-Hebbian motor layer.
>
> The evaluation is fully reproducible: arbitrary tone-to-action associations are learned and causally depend on PFC/memory connections, collision experience transfers to unseen walls, two trained brains fuse losslessly, checkpoints resume deterministically, and the model exhibits think-before-act freeze rehearsal (it stops moving in front of learned danger, replays internally, then detours). `evaluate.py` passes 22/22 checks; ~7 ms/frame p95 on CPU.
>
> Repo (bilingual docs, MIT): https://github.com/leeR1ven/realtime-learning-agi-architecture
>
> I would value your comments, and would be glad to adjust experiments you think would make the claims stronger. Thank you for your time.

### 2b) Chinese — domestic research teams / institutions
> **主题：一个纯神经元连接、可实时学习的仿脑认知架构（小但可完整复现）**
>
> [老师/团队]：我开源了一个小但完整的仿脑认知架构原型，在 2D 物理身体闭环里端到端运行，全程只用神经元连接与放电——不用反向传播、不用查表。它把成对特征竞争、24 小时海马时间环、逐帧时序赫布学习（三个记忆区 + 前额叶）、抑制/去抑制动态门控、天生镜像返回线和“命令→肌肉”反向赫布层拼成一套持续学习系统。
> 验收完全可复现：任意“音调→动作”关联能被学会并因果依赖前额叶/记忆连接；撞墙经验能迁移到未见过的墙；两个分别训练的脑可以无损融合；存档可断点续训；模型还会“先想后动”（危险前冻结推演、内部回放安全路径后再绕行）。`evaluate.py` 22/22 通过，CPU 上约 7 ms/帧（p95）。
> 仓库（中英双语、MIT）：https://github.com/leeR1ven/realtime-learning-agi-architecture
> 非常希望听到您的意见；您认为哪些实验能让结论更扎实，我可以照做并公开结果。

### 2c) Show HN / community post sketch (3 lines)

> Show HN: A pure-connectionist AGI-brain prototype that learns, remembers and thinks without backprop
>
> Pair-competitive feature neurons + 24h hippocampal time ring + frame-to-frame temporal Hebbian learning + dynamic inhibition + mirror return lines + reverse-Hebbian motor. Runs a 2D body end-to-end on CPU: learns tone→action mappings, transfers wall-avoidance to unseen walls, fuses two trained brains, and thinks before acting (freezes, replays internally, then detours). 22/22 checks, MIT, bilingual docs: https://github.com/leeR1ven/realtime-learning-agi-architecture

---

## 3. Channels and target list

### Priority 1 — public, free, high reply rate (do in this order)
1. **arXiv preprint** (submit at https://arxiv.org, categories cs.AI / cs.NE): expand the README core points into a one-page report with the repo link and reproduction commands.
2. **Show HN**: https://news.ycombinator.com/show (use template 2c).
3. **Reddit**: r/MachineLearning (Project flair), r/agi (use template 2c).
4. **X / 知乎 / WeChat**: short thread or column with the 2c abstract and repo link.
5. **GitHub itself**: add topics, enable Discussions, pin the repo; README is already bilingual.

### Priority 2 — related academic / open-source communities (post in their forums or GitHub Discussions)
- Numenta / Thousand Brains: https://www.numenta.com , https://thousandbrains.org
- OpenCog (open-source AGI community): https://opencog.org
- Neuromatch (computational neuroscience community): https://neuromatch.io
- Authors of papers on predictive coding / associative memory / neuromorphic topics (use public paper emails, personalize)

### Priority 3 — AI companies (official public entry points only; few, personalized)
Candidate names: OpenAI, Anthropic, Google DeepMind, Meta AI, xAI, Mistral AI, Hugging Face, Cohere; domestic: Zhipu AI, Moonshot AI, DeepSeek, MiniMax, StepFun.
- **Do not mass-mail `info@`.** Use the official "research / careers / partnerships" pages, researchers on LinkedIn, or public paper emails; one email = one concrete topic (e.g., the mirror return-line routing, or think-before-act with amygdala-like fear gating).
- These companies receive huge volumes of unsolicited mail; a low reply rate is normal. The realistic long path is: open-source contribution → reproduction → discussion.

---

## 4. What to claim (and what not to)

- Positioning: a small embodied closed-loop prototype emphasizing **pure neuron connections and real-time Hebbian learning**; several mechanisms validated; source, acceptance scripts, and failure records are all public.
- **Do not say** "AGI achieved / near-AGI / on par with the brain." README and acceptance docs already state it is not AGI and not evidence of bipedal movement, language, or open-ended planning.
- All numbers must come from `闭环仿真_closed_loop_sim/results/evaluation.json`; do not quote older remembered numbers.
- When asked, clearly separate validated mechanisms (the 22/22 list) from design assumptions (mirror-route precision, long-term anti-forgetting, scaling).

## 5. Cadence advice

1. Contact in batches, not all at once (avoids spam filters and looks sincere).
2. One concrete point per email, plus a reproducible entry point (one evaluation command).
3. When someone replies, fold valuable Q&A back into the repo README / Issues to create a public record.
4. Respond promptly to forks and issues — community trust outlives a single piece of exposure.

---

*Generated: 2026-09-07. Repo link and acceptance numbers should be re-checked against the pushed GitHub page.*