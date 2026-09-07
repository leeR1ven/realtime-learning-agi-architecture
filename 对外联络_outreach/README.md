# 对外联络与宣传 Outreach Kit

> 目的：用**专业、简洁**的方式，让研究界与 AI 公司了解“一种可实时学习的 AGI 架构”。
> 原则：不夸大、不冒充、不群发垃圾邮件；给每个对象一句个性化理由；全部源码与失败记录公开，供复现与改进。

---

## 0. 发送前必做（约 10 分钟）

- [ ] **先提交并推送 22/22 基线**：当前本地比 GitHub 领先一个版本（前额叶因果通路修复 + “先想后动”冻结推演验收）。命令在仓库根目录执行：
  `git add -A && git commit -m "先想后动冻结推演验收 22/22 + 对外文档本地路径改相对" && git push origin main`
- [ ] 仓库 Settings → General 里添加 **topics**（建议：`agi`、`brain-inspired`、`cognitive-architecture`、`hebbian-learning`、`continual-learning`、`embodied-ai`、`closed-loop-simulation`）。
- [ ] 打开 **Discussions**（Settings → General → Discussions），方便别人提问交流。
- [ ] 仓库描述建议（可直接粘贴，两行）：
  `纯神经元连接的仿脑持续认知模型原型，不依赖梯度/查表捷径；成对特征神经元、海马时间环、时序赫布记忆、前额叶联想与镜像返回、2D 身体闭环仿真全部开源。`
  `Pure-connectionist brain-inspired cognition prototype (no gradient/shortcuts): paired neurons, hippocampal time ring, temporal Hebbian memory, PFC association + mirror return, open 2D closed-loop body simulation.`
- [ ] 文档中的本地 `F:\一种AGI架构` 绝对路径已改为仓库相对路径（本轮已完成）；`.venv` 未入库。
- ⚠️ 不要把邮箱密码或 GitHub 令牌交给任何人（包括 AI）。**发送必须用你自己的账号操作。**

---

## 1. 一页简介（可直接复制粘贴）

**中文**
“一种可实时学习的 AGI 架构”是一个纯神经元连接的仿脑持续认知模型原型。它试图只靠神经元之间的连接与放电完成感知、记忆、联想、运动控制与在线学习：成对特征神经元用竞争保证‘每对恰一激活’；稀疏局部网络固定布线、只做信号处理；三个记忆区与前额叶用‘上一帧→下一帧’的时序赫布规则长连接；海马体 172.8 万个时间神经元绕成 24 小时一圈的时间环，让记忆按时间顺序回放；抑制/去抑制神经元把前额叶思考稳定在‘有念头但不多不少’；天生镜像返回线让大脑能从内部反向激活感觉区；反向赫布网络让‘命令→肌肉’一次学会、单调保留。
可运行证据是一个 2D 物理身体闭环：撞墙后学会换场景规避、四音调可任意配对到动作并因果依赖前额叶/记忆、两个分支记忆可无损融合、存档可断点恢复、且出现了‘先想后动’——危险前冻结推演数帧（不动、不撞），回放到安全后继后再绕行。评估 `evaluate.py` 22/22 通过，1000 帧压测 p95 约 7 ms/帧。全部代码 Python + NumPy、CPU 可跑、MIT 许可。
诚实边界：这是验证机制的小型具身原型，**不是 AGI，也尚未证明双足、语言与开放规划**；这些实验只证明了设计里的若干机制可以端到端运转。

**English**
Realtime-Learning AGI Architecture is a pure-connectionist, brain-inspired prototype that tries to do perception, memory, association, motor control and online learning using only connections and firing between neurons — no gradient training, no lookup tables, no shortcuts. Paired feature neurons guarantee one-winner-per-pair by competition; fixed sparse local networks handle signals only; three memory areas plus the PFC grow connections with a frame-to-frame temporal Hebbian rule; a hippocampal ring of 1.728M time neurons (one turn per 24 h) lets memories replay in time order; inhibition/disinhibition pools keep PFC thought sparse but alive; innate mirror return lines let the brain re-activate sensory areas from the inside; a reverse-Hebbian network learns command-to-muscle mappings in one exposure and never forgets them.
Runnable evidence is a closed-loop 2D body: after colliding once it generalizes avoidance to unseen walls; four tones can be paired to arbitrary actions and causally depend on PFC/memory; two trained branches fuse losslessly; checkpoints resume deterministically; and the model shows think-before-act — freezing a few frames in front of learned danger (no movement, no collision), then detouring once internal replay reaches a safe successor. `evaluate.py` passes 22/22; a 1,000-frame soak runs at ~7 ms/frame p95 on CPU. All code is Python + NumPy, MIT-licensed.
Honest scope: a small embodied prototype that validates mechanisms — not AGI, and not evidence of bipedal movement, language, or open-ended planning.

---

## 2. 邮件模板

### 2a) 英文 —— 研究实验室 / 教授 / 开源组织

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

### 2b) 中文 —— 国内研究团队 / 机构

> **主题：一个纯神经元连接、可实时学习的仿脑认知架构（小但可完整复现）**
>
> [老师/团队]：
> 我开源了一个小但完整的仿脑认知架构原型，在 2D 物理身体闭环里端到端运行，全程只用神经元连接与放电——不用反向传播、不用查表。它把成对特征竞争、24 小时海马时间环、逐帧时序赫布学习（三个记忆区 + 前额叶）、抑制/去抑制动态门控、天生镜像返回线和“命令→肌肉”反向赫布层拼成一套持续学习系统。
> 验收完全可复现：任意“音调→动作”关联能被学会并因果依赖前额叶/记忆连接；撞墙经验能迁移到未见过的墙；两个分别训练的脑可以无损融合；存档可断点续训；模型还会“先想后动”（危险前冻结推演、内部回放安全路径后再绕行）。`evaluate.py` 22/22 通过，CPU 上约 7 ms/帧（p95）。
> 仓库（中英双语、MIT）：https://github.com/leeR1ven/realtime-learning-agi-architecture
> 非常希望听到您的意见；您认为哪些实验能让结论更扎实，我可以照做并公开结果。

### 2c) Show HN / 社区帖速写（3 行版）

> Show HN: A pure-connectionist AGI-brain prototype that learns, remembers and thinks without backprop
>
> Pair-competitive feature neurons + 24h hippocampal time ring + frame-to-frame temporal Hebbian learning + dynamic inhibition + mirror return lines + reverse-Hebbian motor. Runs a 2D body end-to-end on CPU: learns tone→action mappings, transfers wall-avoidance to unseen walls, fuses two trained brains, and thinks before acting (freezes, replays internally, then detours). 22/22 checks, MIT, bilingual docs: https://github.com/leeR1ven/realtime-learning-agi-architecture

---

## 3. 渠道与目标清单

### 第一优先：公开、免费、回复率高（建议按此顺序做）
1. **arXiv 预印本**（https://arxiv.org 提交，分类 cs.AI / cs.NE）：把 README 核心要点扩成一页纸报告，附仓库链接与复现命令。
2. **Show HN**：https://news.ycombinator.com/show（用 2c 模板）。
3. **Reddit**：r/MachineLearning（用 Project 标签）、r/agi（用 2c 模板）。
4. **X / 知乎 / 微信公众号**：发线程或专栏，附 2c 式摘要与仓库。
5. **GitHub 本身**：加 topics、开 Discussions、把仓库置顶；README 已中英双语。

### 第二优先：相关学术/开源社区（直接在其论坛或 GitHub Discussions 讨论）
- Numenta / Thousand Brains：https://www.numenta.com 、https://thousandbrains.org
- OpenCog（AGI 开源社区）：https://opencog.org
- Neuromatch（计算神经科学社区）：https://neuromatch.io
- 各实验室关于“预测编码 / 联想记忆 / 神经形态”方向的论文作者（用其论文公开邮箱，个性化联系）

### 第三优先：AI 公司（按官方公开入口，逐个、少量、个性化）
候选名单（供参考）：OpenAI、Anthropic、Google DeepMind、Meta AI、xAI、Mistral AI、Hugging Face、Cohere；国内：智谱、月之暗面、深度求索（DeepSeek）、MiniMax、阶跃星辰。
- **不要群发 `info@`**；建议走官方网站的“研究 / 招聘 / 合作”页面、研究者 LinkedIn 或公开论文邮箱，一封只谈一个具体点（例如镜像返回线的路由机制、或“先想后动”与杏仁核样恐惧门控）。
- 这些公司每天收到大量投稿，**回复率极低是正常现象**；能产生价值的往往是“开源贡献 → 被复现 → 被讨论”的长路径。

---

## 4. 对外口径（务必遵守）

- 定位：一个强调**纯神经元连接、可实时赫布学习**的小型具身闭环原型；验证了若干机制；源码、验收脚本、失败记录全部公开。
- **不要说**“AGI 已完成 / 接近 AGI / 比肩大脑”。验收文档与 README 已声明：这不是 AGI，也不是双足/语言/开放规划的证据。
- 数字口径一律以 `闭环仿真_closed_loop_sim/results/evaluation.json` 为准，不引用记忆中的旧数字。
- 被追问时明确区分：哪些机制已验证（22/22 列表）、哪些只是设计假设（镜像路由精度、长期抗遗忘、规模扩展）。

## 5. 节奏建议

1. 分批联系而不是一次性群发（防封、也显得真诚）。
2. 每封信只提一个“你最想被讨论的点”，并给出可复现入口（一条评估命令）。
3. 对方回复后，把有价值的提问与答复沉淀回仓库 README / Issues，形成公开讨论记录。
4. 若有人 fork 或提 issue，优先回应并致谢——公开社区的信任比单次曝光更值钱。

---

*生成日期：2026-09-07。仓库链接与验收数字以实际推送后的 GitHub 页面为准。*