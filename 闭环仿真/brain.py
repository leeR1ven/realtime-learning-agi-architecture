"""可审计的小规模闭环；旧神经类的新实例，固定本能与可塑连接分开。

实验感官是五束深度/RGB、接触、本体感觉与四个声音频带。
声音通过PCM进入；这里没有文本解析、路径规划或目标坐标输入。
动作词只是查看神经元时用的标签，不代表已经获得自然语言。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from 前额叶区 import 前额叶联想区
from 视觉记忆区 import 记忆区
from 海马体时间区 import 时间环
from neural import PairedEncoder, MotorBank
from checkpoint import save_checkpoint, load_checkpoint

ACTION_NAMES = ('静息', '前进', '后退', '左转', '右转')
FREQUENCIES = np.array([220., 440., 660., 880.])
SAMPLE_RATE = 8000
DT = 0.1


def tone(index, amplitude=0.7, samples=800):
    return amplitude * np.sin(2 * np.pi * FREQUENCIES[index] * np.arange(samples) / SAMPLE_RATE)


def sound_features(waveform):
    x = np.asarray(waveform, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('声音必须是有限值的一维8kHz PCM数组')
    if len(x) == 0:
        return np.zeros(4)
    phases = 2 * np.pi * FREQUENCIES[:, None] * np.arange(len(x)) / SAMPLE_RATE
    amplitude = 2 * np.abs(np.exp(-1j * phases) @ x) / len(x)
    return np.clip(amplitude / 0.5, 0, 1)


class SparseTimeMemory(记忆区):
    """仍用旧类的两条时序赫布规则，沿活动连接传播到时间神经元。

回忆允许多个时间神经元同时放电；不用全时间环稠密扫描。
存档存的是两个方向的连接。next提供每个独立时间环的固定后继。
"""
    def __init__(self, n):
        super().__init__(n)
        self.next = {}
        self.strength = {}
        self.ticks = {}
        self.last_active_events = 0

    def remember(self, old, new, pattern, tick):
        self.next[old] = new
        self.next.setdefault(new, -1)
        self.strength.setdefault(old, 1.)
        self.strength.setdefault(new, 1.)
        self.ticks[old] = tick
        self.ticks[new] = tick + 1
        super().学习(old, new, pattern)

    def recall_signal(self, pattern):
        currents = {}
        for source in np.flatnonzero(pattern):
            for event, weight in self.特征到时间.查(source).items():
                currents[event] = currents.get(event, 0.) + weight * self.strength.get(event, 1.)
        out = np.zeros(self.总数)
        if not currents:
            self.last_active_events = 0
            return out
        # 全部有关联的连接都参与。门槛控制活动，不删除弱连接。
        threshold = 0.9 * max(currents.values())
        self.last_active_events = 0
        for event, current in currents.items():
            if current >= threshold:
                self.last_active_events += 1
                for target, weight in self.时间到特征.查(event).items():
                    out[target] += current * weight
        peak = out.max(initial=0)
        return out / peak if peak else out

    def retire(self, event):
        # 时间环转满后复用旧槽：移除该时间细胞旧连接，防跨日混合。
        for table in (self.特征到时间,):
            for row in table.表.values():
                if event in row:
                    del row[event]
                    table.连接条数 -= 1
        row = self.时间到特征.表.pop(event, {})
        self.时间到特征.连接条数 -= len(row)
        self.next[event] = -1
        self.strength[event] = 1.


class Brain:
    def __init__(self, *, seed=2026):
        self.seed = int(seed)
        # 5深度+15RGB+2速度+角速度+痛觉+4频带+5运动副本。
        self.encoder = PairedEncoder(33)
        self.motor = MotorBank()
        self.n = self.encoder.n
        self.pfc = 前额叶联想区(self.n, 每段=20, 门槛=.5, 目标下限=4,
                                目标上限=48, 强度初值=.5, 强度上限=100.)
        self.pfc.联想.强化量 = .002
        self.pfc.去抑.强化量 = .002
        self.memory = SparseTimeMemory(self.n)
        self.clock = 时间环()
        self.frame = 0
        self.memory_base = 0
        self.previous = np.zeros(self.n, dtype=bool)
        self.last_action = 0
        self.auditory_trace = np.zeros(4)
        self.proximity_trace = np.zeros(5)
        self.pain_trace = 0.
        self.color_adaptation = np.zeros(3)
        self.audio_weights = np.zeros((4, 5))
        self.echo_weights = np.zeros((4, 4))
        self.danger_weights = np.zeros(5)
        self.echo_cooldown = 0
        self.last_info = {}
        # 五个运动簇接收的固定突触：[自发、左显著、右显著、前危险、右危险、左危险、疼痛]。
        self.reflex_weights = np.array([[.08, 0, 0, 0, 0, 0, 0],
            [.65, 0, 0, -2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2.8],
            [.035, .9, 0, 3.2, .4, 0, 0], [.025, 0, .9, 3.2, 0, .4, 0]])
        # 转向簇短时互抑，防障碍两侧信号相近时每帧反向、原地卡住。
        self.motor_recurrence = np.zeros((5, 5))
        self.motor_recurrence[3, 3] = self.motor_recurrence[4, 4] = .18
        self.motor_recurrence[3, 4] = self.motor_recurrence[4, 3] = -.18
        self.salience_inhibition = 4.
        self.adaptation_strength = .97
        self.birth_hash = self._birth_hash()

    def _fixed(self):
        arrays = {}
        for prefix, component in (('encoder', self.encoder), ('motor', self.motor)):
            for key, value in component.fixed_arrays().items():
                arrays['fixed_' + prefix + '_' + key] = np.asarray(value)
        arrays['fixed_reflex'] = self.reflex_weights
        arrays['fixed_motor_recurrence'] = self.motor_recurrence
        arrays['fixed_salience_inhibition'] = np.array([self.salience_inhibition])
        arrays['fixed_color_adaptation'] = np.array([.02, self.adaptation_strength])
        # 五次出生示范长成的运动反向连接，在日常生活中冻结为本能。
        arrays['fixed_motor_reverse'] = self.motor.reverse.矩阵
        return arrays

    def _birth_hash(self):
        h = hashlib.sha256(b'associative-embodied-v1-tone-depth-rgb')
        for name, value in sorted(self._fixed().items()):
            h.update(name.encode()); h.update(value.tobytes())
        return h.hexdigest()

    def reset_activity(self):
        """实验换场景时清除瞬态；所有已学习突触、时间与经历保留。"""
        self.previous[:] = False
        self.memory.激活[:] = False
        self.pfc.念头[:] = False
        self.pfc.强度 = .5
        self.auditory_trace[:] = 0
        self.proximity_trace[:] = 0
        self.pain_trace = 0.
        self.color_adaptation[:] = 0
        self.last_action = 0
        self.echo_cooldown = 0

    def _motor_currents(self, activity):
        blocks = np.asarray(activity).reshape(33, 20)[28:33]
        # 肌肉意图1的特征兴奋，意图0的相反特征抑制；不是符号检索。
        currents = np.maximum(0, blocks[:, 18] - blocks[:, 1])
        return currents / max(1e-8, float(currents.max(initial=0)))

    def step(self, observation, waveform=(), *, stimulation=None, pleasant=0.,
             mouth_stimulation=None, plasticity=True, memory_enabled=True, pfc_enabled=True):
        """唯一逐帧循环。stimulation是外界教学时的运动神经刺激。

        平时和教学同一路径；pleasant为外界强化信号。三个布尔开关只用于
        独立消融验证，交互运行全部为True，绝不切换到不学习的推理模式。
        """
        audio = sound_features(waveform)
        distances = np.asarray(observation['ray_distances'], float)
        colors = np.asarray(observation['ray_colors'], float)
        if distances.shape != (5,) or colors.shape != (5, 3):
            raise ValueError('当前神经元布局需要五束深度/RGB感官')
        proximity = np.exp(-distances / .65)
        velocity = np.asarray(observation['velocity'])
        speed = np.linalg.norm(velocity)
        pain = float(observation['pain'])
        values = np.concatenate((proximity, colors.ravel(), [min(speed / 3, 1),
                  min(abs(observation['angular_velocity']) / 4, 1),
                  np.clip((observation['angular_velocity'] + 4) / 8, 0, 1), pain],
                  audio, np.eye(5)[self.last_action]))
        sensory = self.encoder.encode(values)
        decoded = self.encoder.decode(sensory)
        proximity, colors, pain, audio = decoded[:5], decoded[5:20].reshape(5, 3), decoded[23], decoded[24:28]
        # 听觉正特征经固定注意通路给时间区与前额叶发信号。
        cue = np.zeros(self.n, dtype=bool)
        cue[480:560:2] = sensory[480:560:2]
        recall_input = cue if cue.any() else sensory
        recalled = self.memory.recall_signal(recall_input) if memory_enabled else np.zeros(self.n)
        previous = self.previous.copy()
        thought = self.pfc.微步(previous, sensory.astype(float) * 3.5 + .12 * recalled)
        # 抑制只限制活动。阈值平票由出生时固定的微小兴奋性差异打破。
        if thought.sum() > 48:
            current = self.pfc.驱动(previous) + sensory * 3.5 + .12 * recalled
            current += np.arange(self.n) * 1e-10
            threshold = np.partition(current, -48)[-48]
            thought = current >= threshold
            self.pfc.念头 = thought
        self.previous = thought.copy()

        self.auditory_trace = np.maximum(audio, self.auditory_trace * .35)
        self.proximity_trace = np.maximum(proximity, self.proximity_trace * .88)
        self.pain_trace = max(pain, self.pain_trace * .7)
        # 固定本能投影：自发前进、鲜艳刺激定向、触痛退缩。
        # 持续颜色刺激引起抑制性适应，避免围着同一色标永久犹豫。
        self.color_adaptation += .02 * (colors.max(axis=0) - self.color_adaptation)
        adapted = np.maximum(0, colors - self.adaptation_strength * self.color_adaptation)
        salience = np.maximum(0, adapted.max(axis=1) - adapted.min(axis=1))
        turn = np.dot(salience, np.array([-1., -.6, 0., .6, 1.]))
        risk = proximity * self.danger_weights
        front_risk = float(risk[1:4].max())
        # 抑制运动/定向的权重由疼痛赫布联结提供，不预置避墙规则。
        reflex_inputs = np.array([1, max(turn - self.salience_inhibition * front_risk, 0),
                                  max(-turn - self.salience_inhibition * front_risk, 0), front_risk,
                                  risk[:2].sum(), risk[3:].sum(), self.pain_trace])
        scores = self.reflex_weights @ reflex_inputs + self.motor_recurrence @ np.eye(5)[self.last_action]
        # 前额听觉->运动可塑投射；声音未知时无预置动作映射。
        temporal = self._motor_currents(self.pfc.驱动(cue)) if pfc_enabled and cue.any() else np.zeros(5)
        episodic = self._motor_currents(recalled) if memory_enabled and cue.any() else np.zeros(5)
        learned = (audio @ self.audio_weights if pfc_enabled else np.zeros(5)) + .25 * temporal + .25 * episodic
        confidence = float(learned.max(initial=0))
        scores = scores * (1 - min(1., confidence * 2.)) + 3. * learned
        if stimulation is not None:
            stim = np.asarray(stimulation, dtype=float)
            if stim.shape != (5,) or not np.isfinite(stim).all() or (stim < 0).any():
                raise ValueError('运动神经刺激必须是5维非负向量')
            scores += 10 * stim
        # 运动竞争：一个动作簇胜出；内部记忆允许多个簇共存。
        action = int(np.argmax(scores))
        motor_code = np.eye(5)[action]
        muscles = self.motor.decode(action)
        mouth = audio @ self.echo_weights if pfc_enabled else np.zeros(4)
        if mouth_stimulation is not None:
            mouth += 10 * np.asarray(mouth_stimulation)
        mouth_post = (mouth >= max(.35, float(mouth.max()) * .9)).astype(float)
        emitted = np.zeros(800)
        self.echo_cooldown = max(0, self.echo_cooldown - 1)
        if mouth_post.any() and self.echo_cooldown == 0:
            emitted = sum(tone(i, .18) for i in np.flatnonzero(mouth_post))
            self.echo_cooldown = 4

        if plasticity:
            # 三因子局部塑性：先前感觉活动×当前运动活动×强化/疼痛。
            gain = np.clip(pleasant, 0, 1) * .14
            self.audio_weights += gain * self.auditory_trace[:, None] * motor_code * (1 - self.audio_weights)
            self.echo_weights += gain * self.auditory_trace[:, None] * mouth_post * (1 - self.echo_weights)
            self.danger_weights += .5 * pain * self.proximity_trace * (1 - self.danger_weights)
            # 已学连接由当前使用强化；未用连接缓慢衰减。
            self.danger_weights *= .999995
            self.audio_weights *= .999999
            self.echo_weights *= .999999
            # 固定1:1传入前额叶的感官/运动副本决定经验时序。
            # 内部回声仍可驱动思考，但不充当外界已经发生的教学证据。
            self.pfc.学习(self.memory.激活.copy(), sensory)
            for _ in range(self.clock.每帧步数):
                old, new = self.clock.走一步()
                old += self.memory_base; new += self.memory_base
                if self.frame * 2 >= self.clock.时间神经元数量:
                    self.memory.retire(new)
                self.memory.remember(old, new, sensory, self.frame * 2)
        self.frame += 1
        self.last_action = action
        self.last_info = {'frame': self.frame, 'action': action, 'action_name': ACTION_NAMES[action],
            'muscles': muscles.tolist(), 'scores': scores.tolist(), 'heard': audio.tolist(),
            'emitted': np.flatnonzero(mouth_post).tolist() if emitted.any() else [],
            'pain': pain, 'risk': float(front_risk), 'sensory_active': int(sensory.sum()),
            'thought_active': int(thought.sum()), 'recall_events': self.memory.last_active_events,
            'memory_edges': self.memory.特征到时间.条数() + self.memory.时间到特征.条数(),
            'pfc_edges': self.pfc.连接数(), 'danger_weights': self.danger_weights.tolist(),
            'audio_weights': self.audio_weights.tolist(), 'temporal_motor': temporal.tolist(),
            'episodic_motor': episodic.tolist()}
        return muscles, emitted, self.last_info

    def save(self, path):
        arrays = self._fixed()
        arrays.update(plastic_audio=self.audio_weights, plastic_echo=self.echo_weights,
                      plastic_danger=self.danger_weights)
        for name, table, width in [('pfc', self.pfc.联想, self.n),
                                   ('disinhibit', self.pfc.去抑, (self.n + 19) // 20)]:
            mat = np.zeros((self.n, width))
            for source, row in table.表.items():
                for target, weight in row.items():
                    mat[source, target] = weight
            arrays['plastic_' + name] = mat
        count = max(self.memory.next, default=-1) + 1
        arrays['mem_event_tick'] = np.array([self.memory.ticks.get(t, t) for t in range(count)], dtype=np.int64)
        arrays['mem_event_strength'] = np.array([self.memory.strength.get(t, 1.) for t in range(count)], dtype=float)
        arrays['mem_event_next'] = np.array([self.memory.next.get(t, -1) for t in range(count)], dtype=np.int64)
        sources, times, weights = [], [], []
        for source, row in self.memory.特征到时间.表.items():
            for event, weight in row.items():
                sources.append(source); times.append(event); weights.append(weight)
        for event, row in self.memory.时间到特征.表.items():
            for target, weight in row.items():
                sources.append(self.n + target); times.append(event); weights.append(weight)
        arrays['mem_edge_source'] = np.array(sources, dtype=np.int64)
        arrays['mem_edge_time'] = np.array(times, dtype=np.int64)
        arrays['mem_edge_weight'] = np.array(weights, dtype=float)
        arrays.update(state_previous=self.previous, state_memory_active=self.memory.激活,
            state_color_adaptation=self.color_adaptation,
            state_audio_trace=self.auditory_trace, state_proximity_trace=self.proximity_trace,
            state_numbers=np.array([self.frame, self.last_action, self.pain_trace, self.echo_cooldown,
                                   self.clock.当前时间, self.memory_base, self.pfc.强度], dtype=float))
        arrays['state_maintenance'] = np.array([[t.维护次数, t.上次衰减维护] for t in
            (self.pfc.联想, self.pfc.去抑, self.memory.特征到时间, self.memory.时间到特征)], dtype=np.int64)
        metadata = {'schema_version': 1, 'birth_hash': self.birth_hash,
                    'feature_schema': {'channels': 33, 'levels': 10, 'n': self.n,
                    'tones': FREQUENCIES.tolist(), 'memory_direction': 'source>=n means time-to-feature'},
                    'model_config': {'seed': self.seed}, 'state_defaults': {'inhibition': .5}}
        save_checkpoint(path, arrays, metadata)

    @classmethod
    def load(cls, path):
        arrays, metadata = load_checkpoint(path)
        obj = cls(seed=metadata.get('model_config', {}).get('seed', 2026))
        if metadata['birth_hash'] != obj.birth_hash:
            raise ValueError('固定神经元布线不兼容，不能加载或融合')
        for key, value in obj._fixed().items():
            if key not in arrays or not np.array_equal(value, arrays[key]):
                raise ValueError('固定网络校验失败: ' + key)
        obj.audio_weights = arrays['plastic_audio'].copy()
        obj.echo_weights = arrays['plastic_echo'].copy()
        obj.danger_weights = arrays['plastic_danger'].copy()
        for key, table in [('pfc', obj.pfc.联想), ('disinhibit', obj.pfc.去抑)]:
            a = arrays['plastic_' + key]
            for source, target in zip(*np.nonzero(a)):
                table.学习(source, target, float(a[source, target]))
        for source, event, weight in zip(arrays['mem_edge_source'], arrays['mem_edge_time'], arrays['mem_edge_weight']):
            if source < obj.n:
                obj.memory.特征到时间.学习(source, event, float(weight))
            else:
                obj.memory.时间到特征.学习(event, source - obj.n, float(weight))
        count = len(arrays['mem_event_tick'])
        obj.memory.next = dict(enumerate(arrays['mem_event_next'].tolist()))
        obj.memory.ticks = dict(enumerate(arrays['mem_event_tick'].tolist()))
        obj.memory.strength = dict(enumerate(arrays['mem_event_strength'].tolist()))
        state = arrays.get('state_numbers', np.zeros(7))
        obj.frame, obj.last_action, obj.echo_cooldown = int(state[0]), int(state[1]), int(state[3])
        obj.pain_trace = float(state[2])
        obj.clock.时间激活[:] = False
        obj.clock.当前时间 = int(state[4]); obj.clock.时间激活[int(state[4])] = True
        obj.memory_base = int(state[5])
        for table, row in zip((obj.pfc.联想, obj.pfc.去抑, obj.memory.特征到时间,
                              obj.memory.时间到特征), arrays.get('state_maintenance', [])):
            table.维护次数, table.上次衰减维护 = map(int, row)
        obj.pfc.强度 = float(state[6]) or .5
        obj.previous = arrays.get('state_previous', obj.previous).copy()
        obj.memory.激活 = arrays.get('state_memory_active', obj.memory.激活).copy()
        obj.auditory_trace = arrays.get('state_audio_trace', obj.auditory_trace).copy()
        obj.proximity_trace = arrays.get('state_proximity_trace', obj.proximity_trace).copy()
        obj.color_adaptation = arrays.get('state_color_adaptation', obj.color_adaptation).copy()
        # 融合后各旧经历保留独立编号，新生活从另一个环开始。
        if count and obj.frame == 0:
            obj.memory_base = count
        return obj
