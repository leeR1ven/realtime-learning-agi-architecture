"""色标导航的局部联想控制器。

这是原固定受体、阈值放电、多簇联想、时间经历及局部 Hebb 思想的
具体导航实现，并非直接运行旧前额叶类得到的成绩。脑只接收身体感觉
和 PCM；没有目标编号、位置、朝向、地图、教师路线或环境对象标签。

固定稀疏混合细胞随机接收一个听觉保持细胞及两个视觉/身体受体。
三路同时过阈值才放电；没有按某个声音对应某种颜色布置连接。
放电经特征→时间事件连接，同时唤起多个经历，再由时间→肌肉连接
返回实际肌肉电流。教师动作只写入经历，不替代本帧自主预测。

自主经历也会记录，但其初始教学强度为零，只有之后到达的外界正奖励
才能增强最近真实动作。这样不会把随意做出的动作当作正确答案自教。
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FREQUENCIES = np.array([220., 440., 660., 880.])
SAMPLE_RATE = 8000
RAYS = 31
SCHEMA_VERSION = 2
# 只用于出生时的自发探索与诊断，不包含声音/颜色映射。
MOTOR_TEMPLATES = np.array([[0., 0., 0., 0.], [.6, 0., .6, 0.],
                            [0., .5, 0., .5], [0., .3, .3, 0.],
                            [.3, 0., 0., .3]], dtype=np.float32)
ACTION_NAMES = ('静息', '前进', '后退', '左转', '右转')


def _sound_features(waveform):
    wave = np.asarray(waveform, dtype=np.float64)
    if wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError('waveform 必须是有限值的一维 8 kHz PCM')
    if not wave.size:
        return np.zeros(4, dtype=np.float64)
    phases = 2 * np.pi * FREQUENCIES[:, None] * np.arange(wave.size) / SAMPLE_RATE
    amplitudes = 2 * np.abs(np.exp(-1j * phases) @ wave) / wave.size
    return np.clip(amplitudes / .5, 0., 1.)


def _muscles(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError('肌肉必须是四维有限值数组')
    if (result < 0).any() or (result > 1).any():
        raise ValueError('肌肉电流范围必须是 0~1')
    return result.copy()


class Controller:
    """observe_then_act(packet, teacher_muscles=None, learning=True).

    packet 的 observation 包含 31 束深度/RGB、身体前/左速度、角速度、
    pain 和 touch；顶层 waveform 是 PCM，reward 是可选的真实外界反馈。
    返回 (四肌肉电流, 可审计诊断)。教学时调用者应实际执行所提供的
    teacher_muscles；非教学时应执行返回的预测肌肉。

    三个独立消融只切对应通路；无隐藏的任务答案或阶段参数。
    context_enabled=False 清除听觉保持；view_enabled=False 切视觉/身体
    受体传入；motor_memory_enabled=False 切时间记忆→肌肉返回。
    """

    def __init__(self, *, ray_count=31, seed=2026, mixed_neurons=4096, max_events=50000,
                 context_enabled=True, view_enabled=True,
                 motor_memory_enabled=True, exploration=True,
                 max_recalled_events=16, recall_feedback_enabled=True,
                 sequence_enabled=True, receptor_levels=16,
                 reward_trace_length=4096, reward_trace_decay=.9995,
                 exploration_rate=.015, reflex_enabled=True):
        if ray_count != RAYS:
            raise ValueError('当前出生受体布局要求 ray_count=31')
        if mixed_neurons < 64 or max_events < 1 or max_recalled_events < 1:
            raise ValueError('mixed_neurons>=64，max_events/max_recalled_events>=1')
        self.seed = int(seed)
        self.mixed_neurons = int(mixed_neurons)
        self.max_events = int(max_events)
        self.max_recalled_events = int(max_recalled_events)
        self.context_enabled = bool(context_enabled)
        self.view_enabled = bool(view_enabled)
        self.motor_memory_enabled = bool(motor_memory_enabled)
        self.exploration = bool(exploration)
        self.recall_feedback_enabled = bool(recall_feedback_enabled)
        self.sequence_enabled = bool(sequence_enabled)
        self.receptor_levels = int(receptor_levels)
        self.reward_trace_length = int(reward_trace_length)
        self.reward_trace_decay = float(reward_trace_decay)
        self.exploration_rate = float(exploration_rate)
        self.reflex_enabled = bool(reflex_enabled)
        if not 2 <= self.receptor_levels <= 64 or not 1 <= self.reward_trace_length <= 4096:
            raise ValueError('受体级数须为2~64，奖励链长度须为1~4096')
        if not 0 < self.reward_trace_decay <= 1 or not 0 <= self.exploration_rate <= 1:
            raise ValueError('奖励衰减须为(0,1]，探索率须为[0,1]')
        self.rng = np.random.default_rng(self.seed)
        # 31 深度 + 31*3 RGB + 身体前/左速度 + 角速度 + 痛 + 触。
        self.sensory_channels = RAYS * 4 + 5
        self.levels = np.linspace(1 / (self.receptor_levels + 1),
                                  self.receptor_levels / (self.receptor_levels + 1),
                                  self.receptor_levels, dtype=np.float32)
        self.receptor_count = self.sensory_channels * len(self.levels) * 2
        # 通用跨模态稀疏出生连接，随机种子在教学开始前确定。
        self.context_source = self.rng.integers(0, 4, self.mixed_neurons, dtype=np.int32)
        self.view_sources = self.rng.integers(0, self.receptor_count,
                                            (self.mixed_neurons, 2), dtype=np.int32)
        self.thresholds = np.full(self.mixed_neurons, 2.5, dtype=np.float32)
        self.birth_hash = self._birth_hash()
        # 每个混合细胞的出连接指向时间槽；只沿当前活动细胞传播。
        self.feature_to_events = [set() for _ in range(self.mixed_neurons)]
        # 相同连接的奖励门控索引。原始经历连接全部保留；未增强的动作
        # 不在肌肉读出时逐条遍历。缓存只是执行索引，不是新增学习规则。
        self.readout_feature_to_events = [set() for _ in range(self.mixed_neurons)]
        self.readout_cache = [None] * self.mixed_neurons
        self.event_features = [None] * self.max_events
        self.event_muscles = np.zeros((self.max_events, 4), dtype=np.float32)
        self.event_views = np.zeros((self.max_events, self.sensory_channels), dtype=np.float32)
        self.event_strength = np.zeros(self.max_events, dtype=np.float32)
        self.event_kind = np.zeros(self.max_events, dtype=np.uint8)  # 1 自主，2 教师
        self.event_ids = np.full(self.max_events, -1, dtype=np.int64)
        self.event_previous = np.full(self.max_events, -1, dtype=np.int64)
        self.event_next = np.full(self.max_events, -1, dtype=np.int64)
        self.event_context = np.zeros(self.max_events, dtype=np.uint8)
        self.event_frame = np.zeros(self.max_events, dtype=np.int64)
        self.event_active_count = np.ones(self.max_events, dtype=np.float32)
        self.event_slot_by_id = {}
        self.next_slot = 0
        self.next_event_id = 0
        self.frame = 0
        self.sequence_graph_shuffled = False
        self.reset_activity()

    def _birth_hash(self):
        digest = hashlib.sha256(b'associative-color-nav-threshold-v2')
        for array in (FREQUENCIES, self.levels, self.context_source,
                      self.view_sources, self.thresholds, MOTOR_TEMPLATES):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def reset_activity(self):
        """清除短时上下文和活动，保留已学习的全部突触/经历。"""
        self.context = np.zeros(4, dtype=np.float32)
        self.pfc_activity = np.zeros(self.mixed_neurons, dtype=bool)
        self.recent_actions = deque(maxlen=self.reward_trace_length)
        self.previous_event_id = -1
        self.exploration_left = 0
        self.exploration_action = 0
        self.exploration_burst = 0
        self.escape_back_steps = 0
        self.escape_turn_steps = 0
        self.escape_action = 3
        self.last_recalled_ids = np.empty(0, dtype=np.int64)
        self.last_recalled_weights = np.empty(0, dtype=np.float32)
        self.last_reward_info = {'strengthened_events': 0, 'reward': 0.}
        self.skipped_unconfirmed_events = 0
        self.last_info = {}

    def _observe(self, packet):
        if not isinstance(packet, dict) or 'observation' not in packet:
            raise ValueError('packet 必须包含 observation')
        observation = packet['observation']
        distances = np.asarray(observation['ray_distances'], dtype=np.float32)
        colors = np.asarray(observation['ray_colors'], dtype=np.float32)
        velocity = np.asarray(observation['velocity'], dtype=np.float32)
        if distances.shape != (RAYS,) or colors.shape != (RAYS, 3) or velocity.shape != (2,):
            raise ValueError('需要 31 束深度、31x3 RGB 与身体前/左二维速度')
        if not all(np.isfinite(x).all() for x in (distances, colors, velocity)):
            raise ValueError('身体观察不能包含非有限值')
        if (distances < 0).any() or (colors < 0).any() or (colors > 1).any():
            raise ValueError('距离必须非负，RGB 必须已经归一化到 0~1')
        angular = float(observation['angular_velocity'])
        pain = float(observation.get('pain', 0.))
        touch = float(np.max(np.asarray(observation.get('touch', 0.))))
        if not np.isfinite([angular, pain, touch]).all():
            raise ValueError('角速度/痛/触必须有限')
        values = np.concatenate((np.exp(-distances / 2.), colors.ravel(),
                    np.clip((velocity + 3.) / 6., 0., 1.),
                    [np.clip((angular + 4.) / 8., 0., 1.),
                     np.clip(pain, 0., 1.), np.clip(touch, 0., 1.)])).astype(np.float32)
        audio = _sound_features(packet.get('waveform', ()))
        return values, audio

    def _activate(self, values, audio):
        # 固定自反馈 .85 保持过阈值的声音簇；新声音经共同抑制竞争更新。
        # 静音时无额外外界信号，活动仍由自身循环输入维持。
        old_context = self._context_signature()
        if self.context_enabled:
            current = .85 * self.context + 1.7 * audio - .9 * float(audio.max(initial=0.))
            self.context = (current >= .5).astype(np.float32)
        else:
            self.context[:] = 0
        if old_context and self._context_signature() != old_context:
            # 新声音语境开始一条新经历链，成功不能倒灌给此前另一语境。
            self.recent_actions.clear()
            self.previous_event_id = -1
            self.last_recalled_ids = np.empty(0, dtype=np.int64)
            self.last_recalled_weights = np.empty(0, dtype=np.float32)
        receptors, active = self._mixed_activity(values)
        self.pfc_activity[:] = False
        self.pfc_activity[active] = True
        return receptors, active

    def _context_signature(self):
        return int(np.dot(self.context.astype(np.uint8), np.array([1, 2, 4, 8], dtype=np.uint8)))

    def _mixed_activity(self, values):
        """现实视感与回忆视感共享同一出生投射；不更改保持语境。"""
        positive = values[:, None] >= self.levels[None, :]
        receptors = np.stack((positive, ~positive), axis=-1).ravel()
        if not self.view_enabled:
            receptors[:] = False
        # 阈值 AND 由三条固定兴奋突触自然计算，不调用颜色/目标分类器。
        currents = self.context[self.context_source] + receptors[self.view_sources].sum(axis=1)
        return receptors, np.flatnonzero(currents >= self.thresholds).astype(np.int32)

    def _activity_votes(self, active):
        scores = np.zeros(self.max_events, dtype=np.float32)
        # 高频、到处出现的受体通过出连接的分流获得较小的单事件电流。
        # 这是通用连接度归一化；不使用距离、坐标或环境标签来选经历。
        for feature in active:
            feature = int(feature)
            row = self.readout_feature_to_events[feature]
            if row:
                slots = self.readout_cache[feature]
                if slots is None:
                    slots = np.fromiter(row, dtype=np.int32, count=len(row))
                    self.readout_cache[feature] = slots
                scores[slots] += 1. / np.sqrt(len(row))
        scores /= np.sqrt(self.event_active_count)
        scores[self.event_strength <= 0] = 0
        return scores

    def _select_events(self, scores):
        peak = float(scores.max(initial=0.))
        if peak <= 0:
            return np.empty(0, dtype=np.int32), np.empty(0), np.zeros(4), None, 0.
        # 多个强联想同时放电；抑制弱联想，不把记忆压成单赢家。
        selected = np.flatnonzero(scores >= peak * .96)
        if len(selected) > self.max_recalled_events:
            rank = np.argpartition(scores[selected], -self.max_recalled_events)[-self.max_recalled_events:]
            selected = selected[rank]
        weights = (scores[selected] / peak) ** 8 * self.event_strength[selected]
        weights = weights / max(float(weights.sum()), 1e-12)
        muscle_current = weights @ self.event_muscles[selected]
        recalled_view = weights @ self.event_views[selected]
        return selected, weights, muscle_current, recalled_view, peak

    def _recall(self, active):
        primary_scores = self._activity_votes(active)
        initial = self._select_events(primary_scores)
        first_slots, first_weights, _, remembered_view, primary_peak = initial
        feedback_active = np.empty(0, dtype=np.int32)
        sequence_supported = []
        if not primary_peak:
            return (*initial, feedback_active, sequence_supported)
        scores = primary_scores.copy()
        # 内部回忆只能影响本帧真实感觉已支持的经历，不能凭回放走开环路线。
        supported = primary_scores >= .80 * primary_peak
        if self.recall_feedback_enabled and remembered_view is not None:
            _, feedback_active = self._mixed_activity(remembered_view)
            feedback_scores = self._activity_votes(feedback_active)
            feedback_peak = float(feedback_scores.max(initial=0.))
            if feedback_peak > 0:
                scores[supported] += .08 * primary_peak * feedback_scores[supported] / feedback_peak
        if self.sequence_enabled:
            # 两个固定神经传播微步，只沿实际学到的事件转换连接。
            # 直接后继优先，第二步电流较弱；不是按房间/路线设置的阶段机。
            field = np.zeros(self.max_events, dtype=np.float32)
            frontier = list(zip(self.last_recalled_ids, self.last_recalled_weights))
            for hop_gain in (1., .25):
                following = []
                for event_id, weight in frontier:
                    old_slot = self.event_slot_by_id.get(int(event_id))
                    if old_slot is None:
                        continue
                    next_id = int(self.event_next[old_slot])
                    slot = self.event_slot_by_id.get(next_id)
                    if slot is not None and self.event_strength[slot] > 0:
                        field[slot] += hop_gain * float(weight)
                        following.append((next_id, weight))
                frontier = following
            field[~supported] = 0
            field_peak = float(field.max(initial=0.))
            if field_peak > 0:
                # 分流归一化避免同时回忆16次相似经历就把顺序信号稀释16倍。
                scores += .35 * primary_peak * field / field_peak
                sequence_supported = self.event_ids[field > 0].tolist()
        scores[~supported] = 0
        return (*self._select_events(scores), feedback_active, sequence_supported)

    def shuffle_sequence_connections(self, seed=0):
        """仅供评估副本：语境内打乱有向转换边，保持事件、感觉和权重完整。

        不修改父存档；调用者应对 load 得到的对照副本调用。保留末端
        边界及每个目标的入度，只改变先后配对。event_previous 是原始
        经历审计记录，故不改写；实际传播读取 event_next。
        """
        rng = np.random.default_rng(int(seed))
        changed = 0
        for signature in np.unique(self.event_context[self.event_ids >= 0]):
            sources = np.flatnonzero((self.event_ids >= 0) & (self.event_context == signature) & (self.event_next >= 0))
            targets = self.event_next[sources].copy()
            rng.shuffle(targets)
            changed += int(np.count_nonzero(targets != self.event_next[sources]))
            self.event_next[sources] = targets
        self.last_recalled_ids = np.empty(0, dtype=np.int64)
        self.last_recalled_weights = np.empty(0, dtype=np.float32)
        self.sequence_graph_shuffled = True
        return changed

    sequence_graph_shuffle = shuffle_sequence_connections

    def _explore(self):
        if not self.exploration:
            return np.zeros(4, dtype=np.float32)
        if self.exploration_left <= 0:
            self.exploration_action = int(self.rng.choice(5, p=[.03, .57, .06, .17, .17]))
            self.exploration_left = int(self.rng.integers(8, 23))
        self.exploration_left -= 1
        return MOTOR_TEMPLATES[self.exploration_action].copy()

    def _body_reflex(self, values):
        """疼痛/接触与近距离退缩，不接收目标颜色或世界方向。"""
        if not self.reflex_enabled:
            return None
        front_near = float(values[12:19].max(initial=0.))
        touched = max(float(values[-2]), float(values[-1])) > .02
        if (touched or front_near > .90) and self.escape_back_steps + self.escape_turn_steps == 0:
            self.escape_back_steps = int(self.rng.integers(5, 10))
            self.escape_turn_steps = int(self.rng.integers(9, 18))
            right_near, left_near = float(values[:12].mean()), float(values[19:31].mean())
            if abs(left_near - right_near) > .05:
                self.escape_action = 4 if left_near > right_near else 3
            else:
                self.escape_action = int(self.rng.choice([3, 4]))
        if self.escape_back_steps:
            self.escape_back_steps -= 1
            return MOTOR_TEMPLATES[2].copy()
        if self.escape_turn_steps:
            self.escape_turn_steps -= 1
            return MOTOR_TEMPLATES[self.escape_action].copy()
        return None

    def _reinforce(self, reward):
        if not np.isfinite(reward):
            raise ValueError('reward 必须有限')
        if reward <= 0:
            self.last_reward_info = {'strengthened_events': 0, 'reward': float(reward)}
            return 0
        # 奖励回溯本轮真实经历链；reset 或声音语境变化切断链。
        # 保留最多4096帧，.9995/帧使较早发生的相关行为仍可获得信用。
        signature = self._context_signature()
        strengthened = 0
        newest_frame = max((int(self.event_frame[slot]) for slot, event_id in self.recent_actions
                            if self.event_ids[slot] == event_id), default=self.frame)
        oldest_age = 0
        for age, (slot, event_id) in enumerate(reversed(self.recent_actions)):
            if self.event_ids[slot] == event_id and self.event_context[slot] == signature and signature:
                age = max(0, newest_frame - int(self.event_frame[slot]))
                gain = min(float(reward), 1.) * self.reward_trace_decay ** age
                was_silent = self.event_strength[slot] <= 0
                self.event_strength[slot] += gain * (1. - self.event_strength[slot])
                if was_silent:
                    for feature in self.event_features[slot]:
                        feature = int(feature)
                        self.readout_feature_to_events[feature].add(slot)
                        self.readout_cache[feature] = None
                strengthened += 1
                oldest_age = max(oldest_age, age)
        self.last_reward_info = {'strengthened_events': strengthened, 'reward': float(reward),
                                'context_signature': signature, 'oldest_age_frames': oldest_age}
        return strengthened

    def receive_reward(self, reward: float):
        """env.step 已实际执行动作后调用；只增强已有经历，不生成/记录动作。"""
        return self._reinforce(float(reward))

    def _remember(self, active, values, actual_muscles, taught):
        slot = self.next_slot
        if not taught and self.event_strength[slot] > 0:
            # 未经外界确认的自行动作不能仅因生活时间更长就覆盖教学技能。
            # 先复用未增强的经历槽；全满时保留已确认经历，等待新教学。
            available = np.flatnonzero(self.event_strength <= 0)
            if not len(available):
                self.skipped_unconfirmed_events += 1
                return
            following = available[available >= slot]
            slot = int(following[0] if len(following) else available[0])
        old = self.event_features[slot]
        if old is not None:
            self.event_slot_by_id.pop(int(self.event_ids[slot]), None)
            for feature in old:
                self.feature_to_events[int(feature)].discard(slot)
                self.readout_feature_to_events[int(feature)].discard(slot)
                self.readout_cache[int(feature)] = None
        self.event_features[slot] = active.copy()
        for feature in active:
            self.feature_to_events[int(feature)].add(slot)
            if taught:
                self.readout_feature_to_events[int(feature)].add(slot)
                self.readout_cache[int(feature)] = None
        self.event_muscles[slot] = actual_muscles
        self.event_views[slot] = values
        self.event_strength[slot] = 1. if taught else 0.
        self.event_kind[slot] = 2 if taught else 1
        self.event_ids[slot] = self.next_event_id
        self.event_previous[slot] = self.previous_event_id
        self.event_next[slot] = -1
        self.event_context[slot] = self._context_signature()
        self.event_frame[slot] = self.frame
        self.event_active_count[slot] = max(len(active), 1)
        previous_slot = self.event_slot_by_id.get(self.previous_event_id)
        if previous_slot is not None:
            self.event_next[previous_slot] = self.next_event_id
        self.event_slot_by_id[self.next_event_id] = slot
        self.recent_actions.append((slot, self.next_event_id))
        self.previous_event_id = self.next_event_id
        self.next_event_id += 1
        self.next_slot = (slot + 1) % self.max_events

    def observe_then_act(self, packet, teacher_muscles=None, learning=True):
        values, audio = self._observe(packet)
        teacher = None if teacher_muscles is None else _muscles(teacher_muscles)
        if learning:
            self._reinforce(float(packet.get('reward', 0.)))
        receptors, active = self._activate(values, audio)
        selected, weights, memory_current, recalled_view, peak, feedback_active, sequence_supported = self._recall(active)
        reflex_muscles = self._body_reflex(values)
        memory_available = self.motor_memory_enabled and bool(selected.size)
        if self.exploration and memory_available and self.exploration_burst <= 0 and self.rng.random() < self.exploration_rate:
            self.exploration_burst = int(self.rng.integers(5, 10))
            self.exploration_left = 0
        if reflex_muscles is not None:
            prediction = reflex_muscles
            source = 'body_reflex'
        elif self.exploration_burst > 0:
            self.exploration_burst -= 1
            prediction = self._explore()
            source = 'persistent_exploration'
        elif memory_available:
            prediction = np.clip(memory_current, 0., 1.).astype(np.float32)
            source = 'associative_memory'
        else:
            prediction = self._explore()
            source = 'spontaneous_exploration' if self.exploration else 'rest'
        self.last_recalled_ids = self.event_ids[selected].copy()
        self.last_recalled_weights = np.asarray(weights, dtype=np.float32).copy()
        # 先独立预测、后可选写经验。教学动作绝不替换本帧预测值。
        nearest = int(np.argmin(((MOTOR_TEMPLATES - prediction) ** 2).sum(axis=1)))
        info = {
            'frame': self.frame,
            'heard': audio.tolist(),
            'context_active': np.flatnonzero(self.context).tolist(),
            'context_currents': self.context.tolist(),
            'sensory_active': int(receptors.sum()),
            'pfc_active': active.tolist(),
            'pfc_active_count': len(active),
            'thought_active': len(active),
            'feedback_pfc_active': feedback_active.tolist(),
            'feedback_pfc_active_count': int(len(feedback_active)),
            'sequence_supported_events': sequence_supported,
            'recall_events': self.event_ids[selected].tolist(),
            'recall_count': int(len(selected)),
            'recall_weights': weights.tolist(),
            'recall_peak': peak,
            'recalled_view': None if recalled_view is None else recalled_view.tolist(),
            'muscle_currents': np.asarray(memory_current).tolist(),
            'predicted_muscles': prediction.tolist(),
            'muscles': prediction.tolist(),
            'action': nearest,
            'action_name': ACTION_NAMES[nearest],
            'source': source,
            'teaching': teacher is not None,
            'memory_events': int(np.count_nonzero(self.event_ids >= 0)),
            'trusted_events': int(np.count_nonzero(self.event_strength > 0)),
            'context_enabled': self.context_enabled,
            'view_enabled': self.view_enabled,
            'motor_memory_enabled': self.motor_memory_enabled,
            'recall_feedback_enabled': self.recall_feedback_enabled,
            'sequence_enabled': self.sequence_enabled,
            'sequence_graph_shuffled': self.sequence_graph_shuffled,
            'reflex_enabled': self.reflex_enabled,
            'reward_trace_events': len(self.recent_actions),
            'last_reward': dict(self.last_reward_info),
            'skipped_unconfirmed_events': self.skipped_unconfirmed_events,
        }
        if learning:
            self._remember(active, values, prediction if teacher is None else teacher,
                           teacher is not None)
        self.frame += 1
        self.last_info = info
        return prediction.copy(), info

    def save(self, path):
        """稀疏事件数组原子存档，不展开 n*n PFC 矩阵，不使用 pickle。"""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        counts = np.array([0 if item is None else len(item) for item in self.event_features], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        nonempty = [item for item in self.event_features if item is not None and len(item)]
        features = np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.int32)
        metadata = dict(schema_version=SCHEMA_VERSION, seed=self.seed,
            mixed_neurons=self.mixed_neurons, max_events=self.max_events,
            max_recalled_events=self.max_recalled_events, birth_hash=self.birth_hash,
            next_slot=self.next_slot, next_event_id=self.next_event_id, frame=self.frame,
            previous_event_id=self.previous_event_id,
            exploration_left=self.exploration_left, exploration_action=self.exploration_action,
            exploration_burst=self.exploration_burst,
            escape_back_steps=self.escape_back_steps, escape_turn_steps=self.escape_turn_steps,
            escape_action=self.escape_action,
            context_enabled=self.context_enabled, view_enabled=self.view_enabled,
            motor_memory_enabled=self.motor_memory_enabled, exploration=self.exploration,
            recall_feedback_enabled=self.recall_feedback_enabled, sequence_enabled=self.sequence_enabled,
            receptor_levels=self.receptor_levels, reward_trace_length=self.reward_trace_length,
            reward_trace_decay=self.reward_trace_decay, exploration_rate=self.exploration_rate,
            reflex_enabled=self.reflex_enabled,
            skipped_unconfirmed_events=self.skipped_unconfirmed_events,
            last_reward_info=self.last_reward_info,
            sequence_graph_shuffled=self.sequence_graph_shuffled,
            rng_state=self.rng.bit_generator.state)
        arrays = dict(metadata=np.frombuffer(json.dumps(metadata, ensure_ascii=False).encode('utf-8'), dtype=np.uint8),
            context_source=self.context_source, view_sources=self.view_sources,
            thresholds=self.thresholds, feature_offsets=offsets, features=features,
            event_muscles=self.event_muscles, event_views=self.event_views,
            event_strength=self.event_strength, event_kind=self.event_kind,
            event_ids=self.event_ids, event_previous=self.event_previous,
            event_next=self.event_next, event_context=self.event_context, event_frame=self.event_frame,
            event_active_count=self.event_active_count, context=self.context,
            pfc_activity=self.pfc_activity,
            last_recalled_ids=self.last_recalled_ids, last_recalled_weights=self.last_recalled_weights,
            recent_actions=np.asarray(list(self.recent_actions), dtype=np.int64).reshape(-1, 2))
        temporary = None
        try:
            handle, temporary = tempfile.mkstemp(prefix='.' + destination.name, suffix='.tmp', dir=destination.parent)
            with os.fdopen(handle, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            meta = json.loads(archive['metadata'].tobytes().decode('utf-8'))
            if meta.get('schema_version') != SCHEMA_VERSION:
                raise ValueError('不支持的导航存档版本')
            obj = cls(**{key: meta[key] for key in ('seed', 'mixed_neurons', 'max_events',
                'max_recalled_events', 'context_enabled', 'view_enabled', 'motor_memory_enabled', 'exploration',
                'recall_feedback_enabled', 'sequence_enabled', 'receptor_levels', 'reward_trace_length',
                'reward_trace_decay', 'exploration_rate', 'reflex_enabled')})
            for name in ('context_source', 'view_sources', 'thresholds'):
                if not np.array_equal(getattr(obj, name), archive[name]):
                    raise ValueError('出生固定连接不兼容: ' + name)
            if obj.birth_hash != meta['birth_hash']:
                raise ValueError('出生特征身份校验失败')
            for name in ('event_muscles', 'event_views', 'event_strength', 'event_kind',
                         'event_ids', 'event_previous', 'event_next', 'event_context', 'event_frame',
                         'event_active_count', 'context', 'pfc_activity'):
                value = archive[name]
                if value.shape != getattr(obj, name).shape or not np.isfinite(value).all():
                    raise ValueError('存档数组无效: ' + name)
                setattr(obj, name, value.copy())
            features = archive['features']
            offsets = archive['feature_offsets']
            if offsets.shape != (obj.max_events + 1,) or offsets[0] != 0 or offsets[-1] != len(features):
                raise ValueError('稀疏特征偏移无效')
            if (np.diff(offsets) < 0).any() or (features < 0).any() or (features >= obj.mixed_neurons).any():
                raise ValueError('稀疏特征编号无效')
            for slot in np.flatnonzero(obj.event_ids >= 0):
                active = features[offsets[slot]:offsets[slot + 1]].astype(np.int32, copy=True)
                obj.event_features[int(slot)] = active
                obj.event_slot_by_id[int(obj.event_ids[slot])] = int(slot)
                for feature in active:
                    obj.feature_to_events[int(feature)].add(int(slot))
                    if obj.event_strength[slot] > 0:
                        obj.readout_feature_to_events[int(feature)].add(int(slot))
            obj.recent_actions = deque((tuple(map(int, row)) for row in archive['recent_actions']),
                                       maxlen=obj.reward_trace_length)
            obj.last_recalled_ids = archive['last_recalled_ids'].copy()
            obj.last_recalled_weights = archive['last_recalled_weights'].copy()
            obj.last_reward_info = dict(meta['last_reward_info'])
            obj.sequence_graph_shuffled = bool(meta['sequence_graph_shuffled'])
            for name in ('next_slot', 'next_event_id', 'frame', 'previous_event_id',
                         'exploration_left', 'exploration_action', 'exploration_burst',
                         'escape_back_steps', 'escape_turn_steps', 'escape_action',
                         'skipped_unconfirmed_events'):
                setattr(obj, name, int(meta[name]))
            obj.rng.bit_generator.state = meta['rng_state']
            return obj


AssociativeController = Controller
