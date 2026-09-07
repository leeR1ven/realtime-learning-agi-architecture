"""双上下文路线切换控制器；不修改冻结的 associative_controller.py。

新增的出生结构是两个独立的阈值保持/互抑群：三个目标声音受体，以及
两个规则声音受体加一个“尚未给规则”的静息单元。它们共同参与通用
随机混合细胞的阈值计算。声音到颜色、方向、路线和肌肉的对应不预置，
只能由实际自主经历和外界成功奖励形成。

规则切换保留目标，并抑制此前正在回放的序列活动。实际经历链不改写：
每个事件永久保留当时的目标/规则活动。最终奖励可加强同回合、同目标
实际发生的前缀和后缀，不能把前缀重新标成后来才出现的规则。

观察/动作、局部记忆、两步序列传播和身体本能继承冻结控制器。这个
文件不读取环境目标、路线标签、坐标、世界朝向或地图。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from associative_controller import Controller as BaseController, SAMPLE_RATE


GOAL_FREQUENCIES = np.array([220., 440., 660.], dtype=np.float64)
RULE_FREQUENCIES = np.array([990., 1320.], dtype=np.float64)
FREQUENCIES = np.concatenate((GOAL_FREQUENCIES, RULE_FREQUENCIES))
ROUTE_SCHEMA_VERSION = 1
CONTROLLER_KIND = 'dual_context_route_switch'


def _five_frequency_receptors(waveform):
    wave = np.asarray(waveform, dtype=np.float64)
    if wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError('waveform 必须是有限值的一维 8 kHz PCM')
    if not wave.size:
        return np.zeros(5, dtype=np.float64)
    phase = 2 * np.pi * FREQUENCIES[:, None] * np.arange(wave.size) / SAMPLE_RATE
    amplitude = 2 * np.abs(np.exp(-1j * phase) @ wave) / wave.size
    # 五路相同出生增益，独立或同时出现的短音经过同一幅值处理。
    return np.clip(amplitude / .25, 0., 1.)


class RouteSwitchController(BaseController):
    """纯传感器双上下文控制器。

    goal_enabled/rule_enabled=False 将相应混合输入变为共同本底，去掉
    该群的身份区分，同时保留另一个群及视觉通路。它们用于独立消融，
    不在通常生活中切换。context_enabled 是旧接口对目标区分的别名。
    """

    def __init__(self, *, ray_count=31, seed=2026, mixed_neurons=8192,
                 max_events=100000, goal_enabled=True, rule_enabled=True,
                 **kwargs):
        self.goal_enabled = bool(goal_enabled)
        self.rule_enabled = bool(rule_enabled)
        self.rule_interventions = 0
        super().__init__(ray_count=ray_count, seed=seed,
                         mixed_neurons=mixed_neurons, max_events=max_events,
                         **kwargs)
        # 无任务标签的通用随机稀疏出生连接：目标、规则、两当前受体。
        self.context_source = self.rng.integers(0, 3, self.mixed_neurons, dtype=np.int32)
        self.rule_source = self.rng.integers(0, 3, self.mixed_neurons, dtype=np.int32)
        self.thresholds = np.full(self.mixed_neurons, 3.5, dtype=np.float32)
        self.birth_hash = self._birth_hash()

    def _birth_hash(self):
        # BaseController 在其构造中先计算一次身份；此时规则连接尚未出生。
        if not hasattr(self, 'rule_source'):
            return super()._birth_hash()
        digest = hashlib.sha256(b'associative-route-switch-dual-context-v1')
        for array in (FREQUENCIES, self.levels, self.context_source, self.rule_source,
                      self.view_sources, self.thresholds):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(super()._birth_hash().encode('ascii'))
        return digest.hexdigest()

    def reset_activity(self):
        super().reset_activity()
        # 第0个规则单元是未给规则的静息状态；不代表上/下或任何路线。
        self.rule_context = np.array([1., 0., 0.], dtype=np.float32)

    def _observe(self, packet):
        # 复用旧版严格的身体白名单解析，听觉在本文件独立解析五个频带。
        values, _ = super()._observe({**packet, 'waveform': ()})
        return values, _five_frequency_receptors(packet.get('waveform', ()))

    def _goal_signature(self):
        return int(np.dot(self.context[:3].astype(np.uint8),
                          np.array([1, 2, 4], dtype=np.uint8)))

    def _context_signature(self):
        goal = self._goal_signature()
        if not goal:
            return 0
        rule = int(np.dot(self.rule_context.astype(np.uint8),
                          np.array([1, 2, 4], dtype=np.uint8)))
        return goal | (rule << 3)

    def _clear_replay_activity(self):
        self.last_recalled_ids = np.empty(0, dtype=np.int64)
        self.last_recalled_weights = np.empty(0, dtype=np.float32)

    def _activate(self, values, audio):
        old_goal = self._goal_signature()
        old_rule = self.rule_context.copy()
        # 两组完全分开的固定竞争。规则声音不会进入目标群的抑制场。
        goal_audio = audio[:3]
        goal_current = (.85 * self.context[:3] + 1.7 * goal_audio
                        - .9 * float(goal_audio.max(initial=0.)))
        self.context[:3] = (goal_current >= .5).astype(np.float32)
        self.context[3] = 0.
        rule_audio = np.concatenate(([0.], audio[3:5]))
        rule_current = (.85 * self.rule_context + 1.9 * rule_audio
                        - 1.15 * float(rule_audio.max(initial=0.)))
        # 静息单元有通用本底兴奋，显式规则的活动抑制它。
        rule_current[0] += .55 * max(0., 1. - float(self.rule_context[1:].sum()))
        self.rule_context = (rule_current >= .5).astype(np.float32)
        if old_goal and self._goal_signature() != old_goal:
            self.recent_actions.clear()
            self.previous_event_id = -1
            self._clear_replay_activity()
        elif not np.array_equal(old_rule, self.rule_context):
            # 保留实际先后经历，只切断此刻旧规则的回放惯性。
            self._clear_replay_activity()
        receptors, active = self._mixed_activity(values)
        self.pfc_activity[:] = False
        self.pfc_activity[active] = True
        return receptors, active

    def _mixed_activity(self, values):
        positive = values[:, None] >= self.levels[None, :]
        receptors = np.stack((positive, ~positive), axis=-1).ravel()
        if not self.view_enabled:
            receptors[:] = False
        goal_input = (self.context[self.context_source]
                      if self.goal_enabled and self.context_enabled else 1.)
        rule_input = self.rule_context[self.rule_source] if self.rule_enabled else 1.
        current = goal_input + rule_input + receptors[self.view_sources].sum(axis=1)
        return receptors, np.flatnonzero(current >= self.thresholds).astype(np.int32)

    def intervene_rule_state(self, activity):
        """实验性神经状态干预：接受3维规则群活动，不接受路线/方向标签。

        后续静音保持该活动，后续真实规则音可通过通常感觉通路更新它。
        不改写已存经历，也不改变目标保持状态。
        """
        state = np.asarray(activity, dtype=np.float32)
        if state.shape != (3,) or not np.isfinite(state).all() or (state < 0).any() or (state > 1).any():
            raise ValueError('规则状态干预必须是三维0~1神经活动')
        self.rule_context = (state >= .5).astype(np.float32)
        self._clear_replay_activity()
        self.rule_interventions += 1

    def _reinforce(self, reward):
        if not np.isfinite(reward):
            raise ValueError('reward 必须有限')
        goal = self._goal_signature()
        if reward <= 0 or not goal:
            self.last_reward_info = {'strengthened_events': 0, 'reward': float(reward)}
            return 0
        newest_frame = max((int(self.event_frame[slot]) for slot, event_id in self.recent_actions
                            if self.event_ids[slot] == event_id), default=self.frame)
        strengthened = 0
        counts_by_signature = {}
        oldest_age = 0
        for slot, event_id in reversed(self.recent_actions):
            signature = int(self.event_context[slot])
            if self.event_ids[slot] != event_id or (signature & 7) != goal:
                continue
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
            counts_by_signature[str(signature)] = counts_by_signature.get(str(signature), 0) + 1
        self.last_reward_info = dict(strengthened_events=strengthened, reward=float(reward),
            goal_activity_signature=goal, oldest_age_frames=oldest_age,
            preserved_context_counts=counts_by_signature)
        return strengthened

    def observe_then_act(self, packet, teacher_muscles=None, learning=True):
        prediction, info = super().observe_then_act(packet, teacher_muscles=teacher_muscles,
                                                  learning=learning)
        info.update(goal_active=np.flatnonzero(self.context[:3]).tolist(),
                    rule_active=np.flatnonzero(self.rule_context).tolist(),
                    rule_currents=self.rule_context.tolist(),
                    goal_enabled=self.goal_enabled, rule_enabled=self.rule_enabled,
                    context_signature=self._context_signature(),
                    rule_interventions=self.rule_interventions,
                    route_schema_version=ROUTE_SCHEMA_VERSION)
        self.last_info = info
        return prediction, info

    def save(self, path):
        """原子保存基类完整状态与新增规则状态；不修改任何旧版存档。"""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        base_temp = None
        output_temp = None
        try:
            fd, base_temp = tempfile.mkstemp(prefix='.route-base-', suffix='.npz', dir=destination.parent)
            os.close(fd)
            super().save(base_temp)
            with np.load(base_temp, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            metadata = json.loads(arrays['metadata'].tobytes().decode('utf-8'))
            metadata.update(controller_kind=CONTROLLER_KIND, route_schema_version=ROUTE_SCHEMA_VERSION,
                            goal_enabled=self.goal_enabled, rule_enabled=self.rule_enabled,
                            rule_interventions=self.rule_interventions)
            arrays['metadata'] = np.frombuffer(json.dumps(metadata, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
            arrays['rule_source'] = self.rule_source
            arrays['rule_context'] = self.rule_context
            fd, output_temp = tempfile.mkstemp(prefix='.' + destination.name, suffix='.tmp', dir=destination.parent)
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(output_temp, destination)
            output_temp = None
        finally:
            for temporary in (base_temp, output_temp):
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        # 先拒绝旧单上下文存档，不尝试对旧权重重新解释或迁移。
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(archive['metadata'].tobytes().decode('utf-8'))
            if metadata.get('controller_kind') != CONTROLLER_KIND or metadata.get('route_schema_version') != ROUTE_SCHEMA_VERSION:
                raise ValueError('需要本双上下文控制器的新出生存档；不重解释旧模型')
            rule_source = archive['rule_source'].copy()
            rule_context = archive['rule_context'].copy()
        # 基类 classmethod 使用 cls 构造，故会重新创建相同双上下文出生投射。
        obj = super().load(path)
        if not np.array_equal(rule_source, obj.rule_source):
            raise ValueError('规则群出生连接身份不一致')
        if rule_context.shape != (3,) or not np.isfinite(rule_context).all() or not np.isin(rule_context, [0., 1.]).all():
            raise ValueError('规则群存档活动无效')
        obj.rule_context = rule_context.astype(np.float32, copy=True)
        obj.goal_enabled = bool(metadata['goal_enabled'])
        obj.rule_enabled = bool(metadata['rule_enabled'])
        obj.rule_interventions = int(metadata['rule_interventions'])
        return obj


AssociativeController = RouteSwitchController
Controller = RouteSwitchController
